"""
tests/temporal/test_historical_state.py — Tests for HistoricalStateEngine.

Covers:
  - SnapshotDelta model (immutability, merge, properties)
  - StatePatch model (apply_to_state, merged_delta)
  - HistoricalStateEngine: get_state_at, compute_delta, compute_range
  - HistoricalStateEngine: build_patch, compare_states, list_subjects
  - Tenant isolation in all engine methods
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from pydantic import ValidationError as PydanticValidationError

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.fact_store import FactStore, TemporalFact
from apex_rag.temporal.historical_state import HistoricalStateEngine
from apex_rag.temporal.snapshot_models import SnapshotDelta, StatePatch

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage() -> AsyncGenerator[ApexStorage, None]:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        storage = await ApexStorage.create(f"sqlite+aiosqlite:///{tmp.name}")
    yield storage


@pytest_asyncio.fixture
async def fact_store(storage: ApexStorage) -> FactStore:
    return FactStore(storage)


@pytest_asyncio.fixture
async def engine(fact_store: FactStore, storage: ApexStorage) -> HistoricalStateEngine:
    return HistoricalStateEngine(fact_store, storage)


@pytest_asyncio.fixture
async def seed_facts(fact_store: FactStore) -> dict[str, TemporalFact]:
    """Seed facts with known temporal windows for delta computation."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    facts = {
        "revenue_q1": TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            confidence=0.9,
            source_document_id="doc-123",
            valid_from=base,
            valid_to=datetime(2025, 4, 1, tzinfo=timezone.utc),
        ),
        "revenue_q2": TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$50M",
            confidence=0.9,
            source_document_id="doc-123",
            valid_from=datetime(2025, 4, 1, tzinfo=timezone.utc),
            valid_to=datetime(2025, 7, 1, tzinfo=timezone.utc),
        ),
        "headcount": TemporalFact(
            subject="Headcount",
            predicate="was",
            object="500",
            confidence=0.85,
            source_document_id="doc-123",
            valid_from=base,
            valid_to=None,
        ),
        "policy_x": TemporalFact(
            subject="Travel Policy",
            predicate="shall",
            object="Receipt required",
            confidence=0.8,
            source_document_id="doc-123",
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2025, 12, 31, tzinfo=timezone.utc),
        ),
    }
    saved = await fact_store.save_facts(list(facts.values()), tenant_context="tenant-a")
    result = {}
    for key, f in facts.items():
        for s in saved:
            if s.subject == f.subject and s.valid_from == f.valid_from:
                result[key] = s
                break
    return result


# ═══════════════════════════════════════════════════════════════════════
# SnapshotDelta Model (pure unit — no DB needed)
# ═══════════════════════════════════════════════════════════════════════


class TestSnapshotDeltaModel:
    """SnapshotDelta dataclass immutability and properties."""

    def test_default_delta_id_is_uuid4(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(base_as_of=t1, target_as_of=t2)
        uuid.UUID(delta.delta_id, version=4)

    def test_is_empty_true(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(base_as_of=t1, target_as_of=t2)
        assert delta.is_empty is True
        assert delta.change_count == 0

    def test_is_empty_false_when_added(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(
            base_as_of=t1,
            target_as_of=t2,
            added_fact_ids={"fact-1"},
        )
        assert delta.is_empty is False
        assert delta.change_count == 1

    def test_is_empty_false_when_removed(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(
            base_as_of=t1,
            target_as_of=t2,
            removed_fact_ids={"fact-old"},
        )
        assert delta.is_empty is False

    def test_change_count(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(
            base_as_of=t1,
            target_as_of=t2,
            added_fact_ids={"a", "b"},
            removed_fact_ids={"c"},
            modified_subjects={"Revenue": {"before": "$10", "after": "$20"}},
        )
        assert delta.change_count == 4  # 2 added + 1 removed + 1 modified

    def test_frozen_immutability(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(base_as_of=t1, target_as_of=t2)
        with pytest.raises((AttributeError, TypeError, PydanticValidationError)):
            delta.added_fact_ids = {"new"}  # type: ignore[misc]

    def test_time_span_seconds(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)  # 86400 seconds
        delta = SnapshotDelta(base_as_of=t1, target_as_of=t2)
        assert delta.time_span_seconds == 86400.0

    def test_merge_non_sequential_raises(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)
        d1 = SnapshotDelta(doc_id="doc-1", base_as_of=t1, target_as_of=t2)
        d2 = SnapshotDelta(doc_id="doc-1", base_as_of=t1, target_as_of=t3)  # same base!
        with pytest.raises(ValueError) as exc:
            d1.merge(d2)
        assert "sequential" in str(exc.value).lower()

    def test_merge_different_doc_raises(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)
        d1 = SnapshotDelta(doc_id="doc-1", base_as_of=t1, target_as_of=t2)
        d2 = SnapshotDelta(doc_id="doc-2", base_as_of=t2, target_as_of=t3)
        with pytest.raises(ValueError) as exc:
            d1.merge(d2)
        assert "doc_id" in str(exc.value).lower()


class TestSnapshotDeltaMerge:
    """SnapshotDelta.merge() correctly combines sequential deltas."""

    def test_merge_success(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)
        d1 = SnapshotDelta(
            doc_id="doc-1",
            base_as_of=t1,
            target_as_of=t2,
            added_fact_ids={"a"},
            removed_fact_ids={"b"},
        )
        d2 = SnapshotDelta(
            doc_id="doc-1",
            base_as_of=t2,
            target_as_of=t3,
            added_fact_ids={"c"},
            removed_fact_ids={"a"},  # a was added in d1, removed in d2
        )
        merged = d1.merge(d2)
        assert merged.base_as_of == t1
        assert merged.target_as_of == t3
        assert "a" not in merged.added_fact_ids  # added then removed = net zero
        assert "c" in merged.added_fact_ids
        assert "b" in merged.removed_fact_ids  # b was removed once


# ═══════════════════════════════════════════════════════════════════════
# StatePatch Model (pure unit — no DB needed)
# ═══════════════════════════════════════════════════════════════════════


class TestStatePatchModel:
    """StatePatch creation and apply_to_state."""

    def test_empty_patch_returns_state_unchanged(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        patch = StatePatch(doc_id="doc-1", base_as_of=t1, target_as_of=t2)
        state = {"Revenue": "$40M", "Headcount": 500}
        result = patch.apply_to_state(state)
        assert result == state

    def test_apply_additions(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(
            base_as_of=t1,
            target_as_of=t2,
            added_fact_ids={"fact-new"},
        )
        patch = StatePatch(
            doc_id="doc-1",
            base_as_of=t1,
            target_as_of=t2,
            deltas=[delta],
        )
        result = patch.apply_to_state({"Revenue": "$40M"})
        assert result["Revenue"] == "$40M"
        assert "fact-new" in result  # Added fact marked as present

    def test_apply_removals(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(
            base_as_of=t1,
            target_as_of=t2,
            removed_fact_ids={"revenue"},
        )
        patch = StatePatch(
            doc_id="doc-1",
            base_as_of=t1,
            target_as_of=t2,
            deltas=[delta],
        )
        result = patch.apply_to_state({"Revenue": "$40M", "Headcount": 500})
        assert "Headcount" in result
        # removed_fact_ids removes the key itself if it matches
        assert "revenue" not in result

    def test_apply_modifications(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        delta = SnapshotDelta(
            base_as_of=t1,
            target_as_of=t2,
            modified_subjects={"Revenue": {"before": "$40M", "after": "$60M"}},
        )
        patch = StatePatch(
            doc_id="doc-1",
            base_as_of=t1,
            target_as_of=t2,
            deltas=[delta],
        )
        result = patch.apply_to_state({"Revenue": "$40M", "Headcount": 500})
        assert result["Revenue"] == "$60M"
        assert result["Headcount"] == 500

    def test_merged_delta_none_when_no_deltas(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        patch = StatePatch(doc_id="doc-1", base_as_of=t1, target_as_of=t2)
        assert patch.merged_delta is None

    def test_total_change_count(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        d1 = SnapshotDelta(base_as_of=t1, target_as_of=t2, added_fact_ids={"a"})
        d2 = SnapshotDelta(base_as_of=t1, target_as_of=t2, removed_fact_ids={"b"})
        patch = StatePatch(doc_id="doc-1", base_as_of=t1, target_as_of=t2, deltas=[d1, d2])
        assert patch.delta_count == 2
        assert patch.total_change_count == 2


# ═══════════════════════════════════════════════════════════════════════
# HistoricalStateEngine
# ═══════════════════════════════════════════════════════════════════════


class TestGetStateAt:
    """get_state_at returns the correct state at a point in time."""

    @pytest.mark.asyncio
    async def test_get_state_at_q1(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Q1 2025: Revenue=$40M, Headcount=500, Travel Policy active."""
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        state = await engine.get_state_at("doc-123", as_of, tenant_context="tenant-a")
        assert state["Revenue"]["value"] == "$40M"
        assert state["Headcount"]["value"] == "500"
        assert "Travel Policy" in state

    @pytest.mark.asyncio
    async def test_get_state_at_q2(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Q2 2025: Revenue=$50M (updated from $40M)."""
        as_of = datetime(2025, 5, 15, tzinfo=timezone.utc)
        state = await engine.get_state_at("doc-123", as_of, tenant_context="tenant-a")
        assert state["Revenue"]["value"] == "$50M"

    @pytest.mark.asyncio
    async def test_get_state_at_empty_doc(self, engine: HistoricalStateEngine) -> None:
        """Non-existent document returns empty state."""
        as_of = datetime(2025, 6, 1, tzinfo=timezone.utc)
        state = await engine.get_state_at("nonexistent", as_of, tenant_context="tenant-a")
        assert state == {}

    @pytest.mark.asyncio
    async def test_get_state_at_missing_tenant(self, engine: HistoricalStateEngine) -> None:
        """Missing tenant_context raises error."""
        as_of = datetime(2025, 6, 1, tzinfo=timezone.utc)
        with pytest.raises(Exception) as exc:
            await engine.get_state_at("doc-123", as_of)
        assert "tenant_context" in str(exc.value).lower()


class TestComputeDelta:
    """compute_delta returns correct diffs between time points."""

    @pytest.mark.asyncio
    async def test_delta_q1_to_q2(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Q1→Q2: Revenue changes from $40M to $50M."""
        q1 = datetime(2025, 2, 15, tzinfo=timezone.utc)
        q2 = datetime(2025, 5, 15, tzinfo=timezone.utc)
        delta = await engine.compute_delta("doc-123", q1, q2, tenant_context="tenant-a")
        assert not delta.is_empty
        # Revenue fact_id changed (different valid windows = different fact_ids)
        assert len(delta.added_fact_ids) >= 1  # revenue_q2 added
        assert len(delta.removed_fact_ids) >= 1  # revenue_q1 removed
        assert delta.base_as_of == q1
        assert delta.target_as_of == q2

    @pytest.mark.asyncio
    async def test_delta_no_changes(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Same time point produces an empty delta."""
        t = datetime(2025, 2, 15, tzinfo=timezone.utc)
        delta = await engine.compute_delta("doc-123", t, t, tenant_context="tenant-a")
        assert delta.is_empty

    @pytest.mark.asyncio
    async def test_delta_missing_tenant(self, engine: HistoricalStateEngine) -> None:
        """Missing tenant_context raises error."""
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        with pytest.raises(Exception) as exc:
            await engine.compute_delta("doc-123", t1, t2)
        assert "tenant_context" in str(exc.value).lower()


class TestComputeRange:
    """compute_range returns evenly-spaced deltas."""

    @pytest.mark.asyncio
    async def test_compute_range_deltas(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)
        deltas = await engine.compute_range(
            "doc-123",
            t1,
            t3,
            num_intervals=3,
            tenant_context="tenant-a",
        )
        assert len(deltas) == 2  # (num_intervals - 1) deltas

    @pytest.mark.asyncio
    async def test_compute_range_invalid_intervals(
        self,
        engine: HistoricalStateEngine,
    ) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError) as exc:
            await engine.compute_range("doc-123", t1, t2, num_intervals=1)
        assert "num_intervals" in str(exc.value).lower()


class TestBuildPatch:
    """build_patch creates a StatePatch from two time points."""

    @pytest.mark.asyncio
    async def test_build_patch(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        patch = await engine.build_patch("doc-123", t1, t2, tenant_context="tenant-a")
        assert patch.delta_count == 1
        assert patch.base_as_of == t1
        assert patch.target_as_of == t2
        assert patch.doc_id == "doc-123"
        assert not patch.deltas[0].is_empty


class TestListSubjects:
    """list_subjects returns distinct subjects."""

    @pytest.mark.asyncio
    async def test_list_subjects_all(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        subjects = await engine.list_subjects("doc-123", tenant_context="tenant-a")
        assert "Revenue" in subjects
        assert "Headcount" in subjects
        assert "Travel Policy" in subjects

    @pytest.mark.asyncio
    async def test_list_subjects_as_of(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """In Q1, only subjects active at that time."""
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        subjects = await engine.list_subjects(
            "doc-123",
            as_of=as_of,
            tenant_context="tenant-a",
        )
        assert "Revenue" in subjects  # revenue_q1 active in Q1
        assert "Headcount" in subjects  # Always active
        assert "Travel Policy" in subjects  # Active in Q1


class TestCompareStates:
    """compare_states returns structured comparison."""

    @pytest.mark.asyncio
    async def test_compare_states(
        self,
        engine: HistoricalStateEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = await engine.compare_states("doc-123", t1, t2, tenant_context="tenant-a")
        assert "state_a" in result
        assert "state_b" in result
        assert "delta" in result
        assert "summary" in result
        assert result["summary"]["changed_count"] >= 1
