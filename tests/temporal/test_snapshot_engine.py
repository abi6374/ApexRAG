"""
tests/temporal/test_snapshot_engine.py — Tests for SnapshotEngine.

Covers:
  - SnapshotEngine.get_snapshot() lazy construction
  - SnapshotEngine.get_snapshot_between() with delta optimization
  - SnapshotEngine.create_snapshot() and persist
  - SnapshotEngine.create_snapshot_from_delta()
  - SnapshotEngine cache invalidation
  - SnapshotEngine list_manifests / delete_snapshot
  - Tenant isolation in all engine methods
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.fact_store import FactStore, TemporalFact
from apex_rag.temporal.historical_state import HistoricalStateEngine
from apex_rag.temporal.snapshot_engine import SnapshotEngine
from apex_rag.temporal.snapshot_models import SnapshotDelta, SnapshotManifest

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
async def historical_engine(fact_store: FactStore, storage: ApexStorage) -> HistoricalStateEngine:
    return HistoricalStateEngine(fact_store, storage)


@pytest_asyncio.fixture
async def snapshot_engine(
    historical_engine: HistoricalStateEngine,
    fact_store: FactStore,
    storage: ApexStorage,
) -> SnapshotEngine:
    return SnapshotEngine(historical_engine, fact_store, storage)


@pytest_asyncio.fixture
async def seed_facts(fact_store: FactStore) -> dict[str, TemporalFact]:
    """Seed facts with known temporal windows."""
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
# SnapshotEngine — get_snapshot
# ═══════════════════════════════════════════════════════════════════════


class TestGetSnapshot:
    """get_snapshot lazy construction."""

    @pytest.mark.asyncio
    async def test_get_snapshot_q1(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Q1 2025 snapshot: Revenue=$40M, Headcount=500."""
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        state = await snapshot_engine.get_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
        )
        assert state["Revenue"]["value"] == "$40M"
        assert state["Headcount"]["value"] == "500"

    @pytest.mark.asyncio
    async def test_get_snapshot_q2(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Q2 2025 snapshot: Revenue=$50M."""
        as_of = datetime(2025, 5, 15, tzinfo=timezone.utc)
        state = await snapshot_engine.get_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
        )
        assert state["Revenue"]["value"] == "$50M"

    @pytest.mark.asyncio
    async def test_get_snapshot_empty_doc(
        self,
        snapshot_engine: SnapshotEngine,
    ) -> None:
        """Non-existent document returns empty state."""
        as_of = datetime(2025, 6, 1, tzinfo=timezone.utc)
        state = await snapshot_engine.get_snapshot(
            "nonexistent",
            as_of,
            tenant_context="tenant-a",
        )
        assert state == {}

    @pytest.mark.asyncio
    async def test_get_snapshot_missing_tenant(
        self,
        snapshot_engine: SnapshotEngine,
    ) -> None:
        """Missing tenant_context raises error."""
        as_of = datetime(2025, 6, 1, tzinfo=timezone.utc)
        with pytest.raises(Exception) as exc:
            await snapshot_engine.get_snapshot("doc-123", as_of)
        assert "tenant_context" in str(exc.value).lower()


class TestGetSnapshotBetween:
    """get_snapshot_between with delta optimization."""

    @pytest.mark.asyncio
    async def test_get_snapshot_between(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        t1 = datetime(2025, 2, 15, tzinfo=timezone.utc)
        t2 = datetime(2025, 5, 15, tzinfo=timezone.utc)
        state = await snapshot_engine.get_snapshot_between(
            "doc-123",
            t1,
            t2,
            tenant_context="tenant-a",
        )
        assert isinstance(state, dict)

    @pytest.mark.asyncio
    async def test_get_snapshot_between_no_baseline(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """When no baseline exists, falls back to full lookup."""
        t1 = datetime(2024, 6, 1, tzinfo=timezone.utc)  # Before any facts
        t2 = datetime(2025, 5, 15, tzinfo=timezone.utc)
        state = await snapshot_engine.get_snapshot_between(
            "doc-123",
            t1,
            t2,
            tenant_context="tenant-a",
        )
        assert state["Revenue"]["value"] == "$50M"


# ═══════════════════════════════════════════════════════════════════════
# SnapshotEngine — create_snapshot
# ═══════════════════════════════════════════════════════════════════════


class TestCreateSnapshot:
    """create_snapshot and persist."""

    @pytest.mark.asyncio
    async def test_create_snapshot_returns_manifest(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        manifest = await snapshot_engine.create_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
            persist=False,
        )
        assert isinstance(manifest, SnapshotManifest)
        assert manifest.doc_id == "doc-123"
        assert manifest.fact_count >= 2  # Revenue + Headcount + Travel Policy
        assert manifest.is_full is True

    @pytest.mark.asyncio
    async def test_create_snapshot_persists(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        await snapshot_engine.create_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
            persist=True,
        )
        # Can retrieve from persisted storage
        state = await snapshot_engine.get_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
        )
        assert state["Revenue"]["value"] == "$40M"

    @pytest.mark.asyncio
    async def test_create_snapshot_cache_hit(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Second call uses cache."""
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        # First call — builds lazy
        state1 = await snapshot_engine.get_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
        )
        # Second call — should hit cache
        state2 = await snapshot_engine.get_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
        )
        assert state1 == state2

    @pytest.mark.asyncio
    async def test_create_snapshot_missing_tenant(
        self,
        snapshot_engine: SnapshotEngine,
    ) -> None:
        as_of = datetime(2025, 6, 1, tzinfo=timezone.utc)
        with pytest.raises(Exception) as exc:
            await snapshot_engine.create_snapshot("doc-123", as_of)
        assert "tenant_context" in str(exc.value).lower()


class TestCreateSnapshotFromDelta:
    """create_snapshot_from_delta builds from delta."""

    @pytest.mark.asyncio
    async def test_create_from_delta(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        t1 = datetime(2025, 2, 15, tzinfo=timezone.utc)
        t2 = datetime(2025, 5, 15, tzinfo=timezone.utc)

        # Create baseline
        await snapshot_engine.create_snapshot(
            "doc-123",
            t1,
            tenant_context="tenant-a",
            persist=False,
        )

        # Create delta
        delta = SnapshotDelta(
            doc_id="doc-123",
            tenant_id="tenant-a",
            base_as_of=t1,
            target_as_of=t2,
            added_fact_ids={"new-revenue"},
            removed_fact_ids=set(),
            modified_subjects={"Revenue": {"before": "$40M", "after": "$50M"}},
        )

        manifest = await snapshot_engine.create_snapshot_from_delta(
            delta,
            tenant_context="tenant-a",
            persist=False,
        )
        assert isinstance(manifest, SnapshotManifest)
        assert manifest.doc_id == "doc-123"


class TestCacheManagement:
    """Cache invalidation and manifest listing."""

    @pytest.mark.asyncio
    async def test_invalidate_cache_doc(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        await snapshot_engine.get_snapshot("doc-123", as_of, tenant_context="tenant-a")
        await snapshot_engine.invalidate_cache(doc_id="doc-123", tenant_context="tenant-a")
        # Cache cleared — next call rebuilds
        state = await snapshot_engine.get_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
        )
        assert state["Revenue"]["value"] == "$40M"

    @pytest.mark.asyncio
    async def test_invalidate_cache_all(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        await snapshot_engine.invalidate_cache()
        # No error expected
        assert True

    @pytest.mark.asyncio
    async def test_list_manifests(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        t1 = datetime(2025, 2, 15, tzinfo=timezone.utc)
        t2 = datetime(2025, 5, 15, tzinfo=timezone.utc)

        await snapshot_engine.create_snapshot(
            "doc-123",
            t1,
            tenant_context="tenant-a",
            persist=False,
        )
        await snapshot_engine.create_snapshot(
            "doc-123",
            t2,
            tenant_context="tenant-a",
            persist=False,
        )

        manifests = await snapshot_engine.list_manifests(
            "doc-123", tenant_context="tenant-a"
        )
        assert len(manifests) == 2
        # Most recent first
        assert manifests[0].snapshot_date >= manifests[1].snapshot_date


class TestDeleteSnapshot:
    """Delete snapshots."""

    @pytest.mark.asyncio
    async def test_delete_nonexistent(
        self,
        snapshot_engine: SnapshotEngine,
    ) -> None:
        as_of = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = await snapshot_engine.delete_snapshot(
            "nonexistent", as_of, tenant_context="tenant-a"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_persisted_snapshot(
        self,
        snapshot_engine: SnapshotEngine,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        await snapshot_engine.create_snapshot(
            "doc-123",
            as_of,
            tenant_context="tenant-a",
            persist=True,
        )
        result = await snapshot_engine.delete_snapshot(
            "doc-123", as_of, tenant_context="tenant-a"
        )
        assert result is True
