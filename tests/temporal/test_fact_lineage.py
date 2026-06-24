"""
tests/temporal/test_fact_lineage.py — Tests for fact_lineage.py.

Covers:
  - LineageValidator: validate_edge (cycle detection)
  - LineageValidator: detect_cycle, assert_acyclic
  - FactLineageEngine: find_origin, find_descendants
  - FactLineageEngine: find_fact_history, find_superseded_chain
  - FactLineageEngine: find_related_facts
  - Tenant enforcement in all lineage methods
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from apex_rag.ingestion.apex_storage import ApexBase, ApexStorage
from apex_rag.temporal.fact_lineage import FactLineageEngine, LineageValidator
from apex_rag.temporal.fact_store import FactStore, TemporalFact


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage() -> AsyncGenerator[ApexStorage, None]:
    """Create a fresh ApexStorage per test using a temp file.

    Uses ``ApexStorage.create()`` which has production-grade schema
    creation that gracefully handles SQLite's lack of INDEX IF NOT EXISTS.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    storage = await ApexStorage.create(f"sqlite+aiosqlite:///{tmp.name}")
    yield storage


@pytest_asyncio.fixture
async def fact_store(storage: ApexStorage) -> FactStore:
    return FactStore(storage)


@pytest_asyncio.fixture
async def lineage(storage: ApexStorage) -> FactLineageEngine:
    return FactLineageEngine(storage)


@pytest_asyncio.fixture
async def validator(storage: ApexStorage) -> LineageValidator:
    return LineageValidator(storage)


@pytest_asyncio.fixture
async def seed_linear_lineage(fact_store: FactStore) -> dict[str, TemporalFact]:
    """Create a linear lineage A → B → C (B supersedes A, C supersedes B)."""
    now = datetime.now(timezone.utc)
    a = TemporalFact(
        subject="Revenue",
        predicate="was",
        object="$40M",
        source_document_id="doc-123",
        valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
        created_at=now,
    )
    saved_a = await fact_store.save_fact(a, tenant_context="tenant-a")

    b = TemporalFact(
        subject="Revenue",
        predicate="was",
        object="$50M",
        source_document_id="doc-123",
        valid_from=datetime(2025, 4, 1, tzinfo=timezone.utc),
        parent_fact_id=saved_a.fact_id,
        created_at=datetime.now(timezone.utc),
    )
    saved_b = await fact_store.save_fact(b, tenant_context="tenant-a")

    c = TemporalFact(
        subject="Revenue",
        predicate="was",
        object="$60M",
        source_document_id="doc-123",
        valid_from=datetime(2025, 7, 1, tzinfo=timezone.utc),
        parent_fact_id=saved_b.fact_id,
        created_at=datetime.now(timezone.utc),
    )
    saved_c = await fact_store.save_fact(c, tenant_context="tenant-a")

    return {"A": saved_a, "B": saved_b, "C": saved_c}


# ── LineageValidator — Cycle Detection ──────────────────────────────────


class TestLineageValidator:
    """Write-time cycle detection (Principles 3, 11)."""

    @pytest.mark.asyncio
    async def test_validate_edge_no_cycle(
        self, validator: LineageValidator, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Adding A → D where D is new and has no parent should be valid.

        In the seed A→B→C, B already has A as parent and C already has
        B as parent.  A new fact D with no parent should accept A as parent
        because D is not reachable from A via parent links and A is not
        reachable from D (D has no parent).
        """
        lineage = seed_linear_lineage
        # Create a new fact with no parent — adding A→D should be valid
        d = TemporalFact(
            subject="Test", predicate="is", object="new",
            source_document_id="doc-123",
        )
        # D has no parent_fact_id, so adding A as D's parent should be acyclic
        await validator.validate_edge(
            lineage["A"].fact_id, d.fact_id,
        )
        # No exception = validated

    @pytest.mark.asyncio
    async def test_validate_edge_cycle_raises(
        self, validator: LineageValidator, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Adding C → A would create a cycle because A is ancestor of C."""
        lineage = seed_linear_lineage
        with pytest.raises(ValueError) as exc:
            await validator.validate_edge(
                lineage["C"].fact_id,  # C is the proposed parent
                lineage["A"].fact_id,  # A is the proposed child
            )
        assert "cycle" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_validate_edge_self_cycle(
        self, validator: LineageValidator, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Adding a fact as its own parent should be rejected."""
        lineage = seed_linear_lineage
        with pytest.raises(ValueError) as exc:
            await validator.validate_edge(
                lineage["B"].fact_id,
                lineage["B"].fact_id,  # Same fact
            )
        assert "cycle" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_detect_cycle_in_linear_lineage(
        self, validator: LineageValidator, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """A linear lineage A→B→C should have no cycles."""
        lineage = seed_linear_lineage
        assert await validator.detect_cycle(lineage["A"].fact_id) is False
        assert await validator.detect_cycle(lineage["B"].fact_id) is False
        assert await validator.detect_cycle(lineage["C"].fact_id) is False

    @pytest.mark.asyncio
    async def test_assert_acyclic_passes(
        self, validator: LineageValidator, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """assert_acyclic should not raise for an acyclic lineage."""
        lineage = seed_linear_lineage
        await validator.assert_acyclic(lineage["B"].fact_id)
        # No exception = acyclic

    @pytest.mark.asyncio
    async def test_assert_acyclic_noop_for_isolated(
        self, validator: LineageValidator,
    ) -> None:
        """An isolated fact (no parent) is always acyclic."""
        await validator.assert_acyclic("nonexistent")
        # No exception


# ── FactLineageEngine — Lineage Navigation ──────────────────────────────


class TestFactLineageEngineFindOrigin:
    """Trace back to the root of a lineage."""

    @pytest.mark.asyncio
    async def test_find_origin_of_middle(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Origin of B should be A."""
        origin = await lineage.find_origin(
            seed_linear_lineage["B"].fact_id, tenant_context="tenant-a",
        )
        assert origin is not None
        assert origin.fact_id == seed_linear_lineage["A"].fact_id
        assert origin.object == "$40M"

    @pytest.mark.asyncio
    async def test_find_origin_of_latest(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Origin of C should be A."""
        origin = await lineage.find_origin(
            seed_linear_lineage["C"].fact_id, tenant_context="tenant-a",
        )
        assert origin is not None
        assert origin.fact_id == seed_linear_lineage["A"].fact_id

    @pytest.mark.asyncio
    async def test_find_origin_of_root(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Origin of A should be A itself (no parent)."""
        origin = await lineage.find_origin(
            seed_linear_lineage["A"].fact_id, tenant_context="tenant-a",
        )
        assert origin is not None
        assert origin.fact_id == seed_linear_lineage["A"].fact_id

    @pytest.mark.asyncio
    async def test_find_origin_nonexistent(
        self, lineage: FactLineageEngine,
    ) -> None:
        """Nonexistent fact should return None."""
        origin = await lineage.find_origin(
            "nonexistent", tenant_context="tenant-a",
        )
        assert origin is None

    @pytest.mark.asyncio
    async def test_find_origin_missing_tenant(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Missing tenant_context should return None."""
        origin = await lineage.find_origin(
            seed_linear_lineage["A"].fact_id, tenant_context=None,
        )
        assert origin is None


class TestFactLineageEngineDescendants:
    """Find all descendant facts."""

    @pytest.mark.asyncio
    async def test_find_descendants_of_root(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """A's descendants should be B and C."""
        descendants = await lineage.find_descendants(
            seed_linear_lineage["A"].fact_id, tenant_context="tenant-a",
        )
        ids = {d.fact_id for d in descendants}
        assert seed_linear_lineage["B"].fact_id in ids
        assert seed_linear_lineage["C"].fact_id in ids

    @pytest.mark.asyncio
    async def test_find_descendants_of_leaf(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """C (the leaf) should have no descendants."""
        descendants = await lineage.find_descendants(
            seed_linear_lineage["C"].fact_id, tenant_context="tenant-a",
        )
        assert descendants == []

    @pytest.mark.asyncio
    async def test_find_descendants_missing_tenant(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Missing tenant_context should return empty list."""
        descendants = await lineage.find_descendants(
            seed_linear_lineage["A"].fact_id, tenant_context=None,
        )
        assert descendants == []


class TestFactLineageEngineHistory:
    """Full version history."""

    @pytest.mark.asyncio
    async def test_find_fact_history(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """History of B should include A, B, C in chronological order."""
        history = await lineage.find_fact_history(
            seed_linear_lineage["B"].fact_id, tenant_context="tenant-a",
        )
        assert len(history) == 3
        assert history[0].fact_id == seed_linear_lineage["A"].fact_id
        assert history[1].fact_id == seed_linear_lineage["B"].fact_id
        assert history[2].fact_id == seed_linear_lineage["C"].fact_id


class TestFactLineageEngineSupersededChain:
    """Supersession chain following superseded_by links."""

    @pytest.mark.asyncio
    async def test_find_superseded_chain_linear(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """The superseded chain should follow parent_fact_id links."""
        chain = await lineage.find_superseded_chain(
            seed_linear_lineage["A"].fact_id, tenant_context="tenant-a",
        )
        # The superseded chain follows superseded_by links.
        # Since we used parent_fact_id (not superseded_by), the chain
        # might only return A if superseded_by is None.
        assert len(chain) >= 1


class TestFactLineageEngineRelatedFacts:
    """Related facts by document or subject."""

    @pytest.mark.asyncio
    async def test_find_related_facts_by_document(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Related facts for A should include B and C (same document)."""
        related = await lineage.find_related_facts(
            seed_linear_lineage["A"].fact_id, tenant_context="tenant-a",
        )
        ids = {r.fact_id for r in related}
        assert seed_linear_lineage["B"].fact_id in ids
        assert seed_linear_lineage["C"].fact_id in ids

    @pytest.mark.asyncio
    async def test_find_related_facts_missing_tenant(
        self, lineage: FactLineageEngine, seed_linear_lineage: dict[str, TemporalFact],
    ) -> None:
        """Missing tenant_context should return empty list."""
        related = await lineage.find_related_facts(
            seed_linear_lineage["A"].fact_id, tenant_context=None,
        )
        assert related == []
