"""
tests/temporal/test_fact_store.py — Tests for fact_store.py.

Covers:
  - TemporalFact dataclass immutability and defaults
  - FactStore: save_fact, save_facts, get_fact, get_facts, get_facts_by_document
  - get_facts_at_time (temporal window queries)
  - get_active_facts
  - Tenant enforcement (MissingTenantContextError)
  - Immutable delete_fact tombstone pattern
  - Tenant isolation (cross-tenant access denial)
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.fact_store import FactStore, TemporalFact

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage() -> AsyncGenerator[ApexStorage, None]:
    """Create a fresh ApexStorage per test using a temp file.

    Uses ``ApexStorage.create()`` which has production-grade schema
    creation that gracefully handles SQLite's lack of INDEX IF NOT EXISTS.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        storage = await ApexStorage.create(f"sqlite+aiosqlite:///{tmp.name}")
    yield storage


@pytest_asyncio.fixture
async def fact_store(storage: ApexStorage) -> FactStore:
    return FactStore(storage)


@pytest_asyncio.fixture
async def seed_facts(fact_store: FactStore) -> dict[str, TemporalFact]:
    """Seed a set of test facts with known temporal windows."""
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
        "revenue_q3": TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$60M",
            confidence=0.9,
            source_document_id="doc-123",
            valid_from=datetime(2025, 7, 1, tzinfo=timezone.utc),
            valid_to=None,  # Currently active
        ),
        "policy_x": TemporalFact(
            subject="Travel Policy",
            predicate="shall",
            object="Reimbursement requires receipt",
            confidence=0.8,
            source_document_id="doc-123",
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2025, 12, 31, tzinfo=timezone.utc),
        ),
        "org_acme": TemporalFact(
            subject="Acme Corp",
            predicate="is",
            object="organization",
            confidence=0.7,
            source_document_id="doc-456",
            valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
            valid_to=None,
        ),
    }
    saved = await fact_store.save_facts(
        list(facts.values()),
        tenant_context="tenant-a",
    )
    assert len(saved) == 5
    # Return by key name for easy access in tests
    result: dict[str, TemporalFact] = {}
    for key, f in facts.items():
        # Find matching saved fact by subject + valid_from
        for s in saved:
            if s.subject == f.subject and s.valid_from == f.valid_from:
                result[key] = s
                break
    return result


# ── TemporalFact Model ──────────────────────────────────────────────────


class TestTemporalFactModel:
    """TemporalFact is a frozen dataclass — test its immutability."""

    def test_default_fact_id_is_uuid4(self) -> None:
        fact = TemporalFact(subject="Test", predicate="is", object="test")
        assert fact.fact_id is not None
        uuid.UUID(fact.fact_id, version=4)

    def test_default_tenant_is_default(self) -> None:
        fact = TemporalFact(subject="Test", predicate="is", object="test")
        assert fact.tenant_id == "default"

    def test_default_valid_from_is_now(self) -> None:
        fact = TemporalFact(subject="Test", predicate="is", object="test")
        assert fact.valid_from is not None
        # Should be very recent
        diff = datetime.now(timezone.utc) - fact.valid_from
        assert diff.total_seconds() < 5  # Within 5 seconds

    def test_default_valid_to_is_none(self) -> None:
        fact = TemporalFact(subject="Test", predicate="is", object="test")
        assert fact.valid_to is None

    def test_frozen_immutability(self) -> None:
        fact = TemporalFact(subject="Test", predicate="is", object="test")
        with pytest.raises(AttributeError):
            fact.subject = "Changed"  # type: ignore[misc]

    def test_default_confidence_is_one(self) -> None:
        fact = TemporalFact(subject="Test", predicate="is", object="test")
        assert fact.confidence == 1.0

    def test_custom_fields(self) -> None:
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        fact = TemporalFact(
            fact_id="custom-id",
            tenant_id="tenant-b",
            subject="Revenue",
            predicate="was",
            object="$100k",
            confidence=0.85,
            source_document_id="doc-x",
            source_node_id="node-y",
            valid_from=now,
            extraction_method="llm",
            metadata={"source": "report.pdf"},
        )
        assert fact.fact_id == "custom-id"
        assert fact.tenant_id == "tenant-b"
        assert fact.confidence == 0.85
        assert fact.metadata == {"source": "report.pdf"}


# ── FactStore CRUD ──────────────────────────────────────────────────────


class TestFactStoreSave:
    """FactStore.save_fact() and save_facts()."""

    @pytest.mark.asyncio
    async def test_save_fact(self, fact_store: FactStore) -> None:
        fact = TemporalFact(
            subject="Test",
            predicate="is",
            object="saved",
            source_document_id="doc-1",
        )
        saved = await fact_store.save_fact(fact, tenant_context="tenant-a")
        assert saved is fact  # Same object returned
        assert saved.fact_id == fact.fact_id

    @pytest.mark.asyncio
    async def test_save_fact_missing_tenant(self, fact_store: FactStore) -> None:
        fact = TemporalFact(subject="Test", predicate="is", object="test")
        with pytest.raises(Exception) as exc:
            await fact_store.save_fact(fact)
        assert "tenant_context" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_save_facts(self, fact_store: FactStore) -> None:
        facts = [
            TemporalFact(subject="A", predicate="is", object="1", source_document_id="doc-1"),
            TemporalFact(subject="B", predicate="is", object="2", source_document_id="doc-1"),
        ]
        saved = await fact_store.save_facts(facts, tenant_context="tenant-a")
        assert len(saved) == 2

    @pytest.mark.asyncio
    async def test_save_facts_missing_tenant(self, fact_store: FactStore) -> None:
        with pytest.raises(Exception) as exc:
            await fact_store.save_facts([], tenant_context=None)
        assert "tenant_context" in str(exc.value).lower()


class TestFactStoreRead:
    """FactStore get methods."""

    @pytest.mark.asyncio
    async def test_get_fact(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        f = seed_facts["revenue_q1"]
        retrieved = await fact_store.get_fact(f.fact_id, tenant_context="tenant-a")
        assert retrieved is not None
        assert retrieved.subject == "Revenue"
        assert retrieved.object == "$40M"

    @pytest.mark.asyncio
    async def test_get_fact_missing_tenant(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        f = seed_facts["revenue_q1"]
        with pytest.raises(Exception) as exc:
            await fact_store.get_fact(f.fact_id)
        assert "tenant_context" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_get_fact_not_found(self, fact_store: FactStore) -> None:
        result = await fact_store.get_fact("nonexistent-id", tenant_context="tenant-a")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_facts_by_document(
        self,
        fact_store: FactStore,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        facts = await fact_store.get_facts_by_document("doc-123", tenant_context="tenant-a")
        assert len(facts) == 4  # doc-123 has 4 facts
        subjects = {f.subject for f in facts}
        assert "Revenue" in subjects
        assert "Travel Policy" in subjects

    @pytest.mark.asyncio
    async def test_get_facts_by_document_empty(self, fact_store: FactStore) -> None:
        facts = await fact_store.get_facts_by_document("nonexistent", tenant_context="tenant-a")
        assert facts == []

    @pytest.mark.asyncio
    async def test_get_facts(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        facts = await fact_store.get_facts(tenant_context="tenant-a", limit=10)
        assert len(facts) == 5  # All facts

    @pytest.mark.asyncio
    async def test_get_facts_pagination(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        facts = await fact_store.get_facts(tenant_context="tenant-a", limit=2, offset=0)
        assert len(facts) == 2


class TestTemporalQueries:
    """Temporal window queries — get_facts_at_time and get_active_facts."""

    @pytest.mark.asyncio
    async def test_get_facts_at_time_q1(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        """Q1 2025 — only revenue_q1 and policy_x are valid."""
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        facts = await fact_store.get_facts_at_time("doc-123", as_of, tenant_context="tenant-a")
        subjects = {f.subject for f in facts}
        assert "Revenue" in subjects  # revenue_q1 valid until 2025-04-01
        assert "Travel Policy" in subjects  # valid all 2025
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_get_facts_at_time_q2(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        """Q2 2025 — revenue_q2 and policy_x are valid."""
        as_of = datetime(2025, 5, 15, tzinfo=timezone.utc)
        facts = await fact_store.get_facts_at_time("doc-123", as_of, tenant_context="tenant-a")
        revenue_facts = [f for f in facts if f.subject == "Revenue"]
        assert len(revenue_facts) == 1
        assert revenue_facts[0].object == "$50M"  # revenue_q2
        assert "Travel Policy" in {f.subject for f in facts}

    @pytest.mark.asyncio
    async def test_get_facts_at_time_q4(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        """Q4 2025 — revenue_q3 (active) and policy_x are valid."""
        as_of = datetime(2025, 10, 1, tzinfo=timezone.utc)
        facts = await fact_store.get_facts_at_time("doc-123", as_of, tenant_context="tenant-a")
        revenue_facts = [f for f in facts if f.subject == "Revenue"]
        assert len(revenue_facts) == 1
        assert revenue_facts[0].object == "$60M"  # revenue_q3 (currently active)
        assert "Travel Policy" in {f.subject for f in facts}

    @pytest.mark.asyncio
    async def test_get_facts_at_time_2026(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        """2026 — only revenue_q3 is valid (policy_x expired in 2025)."""
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        facts = await fact_store.get_facts_at_time("doc-123", as_of, tenant_context="tenant-a")
        assert "Travel Policy" not in {f.subject for f in facts}
        assert len(facts) == 1  # Only revenue_q3

    @pytest.mark.asyncio
    async def test_get_active_facts(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        """Active facts are those with valid_to=None (currently valid)."""
        facts = await fact_store.get_active_facts("doc-123", tenant_context="tenant-a")
        # revenue_q3 (no valid_to) should be included; revenue_q1/q2 have ended
        active_subjects = {f.subject for f in facts}
        assert "Revenue" in active_subjects


class TestTenantIsolation:
    """Cross-tenant access must be denied."""

    @pytest.mark.asyncio
    async def test_cross_tenant_get_facts(
        self,
        fact_store: FactStore,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        """Facts saved in tenant-a should not be visible from tenant-b."""
        facts = await fact_store.get_facts_by_document("doc-123", tenant_context="tenant-b")
        assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_cross_tenant_get_fact(
        self,
        fact_store: FactStore,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        f = seed_facts["revenue_q1"]
        result = await fact_store.get_fact(f.fact_id, tenant_context="tenant-b")
        assert result is None

    @pytest.mark.asyncio
    async def test_cross_tenant_get_facts_at_time(
        self,
        fact_store: FactStore,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        as_of = datetime(2025, 2, 15, tzinfo=timezone.utc)
        facts = await fact_store.get_facts_at_time("doc-123", as_of, tenant_context="tenant-b")
        assert len(facts) == 0


class TestDeleteFact:
    """Immutable delete — creates tombstone, never mutates original."""

    @pytest.mark.asyncio
    async def test_delete_fact_creates_tombstone(
        self,
        fact_store: FactStore,
        seed_facts: dict[str, TemporalFact],
    ) -> None:
        f = seed_facts["policy_x"]
        result = await fact_store.delete_fact(f.fact_id, tenant_context="tenant-a")
        assert result is True

        # Original fact should still exist (no mutation)
        original = await fact_store.get_fact(f.fact_id, tenant_context="tenant-a")
        assert original is not None
        assert original.subject == "Travel Policy"

        # Tombstone should be a separate fact with __DELETED__ prefix
        all_facts = await fact_store.get_facts_by_document("doc-123", tenant_context="tenant-a")
        deleted_prefix_facts = [f for f in all_facts if f.subject.startswith("__DELETED__")]
        assert len(deleted_prefix_facts) == 1
        assert deleted_prefix_facts[0].parent_fact_id == f.fact_id

    @pytest.mark.asyncio
    async def test_delete_fact_nonexistent(self, fact_store: FactStore) -> None:
        result = await fact_store.delete_fact("nonexistent", tenant_context="tenant-a")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_fact_missing_tenant(
        self, fact_store: FactStore, seed_facts: dict[str, TemporalFact]
    ) -> None:
        f = seed_facts["revenue_q1"]
        with pytest.raises(Exception) as exc:
            await fact_store.delete_fact(f.fact_id)
        assert "tenant_context" in str(exc.value).lower()


class TestFactStoreRowConversion:
    """Internal _row_to_fact conversion."""

    @pytest.mark.asyncio
    async def test_metadata_roundtrip(self, fact_store: FactStore) -> None:
        meta = {"source": "report.pdf", "page": 5, "tags": ["financial", "Q3"]}
        fact = TemporalFact(
            subject="Test",
            predicate="has",
            object="metadata",
            source_document_id="doc-1",
            metadata=meta,
        )
        await fact_store.save_fact(fact, tenant_context="tenant-a")
        retrieved = await fact_store.get_fact(fact.fact_id, tenant_context="tenant-a")
        assert retrieved is not None
        assert retrieved.metadata == meta

    @pytest.mark.asyncio
    async def test_extraction_method_roundtrip(self, fact_store: FactStore) -> None:
        fact = TemporalFact(
            subject="Test",
            predicate="is",
            object="extracted",
            source_document_id="doc-1",
            extraction_method="llm",
        )
        await fact_store.save_fact(fact, tenant_context="tenant-a")
        retrieved = await fact_store.get_fact(fact.fact_id, tenant_context="tenant-a")
        assert retrieved is not None
        assert retrieved.extraction_method == "llm"
