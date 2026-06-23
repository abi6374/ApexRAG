"""
tests/temporal/test_chain_reconciler.py — Tests for Sprint 6 chain reconciliation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.fact_store import FactStore, TemporalFact
from apex_rag.temporal.chain_reconciler import (
    AnomalyType,
    ChainAnomaly,
    ChainDiagnosticReport,
    ChainGapDetector,
    ChainReconciliationReport,
    CrossChainStateReconstructor,
    ReconciledChain,
    VersionChainReconciler,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
async def storage() -> ApexStorage:
    """Create a fresh in-memory SQLite database for each test."""
    return await ApexStorage.create("sqlite+aiosqlite:///:memory:")


@pytest.fixture
async def fact_store(storage: ApexStorage) -> FactStore:
    """Create a FactStore backed by the in-memory database."""
    return FactStore(storage)


@pytest.fixture
async def store_and_facts(fact_store: FactStore) -> tuple[FactStore, list[TemporalFact]]:
    """Create a FactStore with seed facts for reconciliation tests."""
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
    t3 = datetime(2025, 3, 1, tzinfo=timezone.utc)

    facts = [
        # Revenue chain — 3 versions (v1 → v2 → v3)
        TemporalFact(
            fact_id="rev-v1",
            subject="Revenue",
            predicate="was",
            object="$100K",
            valid_from=t1,
            valid_to=t2,
            created_at=t1,
        ),
        TemporalFact(
            fact_id="rev-v2",
            subject="Revenue",
            predicate="was",
            object="$120K",
            valid_from=t2,
            valid_to=t3,
            created_at=t2,
            parent_fact_id="rev-v1",
            # superseded_by intentionally omitted to avoid circular FK dep
            # rev-v3 references rev-v2 via parent_fact_id, and making rev-v2
            # also reference rev-v3 via superseded_by creates a mutual FK cycle
            # that SQLite's immediate FK enforcement rejects.
        ),
        TemporalFact(
            fact_id="rev-v3",
            subject="Revenue",
            predicate="was",
            object="$150K",
            valid_from=t3,
            valid_to=None,
            created_at=t3,
            parent_fact_id="rev-v2",
        ),
        # Headcount chain — single fact
        TemporalFact(
            fact_id="hc-v1",
            subject="Headcount",
            predicate="was",
            object="500",
            valid_from=t1,
            valid_to=None,
            created_at=t1,
        ),
        # TaxRate chain — 2 versions
        TemporalFact(
            fact_id="tax-v1",
            subject="TaxRate",
            predicate="was",
            object="21%",
            valid_from=t1,
            valid_to=t2,
            created_at=t1,
        ),
        TemporalFact(
            fact_id="tax-v2",
            subject="TaxRate",
            predicate="was",
            object="18%",
            valid_from=t2,
            valid_to=None,
            created_at=t2,
            parent_fact_id="tax-v1",
        ),
    ]

    # Save facts in dependency order to satisfy immediate FK enforcement in SQLite
    roots = [f for f in facts if not f.parent_fact_id and not f.superseded_by]
    branches = [f for f in facts if f.parent_fact_id or f.superseded_by]
    for f in roots:
        await fact_store.save_fact(f, tenant_context="tenant-a")
    for f in branches:
        await fact_store.save_fact(f, tenant_context="tenant-a")
    return fact_store, facts


# ═══════════════════════════════════════════════════════════════
# Data Model Tests
# ═══════════════════════════════════════════════════════════════


class TestChainAnomaly:
    """Tests for the ChainAnomaly frozen dataclass."""

    def test_create_anomaly(self) -> None:
        anomaly = ChainAnomaly(
            anomaly_type=AnomalyType.MISSING_VERSION,
            description="Version gap detected",
            fact_id="fact-1",
        )
        assert anomaly.anomaly_type == AnomalyType.MISSING_VERSION
        assert anomaly.severity == "warning"
        assert anomaly.related_fact_id is None

    def test_immutable(self) -> None:
        anomaly = ChainAnomaly(
            anomaly_type=AnomalyType.FORK_DETECTED,
            description="Fork detected",
            fact_id="fact-1",
        )
        with pytest.raises(AttributeError):
            anomaly.severity = "error"  # type: ignore[misc]

    def test_with_metadata(self) -> None:
        anomaly = ChainAnomaly(
            anomaly_type=AnomalyType.BROKEN_SUPERSEDES_LINK,
            description="Broken link",
            fact_id="fact-1",
            related_fact_id="fact-nonexistent",
            severity="error",
            metadata={"field": "superseded_by"},
        )
        assert anomaly.related_fact_id == "fact-nonexistent"
        assert anomaly.metadata["field"] == "superseded_by"


class TestChainDiagnosticReport:
    """Tests for the ChainDiagnosticReport frozen dataclass."""

    def test_empty_report(self) -> None:
        report = ChainDiagnosticReport(node_id="test", tenant_id="t1")
        assert report.chain_length == 0
        assert report.is_healthy is True
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_error_count(self) -> None:
        report = ChainDiagnosticReport(
            node_id="test",
            tenant_id="t1",
            anomalies=[
                ChainAnomaly(
                    anomaly_type=AnomalyType.BROKEN_SUPERSEDES_LINK,
                    description="err1", fact_id="f1",
                    severity="error",
                ),
                ChainAnomaly(
                    anomaly_type=AnomalyType.MISSING_VERSION,
                    description="warn1", fact_id="f2",
                    severity="warning",
                ),
                ChainAnomaly(
                    anomaly_type=AnomalyType.ORPHANED_FACT,
                    description="info1", fact_id="f3",
                    severity="info",
                ),
            ],
        )
        assert report.anomaly_count == 3
        assert report.error_count == 1
        assert report.warning_count == 1
        assert report.is_healthy is False

    def test_healthy_with_warnings(self) -> None:
        report = ChainDiagnosticReport(
            node_id="test",
            tenant_id="t1",
            anomalies=[
                ChainAnomaly(
                    anomaly_type=AnomalyType.MISSING_VERSION,
                    description="warn", fact_id="f1",
                    severity="warning",
                ),
            ],
        )
        # is_healthy is True because only warnings, no errors
        assert report.is_healthy is True


class TestReconciledChain:
    """Tests for the ReconciledChain frozen dataclass."""

    def test_create(self) -> None:
        chain = ReconciledChain(
            node_id="Revenue",
            tenant_id="tenant-a",
            chain_length=3,
        )
        assert chain.node_id == "Revenue"
        assert chain.reconciled is False
        assert chain.is_forked is False
        assert chain.authoritative is None


class TestChainReconciliationReport:
    """Tests for the ChainReconciliationReport."""

    def test_create(self) -> None:
        report = ChainReconciliationReport(
            doc_id="doc-123",
            tenant_id="tenant-a",
            total_chains=3,
        )
        assert report.total_chains == 3
        assert report.total_anomalies == 0


# ═══════════════════════════════════════════════════════════════
# ChainGapDetector Tests
# ═══════════════════════════════════════════════════════════════


class TestDetectAll:
    """Tests for ChainGapDetector.detect_all()."""

    @pytest.mark.asyncio
    async def test_healthy_chain(self, store_and_facts: tuple[FactStore, list[TemporalFact]]) -> None:
        fact_store, facts = store_and_facts
        detector = ChainGapDetector(fact_store)
        report = await detector.detect_all("Revenue", doc_id="", tenant_context="tenant-a")
        assert report.chain_length >= 3
        # Should be healthy (well-formed supersession chain)
        assert report.is_healthy

    @pytest.mark.asyncio
    async def test_empty_chain(self, fact_store: FactStore) -> None:
        detector = ChainGapDetector(fact_store)
        report = await detector.detect_all("Nonexistent", tenant_context="tenant-a")
        assert report.chain_length == 0
        assert report.is_healthy

    @pytest.mark.asyncio
    async def test_no_tenant_context(self, fact_store: FactStore) -> None:
        detector = ChainGapDetector(fact_store)
        report = await detector.detect_all("Revenue", tenant_context=None)
        assert report.chain_length == 0
        assert report.is_healthy


class TestCheckGaps:
    """Tests for gap detection logic."""

    @pytest.mark.asyncio
    async def test_no_gaps(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        facts = [
            TemporalFact(
                fact_id="v1", subject="X", predicate="was", object="1",
                metadata={"version_number": 1},
            ),
            TemporalFact(
                fact_id="v2", subject="X", predicate="was", object="2",
                metadata={"version_number": 2},
            ),
        ]
        anomalies = await detector._check_gaps(facts)
        assert len(anomalies) == 0


class TestCheckOverlappingValidity:
    """Tests for overlapping validity window detection."""

    @pytest.mark.asyncio
    async def test_no_overlap(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 3, 1, tzinfo=timezone.utc)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         valid_from=t1, valid_to=t2),
            TemporalFact(fact_id="b", subject="X", predicate="was", object="2",
                         valid_from=t2, valid_to=t3),
        ]
        anomalies = detector._check_overlapping_validity(facts)
        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_overlap_detected(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
        # Overlap: a ends at t2, b starts at t1 — overlapping
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         valid_from=t1, valid_to=t2),
            TemporalFact(fact_id="b", subject="X", predicate="was", object="2",
                         valid_from=t1, valid_to=t2),
        ]
        anomalies = detector._check_overlapping_validity(facts)
        assert len(anomalies) >= 1
        assert anomalies[0].anomaly_type == AnomalyType.OVERLAPPING_VALIDITY

    @pytest.mark.asyncio
    async def test_skip_supersession_overlap(self) -> None:
        """Overlap is expected if one fact supersedes the other."""
        detector = ChainGapDetector.__new__(ChainGapDetector)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         valid_from=t1, valid_to=t2, superseded_by="b"),
            TemporalFact(fact_id="b", subject="X", predicate="was", object="2",
                         valid_from=t1, valid_to=None),
        ]
        anomalies = detector._check_overlapping_validity(facts)
        assert len(anomalies) == 0  # Supersession chain, overlap expected


class TestCheckExpiredActive:
    """Tests for expired-but-active detection."""

    @pytest.mark.asyncio
    async def test_expired_detected(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         valid_from=past, valid_to=past + timedelta(days=1),
                         superseded_by=None),
        ]
        anomalies = detector._check_expired_active(facts)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.EXPIRED_ACTIVE

    @pytest.mark.asyncio
    async def test_expired_with_superseder_ok(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         valid_from=past, valid_to=past + timedelta(days=1),
                         superseded_by="b"),
        ]
        anomalies = detector._check_expired_active(facts)
        assert len(anomalies) == 0  # Has superseder, OK


class TestCheckForks:
    """Tests for fork detection."""

    @pytest.mark.asyncio
    async def test_no_fork(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         parent_fact_id=None),
            TemporalFact(fact_id="b", subject="X", predicate="was", object="2",
                         parent_fact_id="a"),
        ]
        anomalies = detector._check_forks(facts)
        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_fork_detected(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         parent_fact_id=None),
            TemporalFact(fact_id="b", subject="X", predicate="was", object="2",
                         parent_fact_id="a"),
            TemporalFact(fact_id="c", subject="X", predicate="was", object="3",
                         parent_fact_id="a"),
        ]
        anomalies = detector._check_forks(facts)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.FORK_DETECTED


class TestCheckOrphans:
    """Tests for orphaned fact detection."""

    @pytest.mark.asyncio
    async def test_singleton_not_orphaned(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1"),
        ]
        anomalies = detector._check_orphans(facts)
        assert len(anomalies) == 0  # Single fact is fine

    @pytest.mark.asyncio
    async def test_orphan_detected(self) -> None:
        detector = ChainGapDetector.__new__(ChainGapDetector)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         parent_fact_id=None, valid_from=t1),
            TemporalFact(fact_id="b", subject="X", predicate="was", object="2",
                         parent_fact_id="a", valid_from=t1),
        ]
        # 'a' has a child 'b', so not orphaned. 'b' has parent 'a', so not orphaned.
        anomalies = detector._check_orphans(facts)
        assert len(anomalies) == 0


# ═══════════════════════════════════════════════════════════════
# VersionChainReconciler Tests
# ═══════════════════════════════════════════════════════════════


class TestResolveAuthoritative:
    """Tests for VersionChainReconciler.resolve_authoritative()."""

    @pytest.mark.asyncio
    async def test_follows_supersession_chain(
        self, store_and_facts: tuple[FactStore, list[TemporalFact]],
    ) -> None:
        fact_store, facts = store_and_facts
        reconciler = VersionChainReconciler(fact_store)
        auth = await reconciler.resolve_authoritative(
            "Revenue", tenant_context="tenant-a",
        )
        assert auth is not None
        assert auth.fact_id == "rev-v3"
        assert auth.object == "$150K"

    @pytest.mark.asyncio
    async def test_single_fact(
        self, store_and_facts: tuple[FactStore, list[TemporalFact]],
    ) -> None:
        fact_store, facts = store_and_facts
        reconciler = VersionChainReconciler(fact_store)
        auth = await reconciler.resolve_authoritative(
            "Headcount", tenant_context="tenant-a",
        )
        assert auth is not None
        assert auth.fact_id == "hc-v1"
        assert auth.object == "500"

    @pytest.mark.asyncio
    async def test_no_tenant(self, fact_store: FactStore) -> None:
        reconciler = VersionChainReconciler(fact_store)
        auth = await reconciler.resolve_authoritative(
            "Revenue", tenant_context=None,
        )
        assert auth is None


class TestReconcileChain:
    """Tests for VersionChainReconciler.reconcile_chain()."""

    @pytest.mark.asyncio
    async def test_reconcile_healthy(
        self, store_and_facts: tuple[FactStore, list[TemporalFact]],
    ) -> None:
        fact_store, facts = store_and_facts
        reconciler = VersionChainReconciler(fact_store)
        chain = await reconciler.reconcile_chain(
            "Revenue", tenant_context="tenant-a",
        )
        assert chain.reconciled
        assert chain.chain_length >= 3
        assert chain.authoritative is not None
        assert chain.authoritative.object == "$150K"
        assert chain.report is not None

    @pytest.mark.asyncio
    async def test_reconcile_missing_tenant(
        self, fact_store: FactStore,
    ) -> None:
        reconciler = VersionChainReconciler(fact_store)
        chain = await reconciler.reconcile_chain(
            "Revenue", tenant_context=None,
        )
        assert chain.chain_length == 0
        assert chain.reconciled is False


class TestResolveDocumentChains:
    """Tests for VersionChainReconciler.resolve_document_chains()."""

    @pytest.mark.asyncio
    async def test_resolve_all_chains(
        self,
    ) -> None:
        fact_store = FactStore(await ApexStorage.create("sqlite+aiosqlite:///:memory:"))
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 3, 1, tzinfo=timezone.utc)

        doc_id = "doc-recon"
        doc_facts = [
            TemporalFact(fact_id="dr-rev1", subject="Revenue", predicate="was",
                         object="$100K", source_document_id=doc_id,
                         valid_from=t1, valid_to=t2, created_at=t1),
            TemporalFact(fact_id="dr-rev2", subject="Revenue", predicate="was",
                         object="$120K", source_document_id=doc_id,
                         valid_from=t2, valid_to=t3, created_at=t2,
                         parent_fact_id="dr-rev1"),
            TemporalFact(fact_id="dr-rev3", subject="Revenue", predicate="was",
                         object="$150K", source_document_id=doc_id,
                         valid_from=t3, valid_to=None, created_at=t3,
                         parent_fact_id="dr-rev2"),
            TemporalFact(fact_id="dr-hc1", subject="Headcount", predicate="was",
                         object="500", source_document_id=doc_id,
                         valid_from=t1, valid_to=None, created_at=t1),
            TemporalFact(fact_id="dr-tax1", subject="TaxRate", predicate="was",
                         object="21%", source_document_id=doc_id,
                         valid_from=t1, valid_to=t2, created_at=t1),
            TemporalFact(fact_id="dr-tax2", subject="TaxRate", predicate="was",
                         object="18%", source_document_id=doc_id,
                         valid_from=t2, valid_to=None, created_at=t2,
                         parent_fact_id="dr-tax1"),
        ]

        # Save in dependency order
        roots = [f for f in doc_facts if not f.parent_fact_id]
        branches = [f for f in doc_facts if f.parent_fact_id]
        for f in roots:
            await fact_store.save_fact(f, tenant_context="tenant-a")
        for f in branches:
            await fact_store.save_fact(f, tenant_context="tenant-a")

        reconciler = VersionChainReconciler(fact_store)
        report = await reconciler.resolve_document_chains(
            doc_id, tenant_context="tenant-a",
        )
        assert report.doc_id == doc_id
        assert report.total_chains >= 3  # Revenue, Headcount, TaxRate
        assert "Revenue" in report.chains
        assert "Headcount" in report.chains
        assert "TaxRate" in report.chains

        # Verify Revenue chain is authoritative
        rev_chain = report.chains["Revenue"]
        assert rev_chain.authoritative is not None
        assert rev_chain.authoritative.object == "$150K"

    @pytest.mark.asyncio
    async def test_empty_document(self, fact_store: FactStore) -> None:
        reconciler = VersionChainReconciler(fact_store)
        report = await reconciler.resolve_document_chains(
            "empty-doc", tenant_context="tenant-a",
        )
        assert report.total_chains == 0


class TestDescribeForks:
    """Tests for VersionChainReconciler.describe_forks()."""

    @pytest.mark.asyncio
    async def test_no_forks(
        self, store_and_facts: tuple[FactStore, list[TemporalFact]],
    ) -> None:
        fact_store, facts = store_and_facts
        reconciler = VersionChainReconciler(fact_store)
        forks = await reconciler.describe_forks(
            "Revenue", tenant_context="tenant-a",
        )
        assert len(forks) == 0

    @pytest.mark.asyncio
    async def test_fork_described(self, fact_store: FactStore) -> None:
        """Create a fork and verify it's described."""
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)

        fork_facts = [
            TemporalFact(
                fact_id="root", subject="Budget", predicate="was", object="100",
                source_document_id="doc-fork",
                valid_from=t1, valid_to=None,
            ),
            TemporalFact(
                fact_id="fork-a", subject="Budget", predicate="was", object="120",
                source_document_id="doc-fork",
                valid_from=t2, valid_to=None,
                parent_fact_id="root",
            ),
            TemporalFact(
                fact_id="fork-b", subject="Budget", predicate="was", object="150",
                source_document_id="doc-fork",
                valid_from=t2, valid_to=None,
                parent_fact_id="root",
            ),
        ]
        # Save in dependency order: root first, then branches
        await fact_store.save_fact(fork_facts[0], tenant_context="tenant-a")
        await fact_store.save_fact(fork_facts[1], tenant_context="tenant-a")
        await fact_store.save_fact(fork_facts[2], tenant_context="tenant-a")

        reconciler = VersionChainReconciler(fact_store)
        forks = await reconciler.describe_forks(
            "Budget", doc_id="doc-fork", tenant_context="tenant-a",
        )
        assert len(forks) == 1
        assert forks[0]["parent_fact_id"] == "root"
        assert forks[0]["child_count"] == 2


# ═══════════════════════════════════════════════════════════════
# CrossChainStateReconstructor Tests
# ═══════════════════════════════════════════════════════════════


class TestReconstructAuthoritativeState:
    """Tests for CrossChainStateReconstructor.reconstruct_authoritative_state()."""

    @pytest.mark.asyncio
    async def test_reconstruct_state(
        self, fact_store: FactStore,
    ) -> None:
        """Test full authoritative state reconstruction."""
        doc_id = "doc-auth"
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 3, 1, tzinfo=timezone.utc)

        facts = [
            TemporalFact(fact_id="r1", subject="Revenue", predicate="was",
                         object="$100K", source_document_id=doc_id,
                         valid_from=t1, valid_to=t2, created_at=t1),
            TemporalFact(fact_id="r2", subject="Revenue", predicate="was",
                         object="$150K", source_document_id=doc_id,
                         valid_from=t2, valid_to=None, created_at=t2,
                         parent_fact_id="r1"),
            TemporalFact(fact_id="h1", subject="Headcount", predicate="was",
                         object="500", source_document_id=doc_id,
                         valid_from=t1, valid_to=None, created_at=t1),
        ]
        # Save in dependency order: r1 first (no parent), then r2 (parent=r1)
        await fact_store.save_fact(facts[0], tenant_context="tenant-a")
        await fact_store.save_fact(facts[1], tenant_context="tenant-a")
        await fact_store.save_fact(facts[2], tenant_context="tenant-a")

        reconciler = VersionChainReconciler(fact_store)
        cross = CrossChainStateReconstructor(fact_store, reconciler)

        state = await cross.reconstruct_authoritative_state(
            doc_id, tenant_context="tenant-a",
        )
        assert "Revenue" in state
        assert state["Revenue"]["value"] == "$150K"
        assert state["Revenue"]["confidence"] == 1.0
        assert "Headcount" in state
        assert state["Headcount"]["value"] == "500"

    @pytest.mark.asyncio
    async def test_no_tenant(self, fact_store: FactStore) -> None:
        reconciler = VersionChainReconciler(fact_store)
        cross = CrossChainStateReconstructor(fact_store, reconciler)
        state = await cross.reconstruct_authoritative_state(
            "doc-123", tenant_context=None,
        )
        assert state == {}


class TestReconstructStateAt:
    """Tests for CrossChainStateReconstructor.reconstruct_state_at()."""

    @pytest.mark.asyncio
    async def test_state_at_time(self, fact_store: FactStore) -> None:
        doc_id = "doc-time"
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)

        facts = [
            TemporalFact(fact_id="r1", subject="Revenue", predicate="was",
                         object="$100K", source_document_id=doc_id,
                         valid_from=t1, valid_to=t2),
            TemporalFact(fact_id="r2", subject="Revenue", predicate="was",
                         object="$200K", source_document_id=doc_id,
                         valid_from=t2, valid_to=None),
        ]
        await fact_store.save_facts(facts, tenant_context="tenant-a")

        reconciler = VersionChainReconciler(fact_store)
        cross = CrossChainStateReconstructor(fact_store, reconciler)

        # At t1 + 1 month, r1 should be active
        state = await cross.reconstruct_state_at(
            doc_id, t1 + timedelta(days=30), tenant_context="tenant-a",
        )
        assert state["Revenue"]["value"] == "$100K"

        # At t2 + 1 month, r2 should be active
        state = await cross.reconstruct_state_at(
            doc_id, t2 + timedelta(days=30), tenant_context="tenant-a",
        )
        assert state["Revenue"]["value"] == "$200K"


class TestCompareStates:
    """Tests for CrossChainStateReconstructor.compare_states()."""

    @pytest.mark.asyncio
    async def test_compare(self, fact_store: FactStore) -> None:
        doc_id = "doc-compare"
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)

        facts = [
            TemporalFact(fact_id="r1", subject="Revenue", predicate="was",
                         object="$100K", source_document_id=doc_id,
                         valid_from=t1, valid_to=t2),
            TemporalFact(fact_id="r2", subject="Revenue", predicate="was",
                         object="$200K", source_document_id=doc_id,
                         valid_from=t2, valid_to=None),
        ]
        await fact_store.save_facts(facts, tenant_context="tenant-a")

        reconciler = VersionChainReconciler(fact_store)
        cross = CrossChainStateReconstructor(fact_store, reconciler)

        result = await cross.compare_states(
            doc_id, t1 + timedelta(days=1), t2 + timedelta(days=1),
            tenant_context="tenant-a",
        )
        assert result["summary"]["total_subjects_a"] == 1
        assert result["summary"]["total_subjects_b"] == 1
        assert result["summary"]["modified_count"] == 1
        assert "Revenue" in result["changes"]["modified"]


class TestGetChainSummaries:
    """Tests for CrossChainStateReconstructor.get_chain_summaries()."""

    @pytest.mark.asyncio
    async def test_summaries(
        self,
    ) -> None:
        fact_store = FactStore(await ApexStorage.create("sqlite+aiosqlite:///:memory:"))
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 3, 1, tzinfo=timezone.utc)

        doc_id = "doc-summary"
        doc_facts = [
            TemporalFact(fact_id="gs-rev1", subject="Revenue", predicate="was",
                         object="$100K", source_document_id=doc_id,
                         valid_from=t1, valid_to=t2, created_at=t1),
            TemporalFact(fact_id="gs-rev2", subject="Revenue", predicate="was",
                         object="$120K", source_document_id=doc_id,
                         valid_from=t2, valid_to=t3, created_at=t2,
                         parent_fact_id="gs-rev1"),
            TemporalFact(fact_id="gs-rev3", subject="Revenue", predicate="was",
                         object="$150K", source_document_id=doc_id,
                         valid_from=t3, valid_to=None, created_at=t3,
                         parent_fact_id="gs-rev2"),
            TemporalFact(fact_id="gs-hc1", subject="Headcount", predicate="was",
                         object="500", source_document_id=doc_id,
                         valid_from=t1, valid_to=None, created_at=t1),
            TemporalFact(fact_id="gs-tax1", subject="TaxRate", predicate="was",
                         object="21%", source_document_id=doc_id,
                         valid_from=t1, valid_to=t2, created_at=t1),
            TemporalFact(fact_id="gs-tax2", subject="TaxRate", predicate="was",
                         object="18%", source_document_id=doc_id,
                         valid_from=t2, valid_to=None, created_at=t2,
                         parent_fact_id="gs-tax1"),
        ]
        # Save in dependency order
        roots = [f for f in doc_facts if not f.parent_fact_id]
        branches = [f for f in doc_facts if f.parent_fact_id]
        for f in roots:
            await fact_store.save_fact(f, tenant_context="tenant-a")
        for f in branches:
            await fact_store.save_fact(f, tenant_context="tenant-a")

        reconciler = VersionChainReconciler(fact_store)
        cross = CrossChainStateReconstructor(fact_store, reconciler)

        summaries = await cross.get_chain_summaries(
            doc_id, tenant_context="tenant-a",
        )
        assert len(summaries) >= 3

        # Find Revenue summary
        rev_summary = next(s for s in summaries if s["subject"] == "Revenue")
        assert rev_summary["chain_length"] == 3
        assert rev_summary["authoritative_value"] == "$150K"
        assert rev_summary["is_healthy"] is True
        assert rev_summary["has_forks"] is False

    @pytest.mark.asyncio
    async def test_empty_doc(self, fact_store: FactStore) -> None:
        reconciler = VersionChainReconciler(fact_store)
        cross = CrossChainStateReconstructor(fact_store, reconciler)
        summaries = await cross.get_chain_summaries(
            "empty", tenant_context="tenant-a",
        )
        assert len(summaries) == 0


# ═══════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for the reconciliation engine."""

    @pytest.mark.asyncio
    async def test_chain_with_broken_link(self, fact_store: FactStore) -> None:
        """Chain with a broken superseded_by link should detect it.

        Note: The broken-link fact can't be saved to the DB because of FK
        constraints (superseded_by REFERENCES temporal_facts.fact_id).
        We test the detector's individual check method with in-memory facts
        and a pre-saved reference fact.
        """
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # Save a valid reference fact first
        await fact_store.save_fact(
            TemporalFact(fact_id="valid", subject="X", predicate="was",
                         object="ref", valid_from=t1),
            tenant_context="tenant-a",
        )
        # Save one fact with valid superseded_by, then delete the target
        fact_a = TemporalFact(fact_id="a", subject="X", predicate="was",
                              object="1", valid_from=t1,
                              superseded_by="valid")
        await fact_store.save_fact(fact_a, tenant_context="tenant-a")

        # Now create a detector and use _check_broken_links with an
        # in-memory version that references a non-existent ID
        detector = ChainGapDetector(fact_store)
        in_mem_facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was",
                         object="1", valid_from=t1,
                         superseded_by="nonexistent-id"),
        ]
        anomalies = await detector._check_broken_links(
            in_mem_facts, tenant_context="tenant-a",
        )
        broken = [a for a in anomalies if a.anomaly_type == AnomalyType.BROKEN_SUPERSEDES_LINK]
        assert len(broken) >= 1

    @pytest.mark.asyncio
    async def test_orphan_with_multiple_facts(self, fact_store: FactStore) -> None:
        """Orphaned fact with no parent and no children."""
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        facts = [
            TemporalFact(fact_id="a", subject="X", predicate="was", object="1",
                         parent_fact_id=None, valid_from=t1),
            TemporalFact(fact_id="b", subject="X", predicate="was", object="2",
                         parent_fact_id=None, valid_from=t1),
        ]
        await fact_store.save_fact(facts[0], tenant_context="tenant-a")
        await fact_store.save_fact(facts[1], tenant_context="tenant-a")

        detector = ChainGapDetector(fact_store)
        report = await detector.detect_all("X", tenant_context="tenant-a")
        # Both are orphans (no parent, no children linking them)
        orphans = [a for a in report.anomalies if a.anomaly_type == AnomalyType.ORPHANED_FACT]
        assert len(orphans) == 2  # Both are isolated

    @pytest.mark.asyncio
    async def test_broken_parent_link(self, fact_store: FactStore) -> None:
        """Chain with a broken parent_fact_id link.

        Can't save to DB with broken FK — test the check method directly.
        """
        detector = ChainGapDetector(fact_store)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        in_mem_facts = [
            TemporalFact(fact_id="a", subject="Y", predicate="was", object="1",
                         valid_from=t1, parent_fact_id="nonexistent-parent"),
        ]
        anomalies = await detector._check_broken_links(
            in_mem_facts, tenant_context="tenant-a",
        )
        broken = [a for a in anomalies if a.anomaly_type == AnomalyType.BROKEN_PARENT_LINK]
        assert len(broken) >= 1

    @pytest.mark.asyncio
    async def test_duplicate_version_numbers(self, fact_store: FactStore) -> None:
        """Facts with duplicate version numbers should be detected."""
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        facts = [
            TemporalFact(fact_id="a", subject="Z", predicate="was", object="1",
                         valid_from=t1, metadata={"version_number": 1}),
            TemporalFact(fact_id="b", subject="Z", predicate="was", object="2",
                         valid_from=t1, metadata={"version_number": 1}),
        ]
        await fact_store.save_fact(facts[0], tenant_context="tenant-a")
        await fact_store.save_fact(facts[1], tenant_context="tenant-a")

        detector = ChainGapDetector(fact_store)
        report = await detector.detect_all("Z", tenant_context="tenant-a")
        dupes = [a for a in report.anomalies if a.anomaly_type == AnomalyType.DUPLICATE_VERSION]
        assert len(dupes) >= 1

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, fact_store: FactStore) -> None:
        """Facts in different tenants should not interfere."""
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        await fact_store.save_fact(
            TemporalFact(fact_id="ta-a", subject="S", predicate="was",
                         object="A", valid_from=t1),
            tenant_context="tenant-a",
        )
        await fact_store.save_fact(
            TemporalFact(fact_id="tb-b", subject="S", predicate="was",
                         object="B", valid_from=t1),
            tenant_context="tenant-b",
        )

        reconciler = VersionChainReconciler(fact_store)
        auth_a = await reconciler.resolve_authoritative("S", tenant_context="tenant-a")
        auth_b = await reconciler.resolve_authoritative("S", tenant_context="tenant-b")

        assert auth_a is not None
        assert auth_b is not None
        assert auth_a.object == "A"
        assert auth_b.object == "B"
