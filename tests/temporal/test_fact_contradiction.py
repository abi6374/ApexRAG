"""
tests/temporal/test_fact_contradiction.py — Tests for FactContradictionDetector.

Covers:
  - ContradictionType enum and Severity enum
  - FactContradiction dataclass
  - ContradictionReport properties
  - detect_pair: VALUE_CONFLICT, WINDOW_OVERLAP, CROSS_TENANT_LINK
  - detect_all: TEMPORAL_ANOMALY, batch pairwise detection
  - detect_document: SUPERSESSION_BREAK, DB-level detection
  - Tenant enforcement
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.fact_contradiction import (
    ContradictionReport,
    ContradictionType,
    FactContradiction,
    FactContradictionDetector,
    Severity,
)
from apex_rag.temporal.fact_store import FactStore, TemporalFact

# ── Pure Unit Tests (no DB needed) ─────────────────────────────────────


class TestContradictionType:
    """ContradictionType enum values."""

    def test_enum_values(self) -> None:
        assert ContradictionType.VALUE_CONFLICT.value == "VALUE_CONFLICT"
        assert ContradictionType.WINDOW_OVERLAP.value == "WINDOW_OVERLAP"
        assert ContradictionType.TEMPORAL_ANOMALY.value == "TEMPORAL_ANOMALY"
        assert ContradictionType.SUPERSESSION_BREAK.value == "SUPERSESSION_BREAK"
        assert ContradictionType.CROSS_TENANT_LINK.value == "CROSS_TENANT_LINK"


class TestSeverity:
    """Severity enum values."""

    def test_enum_values(self) -> None:
        assert Severity.CRITICAL.value == "CRITICAL"
        assert Severity.HIGH.value == "HIGH"
        assert Severity.MEDIUM.value == "MEDIUM"
        assert Severity.LOW.value == "LOW"


class TestFactContradiction:
    """FactContradiction dataclass."""

    def test_default_frozen(self) -> None:
        c = FactContradiction(
            contradiction_type=ContradictionType.VALUE_CONFLICT,
            fact_ids=frozenset({"a", "b"}),
            subject="Revenue",
            details="Test",
        )
        assert c.contradiction_type == ContradictionType.VALUE_CONFLICT
        assert c.severity == Severity.MEDIUM  # default

    def test_custom_severity(self) -> None:
        c = FactContradiction(
            contradiction_type=ContradictionType.TEMPORAL_ANOMALY,
            fact_ids=frozenset({"a"}),
            subject="Test",
            details="Anomaly",
            severity=Severity.CRITICAL,
        )
        assert c.severity == Severity.CRITICAL


class TestContradictionReport:
    """ContradictionReport properties."""

    def test_passed_no_conflicts(self) -> None:
        report = ContradictionReport(doc_id="doc-1", tenant_id="tenant-a", fact_count=5)
        assert report.passed is True
        assert report.has_conflicts is False

    def test_has_conflicts(self) -> None:
        c = FactContradiction(
            contradiction_type=ContradictionType.VALUE_CONFLICT,
            fact_ids=frozenset({"a", "b"}),
            subject="Revenue",
            details="Conflict",
        )
        report = ContradictionReport(
            doc_id="doc-1",
            tenant_id="tenant-a",
            contradictions=[c],
            fact_count=2,
        )
        assert report.has_conflicts is True
        assert report.passed is False

    def test_critical_count(self) -> None:
        c1 = FactContradiction(
            contradiction_type=ContradictionType.TEMPORAL_ANOMALY,
            fact_ids=frozenset({"a"}),
            subject="A",
            details="1",
            severity=Severity.CRITICAL,
        )
        c2 = FactContradiction(
            contradiction_type=ContradictionType.VALUE_CONFLICT,
            fact_ids=frozenset({"b", "c"}),
            subject="B",
            details="2",
            severity=Severity.HIGH,
        )
        report = ContradictionReport(
            doc_id="doc-1",
            tenant_id="tenant-a",
            contradictions=[c1, c2],
            fact_count=3,
        )
        assert report.critical_count == 1
        assert report.high_count == 1


# ── Detection Logic Tests (no DB needed) ──────────────────────────────


class TestDetectPair:
    """FactContradictionDetector.detect_pair()."""

    def _make_detector(self) -> FactContradictionDetector:
        # No DB needed for pairwise comparisons
        return FactContradictionDetector(fact_store=None)  # type: ignore[arg-type]

    def test_value_conflict_detected(self) -> None:
        """Same subject+predicate, overlapping windows, different objects → VALUE_CONFLICT."""
        detector = self._make_detector()
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)

        # Same subject "Revenue", overlapping windows, different values
        a = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=t1,
            valid_to=t2,
        )
        b = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$50M",
            valid_from=t1,
            valid_to=t2,
        )

        import asyncio

        issues = asyncio.run(detector.detect_pair(a, b))
        assert len(issues) >= 1
        types = {i.contradiction_type for i in issues}
        assert ContradictionType.VALUE_CONFLICT in types

    def test_different_subject_no_conflict(self) -> None:
        """Different subjects → no contradictions."""
        detector = self._make_detector()
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        a = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=t1,
        )
        b = TemporalFact(
            subject="Headcount",
            predicate="was",
            object="500",
            valid_from=t1,
        )

        import asyncio

        issues = asyncio.run(detector.detect_pair(a, b))
        assert issues == []

    def test_same_value_no_conflict(self) -> None:
        """Same subject+predicate, same value → no VALUE_CONFLICT."""
        detector = self._make_detector()
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        a = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=t1,
        )
        b = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=t1,
        )

        import asyncio

        issues = asyncio.run(detector.detect_pair(a, b))
        value_conflicts = [
            i for i in issues if i.contradiction_type == ContradictionType.VALUE_CONFLICT
        ]
        assert len(value_conflicts) == 0

    def test_window_overlap_different_predicates(self) -> None:
        """Same subject, overlapping windows, different predicates → WINDOW_OVERLAP."""
        detector = self._make_detector()
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)

        a = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=t1,
            valid_to=t2,
        )
        b = TemporalFact(
            subject="Revenue",
            predicate="forecast",
            object="$45M",
            valid_from=t1,
            valid_to=t2,
        )

        import asyncio

        issues = asyncio.run(detector.detect_pair(a, b))
        overlap_issues = [
            i for i in issues if i.contradiction_type == ContradictionType.WINDOW_OVERLAP
        ]
        assert len(overlap_issues) >= 1

    def test_non_overlapping_windows_no_conflict(self) -> None:
        """Same subject, non-overlapping windows → no conflict."""
        detector = self._make_detector()

        a = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2025, 3, 31, tzinfo=timezone.utc),
        )
        b = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$50M",
            valid_from=datetime(2025, 4, 1, tzinfo=timezone.utc),
            valid_to=datetime(2025, 6, 30, tzinfo=timezone.utc),
        )

        import asyncio

        issues = asyncio.run(detector.detect_pair(a, b))
        value_conflicts = [
            i for i in issues if i.contradiction_type == ContradictionType.VALUE_CONFLICT
        ]
        assert len(value_conflicts) == 0

    def test_cross_tenant_link_detected(self) -> None:
        """parent_fact_id crosses tenant boundary → CROSS_TENANT_LINK."""
        detector = self._make_detector()
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        a = TemporalFact(
            fact_id="fact-a",
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=t1,
            tenant_id="tenant-a",
        )
        b = TemporalFact(
            fact_id="fact-b",
            subject="Revenue",
            predicate="was",
            object="$50M",
            valid_from=t1,
            tenant_id="tenant-b",
            parent_fact_id="fact-a",
        )

        import asyncio

        issues = asyncio.run(detector.detect_pair(a, b))
        cross_tenant = [
            i for i in issues if i.contradiction_type == ContradictionType.CROSS_TENANT_LINK
        ]
        assert len(cross_tenant) >= 1


class TestDetectAll:
    """FactContradictionDetector.detect_all()."""

    def test_temporal_anomaly_detected(self) -> None:
        """valid_from > valid_to → TEMPORAL_ANOMALY."""
        detector = FactContradictionDetector(fact_store=None)  # type: ignore[arg-type]

        fact = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=datetime(2025, 6, 1, tzinfo=timezone.utc),
            valid_to=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        import asyncio

        issues = asyncio.run(detector.detect_all([fact]))
        anomalies = [
            i for i in issues if i.contradiction_type == ContradictionType.TEMPORAL_ANOMALY
        ]
        assert len(anomalies) == 1

    def test_no_anomaly_when_valid_to_none(self) -> None:
        """valid_to=None → no TEMPORAL_ANOMALY."""
        detector = FactContradictionDetector(fact_store=None)  # type: ignore[arg-type]

        fact = TemporalFact(
            subject="Revenue",
            predicate="was",
            object="$40M",
            valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            valid_to=None,
        )

        import asyncio

        issues = asyncio.run(detector.detect_all([fact]))
        anomalies = [
            i for i in issues if i.contradiction_type == ContradictionType.TEMPORAL_ANOMALY
        ]
        assert len(anomalies) == 0


# ── DB-backed Tests ────────────────────────────────────────────────────


class TestDetectDocument:
    """FactContradictionDetector.detect_document() with DB."""

    @pytest_asyncio.fixture
    async def storage(self) -> AsyncGenerator[ApexStorage, None]:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            storage = await ApexStorage.create(f"sqlite+aiosqlite:///{tmp.name}")
        yield storage

    @pytest_asyncio.fixture
    async def fact_store(self, storage: ApexStorage) -> FactStore:
        return FactStore(storage)

    @pytest_asyncio.fixture
    async def detector(self, fact_store: FactStore) -> FactContradictionDetector:
        return FactContradictionDetector(fact_store)

    @pytest.mark.asyncio
    async def test_detect_document_no_issues(
        self,
        detector: FactContradictionDetector,
        fact_store: FactStore,
    ) -> None:
        """Document with consistent facts should pass."""
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 30, tzinfo=timezone.utc)
        t3 = datetime(2025, 7, 1, tzinfo=timezone.utc)

        facts = [
            TemporalFact(
                subject="Revenue",
                predicate="was",
                object="$40M",
                valid_from=t1,
                valid_to=t2,
                source_document_id="doc-123",
            ),
            TemporalFact(
                subject="Revenue",
                predicate="was",
                object="$50M",
                valid_from=t3,
                valid_to=None,
                source_document_id="doc-123",
            ),
        ]
        await fact_store.save_facts(facts, tenant_context="tenant-a")
        report = await detector.detect_document("doc-123", tenant_context="tenant-a")
        assert report.passed, f"Expected no contradictions but got: {report.contradictions}"

    @pytest.mark.asyncio
    async def test_detect_document_value_conflict(
        self,
        detector: FactContradictionDetector,
        fact_store: FactStore,
    ) -> None:
        """Overlapping windows with different values → VALUE_CONFLICT."""
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)

        facts = [
            TemporalFact(
                subject="Revenue",
                predicate="was",
                object="$40M",
                valid_from=t1,
                valid_to=t2,
                source_document_id="doc-123",
            ),
            TemporalFact(
                subject="Revenue",
                predicate="was",
                object="$50M",
                valid_from=t1,
                valid_to=t2,
                source_document_id="doc-123",
            ),
        ]
        await fact_store.save_facts(facts, tenant_context="tenant-a")
        report = await detector.detect_document("doc-123", tenant_context="tenant-a")
        assert report.has_conflicts
        value_conflicts = [
            c
            for c in report.contradictions
            if c.contradiction_type == ContradictionType.VALUE_CONFLICT
        ]
        assert len(value_conflicts) >= 1

    @pytest.mark.asyncio
    async def test_detect_document_supersession_break(
        self,
        detector: FactContradictionDetector,
        fact_store: FactStore,
    ) -> None:
        """superseded_by points to non-existent fact → SUPERSESSION_BREAK.

        Note: FK constraint prevents saving broken links. We test with
        facts that have valid links (one fact referencing another) and
        then verify the detection works by passing facts to detect_all().
        """
        fact_a = TemporalFact(
            fact_id="fact-a",
            subject="Revenue",
            predicate="was",
            object="$40M",
            source_document_id="doc-123",
        )
        fact_b = TemporalFact(
            fact_id="fact-b",
            subject="Revenue",
            predicate="was",
            object="$40M",
            superseded_by="nonexistent",
            parent_fact_id="fact-a",
            source_document_id="doc-123",
        )
        # Test detection on in-memory facts (can't save broken FK links)
        issues = await detector.detect_all([fact_a, fact_b])
        breaks = [c for c in issues if c.contradiction_type == ContradictionType.SUPERSESSION_BREAK]
        assert len(breaks) >= 1

    @pytest.mark.asyncio
    async def test_detect_document_missing_tenant(
        self,
        detector: FactContradictionDetector,
    ) -> None:
        """Missing tenant_context raises error."""
        with pytest.raises(Exception) as exc:
            await detector.detect_document("doc-123")
        assert "tenant_context" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_empty_document_passes(
        self,
        detector: FactContradictionDetector,
    ) -> None:
        """Document with no facts should pass."""
        report = await detector.detect_document("doc-empty", tenant_context="tenant-a")
        assert report.passed
        assert report.fact_count == 0
