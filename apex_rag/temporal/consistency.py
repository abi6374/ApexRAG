"""
temporal/consistency.py — Consistency Verification Engine for the Fact Store.

Verifies structural and temporal integrity of :class:`TemporalFact` objects
in the fact store.  Runs deterministic, rule-based checks to ensure data
quality and catch anomalies before they affect retrieval or synthesis.

Verification Checks:
  - SUPERSESSION_INTEGRITY:  superseded_by/parent_fact_id chains are valid.
  - TEMPORAL_ORDERING:       valid_from <= valid_to (when both set).
  - CONFIDENCE_RANGE:        confidence in [0, 1].
  - TENANT_ISOLATION:        all facts in a batch have consistent tenant_id.
  - EXTRACTION_METHOD:       known extraction method.
  - VALIDITY_WINDOW_OVERLAP: same-subject facts don't have overlapping
                             windows without supersession.
  - MISSING_REQUIRED_FIELDS: required fields are not empty.
  - DUPLICATE_FACT_ID:       duplicate fact_id in the same document.

PRINCIPLE 1 — Immutable Temporal Facts.
  Verification is always read-only.  Reports findings but never mutates.

PRINCIPLE 18 — Tenant Isolation.
  All verification methods require tenant_context.

Usage:
    verifier = ConsistencyVerifier(fact_store)
    report = await verifier.verify_document("doc-123", tenant_context="tenant-a")
    if not report.passed:
        for issue in report.issues:
            print(f"[{issue.severity}] {issue.check_name}: {issue.details}")
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apex_rag.temporal.fact_store import FactStore, TemporalFact
from apex_rag.temporal.utils import windows_overlap

logger = logging.getLogger("apex_rag.temporal.consistency")


# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════


class CheckType(str, enum.Enum):
    """Enumeration of consistency check types."""

    SUPERSESSION_INTEGRITY = "SUPERSESSION_INTEGRITY"
    """superseded_by/parent_fact_id references exist and are valid."""

    TEMPORAL_ORDERING = "TEMPORAL_ORDERING"
    """valid_from <= valid_to when both are set."""

    CONFIDENCE_RANGE = "CONFIDENCE_RANGE"
    """confidence is within [0, 1]."""

    TENANT_ISOLATION = "TENANT_ISOLATION"
    """All facts in a document share the same tenant_id."""

    EXTRACTION_METHOD = "EXTRACTION_METHOD"
    """extraction_method is from the known set."""

    VALIDITY_WINDOW_OVERLAP = "VALIDITY_WINDOW_OVERLAP"
    """Same-subject facts don't overlap without supersession."""

    MISSING_REQUIRED_FIELDS = "MISSING_REQUIRED_FIELDS"
    """Required fields (subject, predicate, object) are not empty."""

    DUPLICATE_FACT_ID = "DUPLICATE_FACT_ID"
    """No duplicate fact_id exists in the document."""


class CheckSeverity(str, enum.Enum):
    """Severity level for a verification issue."""

    ERROR = "ERROR"
    """Must fix — data integrity violation."""

    WARNING = "WARNING"
    """Should review — possible data quality issue."""

    INFO = "INFO"
    """Informational — minor anomaly."""


_KNOWN_EXTRACTION_METHODS = frozenset({
    "regex", "llm", "manual", "hybrid", "rule_based", "ml",
})


@dataclass(frozen=True)
class VerificationIssue:
    """A single issue found during consistency verification.

    Attributes:
        check_type: The type of check that failed.
        fact_id:    The fact ID involved (or None for document-level checks).
        subject:    The fact subject involved.
        details:    Human-readable description.
        severity:   Severity level.
        metadata:   Additional context.
    """

    check_type: CheckType
    fact_id: str | None
    subject: str
    details: str
    severity: CheckSeverity = CheckSeverity.WARNING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    """Report from a consistency verification run.

    Attributes:
        doc_id:    The document that was verified.
        tenant_id: The tenant context.
        issues:    List of issues found.
        fact_count: Number of facts checked.
        passed:    True if no ERROR or WARNING issues were found.
    """

    doc_id: str
    tenant_id: str
    issues: list[VerificationIssue] = field(default_factory=list)
    fact_count: int = 0
    verified_at: datetime = field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        return all(
            i.severity not in (CheckSeverity.ERROR, CheckSeverity.WARNING)
            for i in self.issues
        )

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == CheckSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == CheckSeverity.WARNING)


# ═══════════════════════════════════════════════════════════════════════
# ConsistencyVerifier
# ═══════════════════════════════════════════════════════════════════════


class ConsistencyVerifier:
    """Verifies structural and temporal consistency of TemporalFact objects.

    All checks are deterministic — no LLM calls, no embeddings.
    Designed to run as a background validation step after ingestion
    or on-demand for data quality auditing.
    """

    def __init__(self, fact_store: FactStore) -> None:
        self._fact_store = fact_store

    # ── Public API ─────────────────────────────────────────────────────────

    async def verify_fact(
        self,
        fact: TemporalFact,
    ) -> list[VerificationIssue]:
        """Run all single-fact consistency checks.

        Checks:
          - TEMPORAL_ORDERING: valid_from <= valid_to
          - CONFIDENCE_RANGE:  confidence in [0, 1]
          - EXTRACTION_METHOD: known method
          - MISSING_REQUIRED_FIELDS: subject, predicate, object not empty

        Args:
            fact: The fact to verify.

        Returns:
            List of issues found for this single fact.
        """
        issues: list[VerificationIssue] = []

        # MISSING_REQUIRED_FIELDS
        for field_name, value in [
            ("subject", fact.subject),
            ("predicate", fact.predicate),
            ("object", fact.object),
        ]:
            if not value or not value.strip():
                issues.append(VerificationIssue(
                    check_type=CheckType.MISSING_REQUIRED_FIELDS,
                    fact_id=fact.fact_id,
                    subject=fact.subject,
                    details=f"Required field '{field_name}' is empty.",
                    severity=CheckSeverity.ERROR,
                    metadata={"field": field_name},
                ))

        # TEMPORAL_ORDERING
        if fact.valid_from and fact.valid_to and fact.valid_from > fact.valid_to:
            issues.append(VerificationIssue(
                check_type=CheckType.TEMPORAL_ORDERING,
                fact_id=fact.fact_id,
                subject=fact.subject,
                details=(
                    f"valid_from ({fact.valid_from.isoformat()}) is after "
                    f"valid_to ({fact.valid_to.isoformat()})."
                ),
                severity=CheckSeverity.ERROR,
                metadata={
                    "valid_from": fact.valid_from.isoformat(),
                    "valid_to": fact.valid_to.isoformat(),
                },
            ))

        # CONFIDENCE_RANGE
        if not (0.0 <= fact.confidence <= 1.0):
            issues.append(VerificationIssue(
                check_type=CheckType.CONFIDENCE_RANGE,
                fact_id=fact.fact_id,
                subject=fact.subject,
                details=f"Confidence {fact.confidence} is outside [0, 1] range.",
                severity=CheckSeverity.ERROR,
                metadata={"confidence": fact.confidence},
            ))

        # EXTRACTION_METHOD
        if fact.extraction_method not in _KNOWN_EXTRACTION_METHODS:
            issues.append(VerificationIssue(
                check_type=CheckType.EXTRACTION_METHOD,
                fact_id=fact.fact_id,
                subject=fact.subject,
                details=(
                    f"Extraction method '{fact.extraction_method}' is not "
                    f"in the known set: {sorted(_KNOWN_EXTRACTION_METHODS)}."
                ),
                severity=CheckSeverity.WARNING,
                metadata={
                    "extraction_method": fact.extraction_method,
                    "known_methods": sorted(_KNOWN_EXTRACTION_METHODS),
                },
            ))

        return issues

    async def verify_document(
        self,
        doc_id: str,
        *,
        tenant_context: str | None = None,
    ) -> VerificationReport:
        """Run all consistency checks on a document's facts.

        Fetches all facts for the document and runs:
          - ``verify_fact()`` on each fact
          - SUPERSESSION_INTEGRITY: links point to existing facts
          - TENANT_ISOLATION: all facts share the same tenant
          - VALIDITY_WINDOW_OVERLAP: same-subject windows don't overlap
            without supersession
          - DUPLICATE_FACT_ID: no duplicate fact_ids

        Args:
            doc_id:          The document ID to verify.
            tenant_context:  Required tenant ID.

        Returns:
            A :class:`VerificationReport` with all findings.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for verify_document."
            )

        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )

        issues: list[VerificationIssue] = []
        fact_ids = {f.fact_id for f in facts}

        # Per-fact checks
        for fact in facts:
            issues.extend(await self.verify_fact(fact))

        # TENANT_ISOLATION: all facts should have the same tenant
        tenant_ids = {f.tenant_id for f in facts}
        if len(tenant_ids) > 1:
            issues.append(VerificationIssue(
                check_type=CheckType.TENANT_ISOLATION,
                fact_id=None,
                subject="(document-level)",
                details=(
                    f"Document '{doc_id}' contains facts from multiple "
                    f"tenants: {tenant_ids}. Expected all facts to have "
                    f"tenant_id='{tenant_context}'."
                ),
                severity=CheckSeverity.ERROR,
                metadata={"tenant_ids": list(tenant_ids)},
            ))

        # DUPLICATE_FACT_ID
        if len(fact_ids) < len(facts):
            from collections import Counter
            id_counts = Counter(f.fact_id for f in facts)
            for fid, count in id_counts.items():
                if count > 1:
                    issues.append(VerificationIssue(
                        check_type=CheckType.DUPLICATE_FACT_ID,
                        fact_id=fid,
                        subject="(duplicate)",
                        details=f"fact_id '{fid[:8]}' appears {count} times.",
                        severity=CheckSeverity.ERROR,
                        metadata={"fact_id": fid, "count": count},
                    ))

        # SUPERSESSION_INTEGRITY: verify links
        for fact in facts:
            for link_name, link_val in [
                ("superseded_by", fact.superseded_by),
                ("parent_fact_id", fact.parent_fact_id),
            ]:
                if link_val and link_val not in fact_ids and not link_val.startswith("__DELETED__"):
                    issues.append(VerificationIssue(
                        check_type=CheckType.SUPERSESSION_INTEGRITY,
                        fact_id=fact.fact_id,
                        subject=fact.subject,
                        details=(
                            f"'{link_name}' references fact "
                            f"'{link_val[:8]}' which does not exist "
                            f"in document '{doc_id}'."
                        ),
                        severity=CheckSeverity.ERROR,
                        metadata={
                            "link_field": link_name,
                            "target_fact_id": link_val,
                            "source_fact_id": fact.fact_id,
                        },
                    ))

        # VALIDITY_WINDOW_OVERLAP: group by subject, check for overlaps
        from collections import defaultdict
        by_subject: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            by_subject[fact.subject].append(fact)

        for subject, subject_facts in by_subject.items():
            for i in range(len(subject_facts)):
                for j in range(i + 1, len(subject_facts)):
                    a, b = subject_facts[i], subject_facts[j]

                    # Skip if one supersedes the other
                    if (a.superseded_by == b.fact_id or b.superseded_by == a.fact_id or
                            a.parent_fact_id == b.fact_id or b.parent_fact_id == a.fact_id):
                        continue

                    if windows_overlap(
                        a.valid_from, a.valid_to,
                        b.valid_from, b.valid_to,
                    ):
                        issues.append(VerificationIssue(
                            check_type=CheckType.VALIDITY_WINDOW_OVERLAP,
                            fact_id=a.fact_id,
                            subject=subject,
                            details=(
                                f"Facts '{a.fact_id[:8]}' and '{b.fact_id[:8]}' "
                                f"have overlapping validity windows for subject "
                                f"'{subject}' without supersession."
                            ),
                            severity=CheckSeverity.WARNING,
                            metadata={
                                "fact_a_id": a.fact_id,
                                "fact_b_id": b.fact_id,
                                "fact_a": {
                                    "from": a.valid_from.isoformat() if a.valid_from else None,
                                    "to": a.valid_to.isoformat() if a.valid_to else None,
                                },
                                "fact_b": {
                                    "from": b.valid_from.isoformat() if b.valid_from else None,
                                    "to": b.valid_to.isoformat() if b.valid_to else None,
                                },
                            },
                        ))

        return VerificationReport(
            doc_id=doc_id,
            tenant_id=tenant_context,
            issues=issues,
            fact_count=len(facts),
        )


