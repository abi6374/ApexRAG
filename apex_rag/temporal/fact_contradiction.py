"""
temporal/fact_contradiction.py — Fact-Level Contradiction Detection Engine.

Detects contradictions and conflicts between structured :class:`TemporalFact`
objects at the fact-store level.  Unlike :class:`TemporalContradictionDetector`
(which works on raw ASTNode content via embeddings/LLM), this engine operates
on structured fact fields — subject, predicate, object, valid_from, valid_to,
confidence — enabling deterministic, rule-based contradiction detection.

Contradiction Types:
    - VALUE_CONFLICT:       Two facts with the same subject+predicate and
                            overlapping validity windows have different objects.
    - WINDOW_OVERLAP:       Two facts with the same subject have validity
                            windows that overlap without clear supersession.
    - TEMPORAL_ANOMALY:     A single fact has valid_from > valid_to.
    - SUPERSESSION_BREAK:   A fact's superseded_by or parent_fact_id points to
                            a fact that does not exist in the store.

PRINCIPLE 1 — Immutable Temporal Facts.
  Facts are never mutated.  Contradiction detection is always read-only.

PRINCIPLE 18 — Tenant Isolation.
  All detection methods require tenant_context.

Usage:
    detector = FactContradictionDetector(fact_store)
    issues = await detector.detect_all(facts)
    for issue in issues:
        print(f"{issue.contradiction_type}: {issue.details}")
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apex_rag.temporal.fact_store import FactStore, TemporalFact
from apex_rag.temporal.utils import windows_overlap

logger = logging.getLogger("apex_rag.temporal.fact_contradiction")


# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════


class ContradictionType(str, enum.Enum):
    """Enumeration of contradiction types detected by the engine."""

    VALUE_CONFLICT = "VALUE_CONFLICT"
    """Two facts with same subject+predicate and overlapping windows have different objects."""

    WINDOW_OVERLAP = "WINDOW_OVERLAP"
    """Two facts with the same subject have overlapping validity windows without supersession."""

    TEMPORAL_ANOMALY = "TEMPORAL_ANOMALY"
    """A single fact has valid_from > valid_to (backwards temporal window)."""

    SUPERSESSION_BREAK = "SUPERSESSION_BREAK"
    """A fact's superseded_by or parent_fact_id references a non-existent fact."""

    CROSS_TENANT_LINK = "CROSS_TENANT_LINK"
    """A fact references another fact in a different tenant."""


class Severity(str, enum.Enum):
    """Severity level for a detected contradiction."""

    CRITICAL = "CRITICAL"
    """Data integrity issue — should never occur in a healthy store."""

    HIGH = "HIGH"
    """Likely a genuine factual contradiction requiring human review."""

    MEDIUM = "MEDIUM"
    """Possible conflict — may be legitimate temporal evolution."""

    LOW = "LOW"
    """Minor inconsistency — informational."""


@dataclass(frozen=True)
class FactContradiction:
    """A single detected contradiction between facts.

    Attributes:
        contradiction_type: The type of contradiction detected.
        fact_ids:           The set of fact IDs involved (1 or 2).
        subject:            The fact subject involved.
        details:            Human-readable description of the contradiction.
        severity:           Severity level.
        resolution_suggestion: Suggested action to resolve.
        metadata:           Additional context (e.g. window boundaries, values).
    """

    contradiction_type: ContradictionType
    fact_ids: frozenset[str]
    subject: str
    details: str
    severity: Severity = Severity.MEDIUM
    resolution_suggestion: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContradictionReport:
    """Report from a contradiction detection run.

    Attributes:
        doc_id:          The document that was analyzed.
        tenant_id:       The tenant context.
        contradictions:  List of detected contradictions.
        fact_count:      Number of facts analyzed.
        has_conflicts:   True if any contradictions were found.
        passed:          True if no contradictions were found.
    """

    doc_id: str
    tenant_id: str
    contradictions: list[FactContradiction] = field(default_factory=list)
    fact_count: int = 0
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_conflicts(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def passed(self) -> bool:
        return not self.has_conflicts

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.contradictions if c.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for c in self.contradictions if c.severity == Severity.HIGH)


# ═══════════════════════════════════════════════════════════════════════
# FactContradictionDetector
# ═══════════════════════════════════════════════════════════════════════


class FactContradictionDetector:
    """Detects contradictions and conflicts between structured TemporalFact objects.

    Detection modes:
      - **Pairwise** (``detect_pair``): Compare two facts directly.
      - **Batch** (``detect_all``): Compare all facts in a list.
      - **Document** (``detect_document``): Fetch and analyze all facts for a doc.

    All detection is deterministic — no embeddings, no LLM calls.
    """

    def __init__(self, fact_store: FactStore) -> None:
        self._fact_store = fact_store

    # ── Public API ─────────────────────────────────────────────────────────

    async def detect_pair(
        self,
        fact_a: TemporalFact,
        fact_b: TemporalFact,
    ) -> list[FactContradiction]:
        """Detect contradictions between two facts.

        Checks:
          1. VALUE_CONFLICT — Same subject+predicate, overlapping windows,
             different objects.
          2. WINDOW_OVERLAP — Same subject, overlapping validity windows
             without supersession link.
          3. CROSS_TENANT_LINK — Facts reference each other across tenants
             (if parent_fact_id/superseded_by crosses tenant boundary).

        Args:
            fact_a: First fact to compare.
            fact_b: Second fact to compare.

        Returns:
            List of :class:`FactContradiction` objects found.
        """
        contradictions: list[FactContradiction] = []

        # Skip if different subjects — no basis for comparison
        if fact_a.subject != fact_b.subject:
            return contradictions

        same_predicate = fact_a.predicate == fact_b.predicate
        windows_overlap_bool = windows_overlap(
            fact_a.valid_from, fact_a.valid_to,
            fact_b.valid_from, fact_b.valid_to,
        )

        # 1. VALUE_CONFLICT: same subject+predicate, overlapping windows, different objects
        if same_predicate and windows_overlap_bool and fact_a.object != fact_b.object:
            contradictions.append(FactContradiction(
                contradiction_type=ContradictionType.VALUE_CONFLICT,
                fact_ids=frozenset({fact_a.fact_id, fact_b.fact_id}),
                subject=fact_a.subject,
                details=(
                    f"'{fact_a.subject}' has conflicting values: "
                    f"'{fact_a.object}' vs '{fact_b.object}' "
                    f"with overlapping validity windows."
                ),
                severity=Severity.HIGH,
                resolution_suggestion=(
                    "One fact should supersede the other. "
                    "Set superseded_by on the older fact."
                ),
                metadata={
                    "fact_a_value": fact_a.object,
                    "fact_b_value": fact_b.object,
                    "fact_a_window": {
                        "from": fact_a.valid_from.isoformat() if fact_a.valid_from else None,
                        "to": fact_a.valid_to.isoformat() if fact_a.valid_to else None,
                    },
                    "fact_b_window": {
                        "from": fact_b.valid_from.isoformat() if fact_b.valid_from else None,
                        "to": fact_b.valid_to.isoformat() if fact_b.valid_to else None,
                    },
                    "same_predicate": same_predicate,
                },
            ))

        # 2. WINDOW_OVERLAP: same subject, overlapping windows, no supersession
        if windows_overlap_bool and not same_predicate:
            contradictions.append(FactContradiction(
                contradiction_type=ContradictionType.WINDOW_OVERLAP,
                fact_ids=frozenset({fact_a.fact_id, fact_b.fact_id}),
                subject=fact_a.subject,
                details=(
                    f"'{fact_a.subject}' has facts with overlapping validity "
                    f"windows but different predicates: "
                    f"'{fact_a.predicate}: {fact_a.object}' vs "
                    f"'{fact_b.predicate}: {fact_b.object}'."
                ),
                severity=Severity.LOW,
                resolution_suggestion=(
                    "Review if these facts are compatible or if one should "
                    "supersede the other."
                ),
                metadata={
                    "fact_a": {"predicate": fact_a.predicate, "object": fact_a.object},
                    "fact_b": {"predicate": fact_b.predicate, "object": fact_b.object},
                },
            ))

        # 3. CROSS_TENANT_LINK: parent/superseded_by crosses tenants
        # Check fact_a's links against fact_b
        for link_field, link_val in [
            ("parent_fact_id", fact_a.parent_fact_id),
            ("superseded_by", fact_a.superseded_by),
        ]:
            if link_val == fact_b.fact_id and fact_a.tenant_id != fact_b.tenant_id:
                contradictions.append(self._make_cross_tenant_issue(
                    link_field, link_val, fact_a, fact_b,
                ))
        # Also check fact_b's links against fact_a
        for link_field, link_val in [
            ("parent_fact_id", fact_b.parent_fact_id),
            ("superseded_by", fact_b.superseded_by),
        ]:
            if link_val == fact_a.fact_id and fact_b.tenant_id != fact_a.tenant_id:
                contradictions.append(self._make_cross_tenant_issue(
                    link_field, link_val, fact_b, fact_a,
                ))

        return contradictions

    async def detect_all(
        self,
        facts: list[TemporalFact],
    ) -> list[FactContradiction]:
        """Run contradiction detection across all facts in a list.

        Checks every pair for pairwise contradictions, plus per-fact
        structural checks (TEMPORAL_ANOMALY).

        Args:
            facts: List of facts to analyze.

        Returns:
            All contradictions found.
        """
        contradictions: list[FactContradiction] = []
        fact_ids = {f.fact_id for f in facts}

        # Per-fact checks
        for fact in facts:
            # TEMPORAL_ANOMALY: valid_from > valid_to
            if fact.valid_from and fact.valid_to and fact.valid_from > fact.valid_to:
                contradictions.append(FactContradiction(
                    contradiction_type=ContradictionType.TEMPORAL_ANOMALY,
                    fact_ids=frozenset({fact.fact_id}),
                    subject=fact.subject,
                    details=(
                        f"Fact '{fact.subject}' has valid_from "
                        f"({fact.valid_from.isoformat()}) after valid_to "
                        f"({fact.valid_to.isoformat()})."
                    ),
                    severity=Severity.CRITICAL,
                    resolution_suggestion="Swap or correct the valid_from/valid_to values.",
                    metadata={
                        "valid_from": fact.valid_from.isoformat(),
                        "valid_to": fact.valid_to.isoformat(),
                    },
                ))

            # SUPERSESSION_BREAK: check cross-fact references
            for link_name, link_val in [
                ("superseded_by", fact.superseded_by),
                ("parent_fact_id", fact.parent_fact_id),
            ]:
                if link_val and link_val not in fact_ids and not link_val.startswith("__DELETED__"):
                    contradictions.append(FactContradiction(
                        contradiction_type=ContradictionType.SUPERSESSION_BREAK,
                        fact_ids=frozenset({fact.fact_id}),
                        subject=fact.subject,
                        details=(
                            f"'{link_name}' of fact {fact.fact_id[:8]} points to "
                            f"{link_val[:8]} which does not exist in the fact set."
                        ),
                        severity=Severity.HIGH,
                        resolution_suggestion=(
                            f"Ensure fact {link_val[:8]} exists or remove the {link_name} link."
                        ),
                        metadata={
                            "link_field": link_name,
                            "target_fact_id": link_val,
                            "source_fact_id": fact.fact_id,
                        },
                    ))

        # Pairwise checks
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                pair_issues = await self.detect_pair(facts[i], facts[j])
                contradictions.extend(pair_issues)

        return contradictions

    async def detect_document(
        self,
        doc_id: str,
        *,
        tenant_context: str | None = None,
    ) -> ContradictionReport:
        """Run contradiction detection on all facts for a document.

        Fetches facts from the store and runs ``detect_all()`` which
        handles per-fact (TEMPORAL_ANOMALY, SUPERSESSION_BREAK) and
        pairwise (VALUE_CONFLICT, WINDOW_OVERLAP, CROSS_TENANT_LINK)
        checking.

        Args:
            doc_id:          The document ID to analyze.
            tenant_context:  Required tenant ID.

        Returns:
            A :class:`ContradictionReport` with all findings.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for detect_document."
            )

        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )
        contradictions = await self.detect_all(facts)

        return ContradictionReport(
            doc_id=doc_id,
            tenant_id=tenant_context,
            contradictions=contradictions,
            fact_count=len(facts),
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_cross_tenant_issue(
        link_field: str,
        link_val: str,  # noqa: ARG004
        source_fact: TemporalFact,
        target_fact: TemporalFact,
    ) -> FactContradiction:
        """Create a CROSS_TENANT_LINK contradiction."""
        return FactContradiction(
            contradiction_type=ContradictionType.CROSS_TENANT_LINK,
            fact_ids=frozenset({source_fact.fact_id, target_fact.fact_id}),
            subject=source_fact.subject,
            details=(
                f"'{link_field}' of fact {source_fact.fact_id[:8]} "
                f"(tenant={source_fact.tenant_id}) points to fact "
                f"{target_fact.fact_id[:8]} (tenant={target_fact.tenant_id}). "
                f"Cross-tenant lineage is not allowed."
            ),
            severity=Severity.CRITICAL,
            resolution_suggestion="Fix the link to point within the same tenant.",
            metadata={
                "source_tenant": source_fact.tenant_id,
                "target_tenant": target_fact.tenant_id,
                "link_field": link_field,
            },
        )


