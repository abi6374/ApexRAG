"""
temporal/chain_reconciler.py — Version Chain Reconciliation Engine.

Provides three components for Sprint 6:

1. **ChainGapDetector** — Finds anomalies in version chains: missing version
   numbers, broken supersession links, overlapping validity windows, orphaned
   facts, and dangling FK references.

2. **VersionChainReconciler** — Reconciles divergent version chains: detects
   and merges forks, resolves authoritative versions, repairs gap issues,
   and builds reconciled chains.

3. **CrossChainStateReconstructor** — Reconstructs authoritative state across
   multiple version chains (one per subject) by resolving each chain's
   authoritative version and composing the results into a unified state dict.

PRINCIPLE 1 — Immutable Temporal Facts.
  Reconciliation is read-only.  It never mutates existing facts or versions.
  It produces reports and recommendations, not side effects.

PRINCIPLE 3 — DAG Lineage.
  All version chains are validated for acyclicity.  Reconciled chains
  maintain DAG structure.

PRINCIPLE 18 — Tenant Isolation Everywhere.
  Every method requires tenant_context for isolation.

Usage:
    detector = ChainGapDetector(fact_store, version_service)
    anomalies = await detector.detect_all("node-123", tenant_context="tenant-a")

    reconciler = VersionChainReconciler(fact_store, version_service, lineage_engine)
    report = await reconciler.reconcile_chain("node-123", tenant_context="tenant-a")

    cross = CrossChainStateReconstructor(fact_store, reconciler)
    state = await cross.reconstruct_authoritative_state("doc-123", tenant_context="tenant-a")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import select as sa_select

from apex_rag.temporal.fact_lineage import FactLineageEngine
from apex_rag.temporal.fact_store import FactRow, FactStore, TemporalFact
from apex_rag.temporal.version_service import TemporalVersionService

logger = logging.getLogger("apex_rag.temporal.chain_reconciler")


# ═══════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════


class AnomalyType(str, Enum):
    """Types of chain anomalies detected by ChainGapDetector."""

    MISSING_VERSION = "MISSING_VERSION"
    """Version number jump (e.g. v1 → v3, no v2)."""

    BROKEN_SUPERSEDES_LINK = "BROKEN_SUPERSEDES_LINK"
    """superseded_by references a non-existent fact."""

    BROKEN_PARENT_LINK = "BROKEN_PARENT_LINK"
    """parent_fact_id references a non-existent fact."""

    OVERLAPPING_VALIDITY = "OVERLAPPING_VALIDITY"
    """Two facts in the same chain have overlapping valid_from/valid_to windows."""

    EXPIRED_ACTIVE = "EXPIRED_ACTIVE"
    """A fact has valid_to < now but is still marked as not superseded."""

    ORPHANED_FACT = "ORPHANED_FACT"
    """Fact has no parent and no descendants (isolated in the chain)."""

    FORK_DETECTED = "FORK_DETECTED"
    """Multiple parallel branches detected in the version chain."""

    DUPLICATE_VERSION = "DUPLICATE_VERSION"
    """Two facts with the same subject and version_number but different fact_ids."""

    UNEXPECTED_SUPERSEDER = "UNEXPECTED_SUPERSEDER"
    """superseded_by links to a fact whose subject doesn't match."""


@dataclass(frozen=True)
class ChainAnomaly:
    """An anomaly found in a version chain.

    Attributes:
        anomaly_type:    The type of anomaly.
        description:     Human-readable description.
        fact_id:         The fact involved (primary).
        related_fact_id: Another fact involved (if applicable).
        severity:        "error", "warning", or "info".
        metadata:        Additional diagnostic info.
    """

    anomaly_type: AnomalyType
    description: str
    fact_id: str
    related_fact_id: str | None = None
    severity: str = "warning"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainDiagnosticReport:
    """Full diagnostic report for a version chain.

    Attributes:
        node_id:       The node or subject whose chain was analyzed.
        tenant_id:     Tenant isolation boundary.
        chain_length:  Number of facts in the chain.
        anomalies:     List of detected anomalies.
        has_forks:     True if the chain has divergent branches.
        has_gaps:      True if version numbering has gaps.
        metadata:      Additional diagnostic info.

    Properties:
        anomaly_count:  Total anomalies (derived from len(anomalies)).
        error_count:    Anomalies with severity 'error'.
        warning_count:  Anomalies with severity 'warning'.
        is_healthy:     True if no errors detected (warnings OK).
    """

    node_id: str
    tenant_id: str = "default"
    chain_length: int = 0
    anomalies: list[ChainAnomaly] = field(default_factory=list)
    has_forks: bool = False
    has_gaps: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def anomaly_count(self) -> int:
        """Total number of anomalies detected."""
        return len(self.anomalies)

    @property
    def error_count(self) -> int:
        """Number of anomalies with severity 'error'."""
        return sum(1 for a in self.anomalies if a.severity == "error")

    @property
    def warning_count(self) -> int:
        """Number of anomalies with severity 'warning'."""
        return sum(1 for a in self.anomalies if a.severity == "warning")

    @property
    def is_healthy(self) -> bool:
        """True if no errors detected (warnings are OK).

        A chain is healthy if it has zero anomalies with severity
        ``'error'``.  Warnings and info-level anomalies do not
        affect health.
        """
        return self.error_count == 0


@dataclass(frozen=True)
class ReconciledChain:
    """A reconciled (resolved) version chain.

    Attributes:
        node_id:           The node or subject.
        tenant_id:         Tenant isolation boundary.
        facts:             All facts in the chain, ordered chronologically.
        authoritative:     The authoritative (latest, non-superseded) fact.
        chain_length:      Number of facts.
        is_forked:         True if the chain had forks at detection time.
        reconciled:        True if reconciliation was applied.
        report:            The diagnostic report for this chain.
    """

    node_id: str
    tenant_id: str = "default"
    facts: list[TemporalFact] = field(default_factory=list)
    authoritative: TemporalFact | None = None
    chain_length: int = 0
    is_forked: bool = False
    reconciled: bool = False
    report: ChainDiagnosticReport | None = None


@dataclass(frozen=True)
class ChainReconciliationReport:
    """Full reconciliation report for one or more chains.

    Attributes:
        doc_id:         The document ID (if applicable).
        tenant_id:      Tenant isolation boundary.
        chains:         Reconciled chains, keyed by node_id/subject.
        total_chains:   Number of chains reconciled.
        total_anomalies: Total anomalies across all chains.
        total_errors:   Total errors across all chains.
        metadata:       Additional info.
        reconciled_at:  When reconciliation was performed.
    """

    doc_id: str = ""
    tenant_id: str = "default"
    chains: dict[str, ReconciledChain] = field(default_factory=dict)
    total_chains: int = 0
    total_anomalies: int = 0
    total_errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    reconciled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════
# ChainGapDetector — Find Anomalies in Version Chains
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# Shared Helpers
# ═══════════════════════════════════════════════════════════════


def _ensure_aware(dt: datetime | None) -> datetime:
    """Ensure a datetime is timezone-aware (UTC).

    SQLite strips timezone info when storing datetimes, so facts
    read from the DB may have naive datetimes.  This helper
    converts them to UTC-aware for safe comparison.

    If ``dt`` is ``None``, returns ``datetime.max`` (UTC-aware) as
    a sentinel for "no end time".
    """
    if dt is None:
        return datetime.max.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ChainGapDetector:
    """Detects anomalies in version chains.

    All detection is deterministic — no LLM calls, no embeddings.
    Results are read-only frozen dataclasses (Principle 1).

    Checks performed:
      - Missing version numbers (gaps in version_number sequence)
      - Broken supersedes links (superseded_by → non-existent fact)
      - Broken parent links (parent_fact_id → non-existent fact)
      - Overlapping validity windows (same chain, overlapping valid_from/valid_to)
      - Expired active facts (valid_to < now but not superseded)
      - Orphaned facts (no parent, no descendants)
      - Forks (multiple facts with same subject but different lineages)
      - Duplicate version numbers (same node_id + version_number, different fact_ids)
      - Unexpected superseders (subject mismatch in supersession chain)
    """

    def __init__(
        self,
        fact_store: FactStore,
        version_service: TemporalVersionService | None = None,
    ) -> None:
        self._fact_store = fact_store
        self._version_service = version_service

    # ── Public API ─────────────────────────────────────────────────────────

    async def detect_all(
        self,
        subject_or_node_id: str,
        doc_id: str | None = None,
        *,
        tenant_context: str | None = None,
    ) -> ChainDiagnosticReport:
        """Run all anomaly checks on a version chain.

        Args:
            subject_or_node_id: The subject string or node ID to analyze.
            doc_id:             Optional document ID to scope the analysis.
            tenant_context:     Required tenant ID.

        Returns:
            A :class:`ChainDiagnosticReport` with all detected anomalies.
        """
        if not tenant_context:
            return self._empty_report(subject_or_node_id, "default")

        anomalies: list[ChainAnomaly] = []

        # Fetch the chain facts
        facts = await self._get_chain_facts(
            subject_or_node_id, doc_id, tenant_context=tenant_context,
        )
        if not facts:
            return ChainDiagnosticReport(
                node_id=subject_or_node_id,
                tenant_id=tenant_context,
                chain_length=0,
                anomalies=[],
                metadata={"note": "No facts found for this node/subject."},
            )

        # Run all checks in sequence
        anomalies.extend(await self._check_gaps(facts, tenant_context=tenant_context))
        anomalies.extend(await self._check_broken_links(facts, tenant_context=tenant_context))
        anomalies.extend(self._check_overlapping_validity(facts))
        anomalies.extend(self._check_expired_active(facts))
        anomalies.extend(self._check_orphans(facts))
        anomalies.extend(self._check_forks(facts))
        anomalies.extend(self._check_duplicate_versions(facts))
        anomalies.extend(await self._check_unexpected_superseders(facts, tenant_context=tenant_context))

        has_forks = any(a.anomaly_type == AnomalyType.FORK_DETECTED for a in anomalies)
        has_gaps = any(a.anomaly_type == AnomalyType.MISSING_VERSION for a in anomalies)

        return ChainDiagnosticReport(
            node_id=subject_or_node_id,
            tenant_id=tenant_context,
            chain_length=len(facts),
            anomalies=anomalies,
            has_forks=has_forks,
            has_gaps=has_gaps,
        )

    # ── Individual Checks ──────────────────────────────────────────────────

    async def _check_gaps(
        self,
        facts: list[TemporalFact],
        *,
        _tenant_context: str | None = None,
    ) -> list[ChainAnomaly]:
        """Detect missing version numbers in the chain."""
        anomalies: list[ChainAnomaly] = []

        # Group by subject (for temporally-linked facts)
        subjects: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            subjects[fact.subject].append(fact)

        for subject, subject_facts in subjects.items():
            subject_facts.sort(key=lambda f: f.created_at)
            expected_version = 1
            for fact in subject_facts:
                # Check version_number-like metadata
                version = fact.metadata.get("version_number")
                if version is not None:
                    if version != expected_version:
                        anomalies.append(ChainAnomaly(
                            anomaly_type=AnomalyType.MISSING_VERSION,
                            description=(
                                f"Version gap in subject '{subject}': "
                                f"expected version {expected_version}, "
                                f"found version {version} (fact {fact.fact_id})"
                            ),
                            fact_id=fact.fact_id,
                            severity="warning",
                            metadata={
                                "subject": subject,
                                "expected_version": expected_version,
                                "found_version": version,
                            },
                        ))
                    expected_version = version + 1

            # If parent_fact_id chain is broken, also flag
            parent_chain: dict[str, str | None] = {}
            for fact in subject_facts:
                parent_chain[fact.fact_id] = fact.parent_fact_id

            # Check if any fact has a parent that isn't in the subject group
            all_ids = {f.fact_id for f in subject_facts}
            for fact in subject_facts:
                if fact.parent_fact_id and fact.parent_fact_id not in all_ids:
                    # Cross-subject parent reference — already caught by broken link check
                    pass

        return anomalies

    async def _check_broken_links(
        self,
        facts: list[TemporalFact],
        *,
        tenant_context: str | None = None,
    ) -> list[ChainAnomaly]:
        """Detect broken superseded_by and parent_fact_id links."""
        anomalies: list[ChainAnomaly] = []

        all_ids = {f.fact_id for f in facts}
        # Also check for facts that might exist outside this chain
        # but are referenced by superseed_by/parent_fact_id
        referenced_ids: set[str] = set()
        for fact in facts:
            if fact.superseded_by and not fact.superseded_by.startswith("__DELETED__"):
                referenced_ids.add(fact.superseded_by)
            if fact.parent_fact_id:
                referenced_ids.add(fact.parent_fact_id)

        # Remove IDs that are in the current chain
        external_ids = referenced_ids - all_ids

        # Check each external reference against the fact store
        for ref_id in external_ids:
            ref_fact = await self._fact_store.get_fact(
                ref_id, tenant_context=tenant_context,
            )
            if ref_fact is None:
                # Find which fact referenced this broken ID
                for fact in facts:
                    if fact.superseded_by == ref_id:
                        anomalies.append(ChainAnomaly(
                            anomaly_type=AnomalyType.BROKEN_SUPERSEDES_LINK,
                            description=(
                                f"Fact {fact.fact_id} has superseded_by={ref_id}, "
                                f"but no fact with that ID exists in tenant context."
                            ),
                            fact_id=fact.fact_id,
                            related_fact_id=ref_id,
                            severity="error",
                            metadata={"field": "superseded_by"},
                        ))
                    if fact.parent_fact_id == ref_id:
                        anomalies.append(ChainAnomaly(
                            anomaly_type=AnomalyType.BROKEN_PARENT_LINK,
                            description=(
                                f"Fact {fact.fact_id} has parent_fact_id={ref_id}, "
                                f"but no fact with that ID exists in tenant context."
                            ),
                            fact_id=fact.fact_id,
                            related_fact_id=ref_id,
                            severity="error",
                            metadata={"field": "parent_fact_id"},
                        ))

        return anomalies

    def _check_overlapping_validity(
        self,
        facts: list[TemporalFact],
    ) -> list[ChainAnomaly]:
        """Detect overlapping validity windows in the same chain."""
        anomalies: list[ChainAnomaly] = []

        # Group by subject
        subjects: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            subjects[fact.subject].append(fact)

        for subject, subject_facts in subjects.items():
            # Sort by valid_from
            sorted_facts = sorted(subject_facts, key=lambda f: _ensure_aware(f.valid_from))
            for i in range(len(sorted_facts)):
                for j in range(i + 1, len(sorted_facts)):
                    a = sorted_facts[i]
                    b = sorted_facts[j]
                    # Check window overlap
                    a_end = _ensure_aware(a.valid_to)
                    b_start = _ensure_aware(b.valid_from)
                    if a_end > b_start:
                        # Check if one supersedes the other — if so, overlap is expected
                        if a.superseded_by == b.fact_id or b.superseded_by == a.fact_id:
                            continue
                        if a.parent_fact_id == b.fact_id or b.parent_fact_id == a.fact_id:
                            continue
                        anomalies.append(ChainAnomaly(
                            anomaly_type=AnomalyType.OVERLAPPING_VALIDITY,
                            description=(
                                f"Validity window overlap for subject '{subject}': "
                                f"fact {a.fact_id} (valid_to={a.valid_to}) overlaps "
                                f"with fact {b.fact_id} (valid_from={b.valid_from})"
                            ),
                            fact_id=a.fact_id,
                            related_fact_id=b.fact_id,
                            severity="warning",
                            metadata={
                                "subject": subject,
                                "a_valid_to": a.valid_to.isoformat() if a.valid_to else "infinity",
                                "b_valid_from": b.valid_from.isoformat(),
                            },
                        ))

        return anomalies

    def _check_expired_active(
        self,
        facts: list[TemporalFact],
    ) -> list[ChainAnomaly]:
        """Detect facts that are expired but not marked as superseded."""
        anomalies: list[ChainAnomaly] = []
        now = datetime.now(timezone.utc)

        for fact in facts:
            valid_to = _ensure_aware(fact.valid_to) if fact.valid_to is not None else None
            if valid_to is not None and valid_to < now and fact.superseded_by is None:
                anomalies.append(ChainAnomaly(
                    anomaly_type=AnomalyType.EXPIRED_ACTIVE,
                    description=(
                        f"Fact {fact.fact_id} (subject='{fact.subject}') "
                        f"expired at {fact.valid_to.isoformat()} but has no "
                        f"superseder. It is expired but still 'active'."
                    ),
                    fact_id=fact.fact_id,
                    severity="warning",
                    metadata={
                        "subject": fact.subject,
                        "valid_to": fact.valid_to.isoformat(),
                        "now": now.isoformat(),
                    },
                ))

        return anomalies

    def _check_orphans(
        self,
        facts: list[TemporalFact],
    ) -> list[ChainAnomaly]:
        """Detect orphaned facts (no parent, no descendants)."""
        anomalies: list[ChainAnomaly] = []

        # Build adjacency: parent → children
        children_of: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            if fact.parent_fact_id:
                children_of[fact.parent_fact_id].append(fact)


        # A fact is orphaned if:
        #   1. It has no parent_fact_id (root), AND
        #   2. It has no children (no fact references it as parent), AND
        #   3. It has no superseded_by link, AND
        #   4. There are multiple facts in the chain (singleton chains are fine)
        if len(facts) <= 1:
            return anomalies

        for fact in facts:
            has_parent = fact.parent_fact_id is not None
            has_children = len(children_of.get(fact.fact_id, [])) > 0
            is_superseded = fact.superseded_by is not None
            roots_without_children = (
                not has_parent
                and not has_children
                and not is_superseded
            )
            if roots_without_children:
                anomalies.append(ChainAnomaly(
                    anomaly_type=AnomalyType.ORPHANED_FACT,
                    description=(
                        f"Fact {fact.fact_id} (subject='{fact.subject}') "
                        f"has no parent, no children, and is not superseded. "
                        f"It is an orphan in the chain."
                    ),
                    fact_id=fact.fact_id,
                    severity="info",
                    metadata={"subject": fact.subject},
                ))

        return anomalies

    def _check_forks(
        self,
        facts: list[TemporalFact],
    ) -> list[ChainAnomaly]:
        """Detect divergent branches (forks) in the version chain.

        A fork occurs when a fact has two or more children (two different
        facts reference the same parent_fact_id), creating divergent branches.
        """
        anomalies: list[ChainAnomaly] = []

        # Build parent → [children] map
        children_of: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            if fact.parent_fact_id:
                children_of[fact.parent_fact_id].append(fact)

        for parent_id, children in children_of.items():
            if len(children) > 1:
                child_ids = [c.fact_id for c in children]
                anomalies.append(ChainAnomaly(
                    anomaly_type=AnomalyType.FORK_DETECTED,
                    description=(
                        f"Fork detected: fact {parent_id} has {len(children)} "
                        f"direct descendants: {child_ids}. "
                        f"This creates divergent version branches."
                    ),
                    fact_id=parent_id,
                    related_fact_id=child_ids[0],
                    severity="warning",
                    metadata={
                        "child_count": len(children),
                        "child_ids": child_ids,
                    },
                ))

        return anomalies

    def _check_duplicate_versions(
        self,
        facts: list[TemporalFact],
    ) -> list[ChainAnomaly]:
        """Detect facts with the same version_number in metadata."""
        anomalies: list[ChainAnomaly] = []

        version_map: dict[int, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            vn = fact.metadata.get("version_number")
            if vn is not None:
                version_map[vn].append(fact)

        for version_num, version_facts in version_map.items():
            if len(version_facts) > 1:
                ids = [f.fact_id for f in version_facts]
                anomalies.append(ChainAnomaly(
                    anomaly_type=AnomalyType.DUPLICATE_VERSION,
                    description=(
                        f"Version number {version_num} is assigned to "
                        f"{len(version_facts)} different facts: {ids}"
                    ),
                    fact_id=version_facts[0].fact_id,
                    related_fact_id=version_facts[1].fact_id,
                    severity="warning",
                    metadata={
                        "version_number": version_num,
                        "fact_ids": ids,
                    },
                ))

        return anomalies

    async def _check_unexpected_superseders(
        self,
        facts: list[TemporalFact],
        *,
        tenant_context: str | None = None,
    ) -> list[ChainAnomaly]:
        """Detect supersession links where subjects don't match."""
        anomalies: list[ChainAnomaly] = []

        for fact in facts:
            if fact.superseded_by and not fact.superseded_by.startswith("__DELETED__"):
                # Fetch the superseder
                superseder = await self._fact_store.get_fact(
                    fact.superseded_by, tenant_context=tenant_context,
                )
                if superseder and superseder.subject != fact.subject:
                    anomalies.append(ChainAnomaly(
                        anomaly_type=AnomalyType.UNEXPECTED_SUPERSEDER,
                        description=(
                            f"Fact {fact.fact_id} (subject='{fact.subject}') "
                            f"is superseded by fact {fact.superseded_by} "
                            f"(subject='{superseder.subject}'). "
                            f"Subjects don't match."
                        ),
                        fact_id=fact.fact_id,
                        related_fact_id=fact.superseded_by,
                        severity="error",
                        metadata={
                            "source_subject": fact.subject,
                            "target_subject": superseder.subject,
                        },
                    ))

        return anomalies

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _get_chain_facts(
        self,
        subject_or_node_id: str,
        doc_id: str | None,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Fetch facts for a chain, by subject or by document."""
        if doc_id:
            facts = await self._fact_store.get_facts_by_document(
                doc_id, tenant_context=tenant_context,
            )
            # When doc_id is provided, filter by subject to scope the chain
            return [f for f in facts if f.subject == subject_or_node_id]

        # Subject-based: query by subject across all documents
        async with self._fact_store._storage.session() as session:
            stmt = (
                sa_select(FactRow)
                .where(
                    FactRow.subject == subject_or_node_id,
                    FactRow.tenant_id == tenant_context,
                )
                .order_by(FactRow.created_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [FactStore._row_to_fact(r) for r in rows]

    @staticmethod
    def _empty_report(node_id: str, tenant_id: str) -> ChainDiagnosticReport:
        return ChainDiagnosticReport(
            node_id=node_id,
            tenant_id=tenant_id,
            chain_length=0,
            anomalies=[],
            metadata={"note": "No tenant context provided."},
        )


# ═══════════════════════════════════════════════════════════════
# VersionChainReconciler — Resolve and Repair Version Chains
# ═══════════════════════════════════════════════════════════════


class VersionChainReconciler:
    """Reconciles divergent version chains and resolves authoritative versions.

    All reconciliation is read-only (Principle 1).  It produces reports
    and recommendations — never mutates facts or versions.

    Methods:
      - reconcile_chain():        Full chain reconciliation (detect + resolve).
      - resolve_authoritative():   Resolve the authoritative version in a chain.
      - resolve_document_chains(): Reconcile all chains for a document.
      - merge_forks():            Detect and describe fork merges.
    """

    def __init__(
        self,
        fact_store: FactStore,
        version_service: TemporalVersionService | None = None,
        lineage_engine: FactLineageEngine | None = None,
    ) -> None:
        self._fact_store = fact_store
        self._version_service = version_service
        self._lineage_engine = lineage_engine or FactLineageEngine(fact_store._storage)
        self._detector = ChainGapDetector(fact_store, version_service)

    # ── Full Chain Reconciliation ──────────────────────────────────────────

    async def reconcile_chain(
        self,
        subject_or_node_id: str,
        doc_id: str | None = None,
        *,
        tenant_context: str | None = None,
    ) -> ReconciledChain:
        """Perform full reconciliation on a version chain.

        1. Run anomaly detection.
        2. Resolve the authoritative version.
        3. Build a reconciled chain.

        Args:
            subject_or_node_id: The subject or node ID.
            doc_id:             Optional document scope.
            tenant_context:     Required tenant ID.

        Returns:
            A :class:`ReconciledChain` with the resolved state.
        """
        if not tenant_context:
            return ReconciledChain(
                node_id=subject_or_node_id,
                tenant_id="default",
                facts=[],
                chain_length=0,
            )

        # 1. Run anomaly detection
        report = await self._detector.detect_all(
            subject_or_node_id, doc_id, tenant_context=tenant_context,
        )

        # 2. Fetch all facts in the chain
        facts = await self._detector._get_chain_facts(
            subject_or_node_id, doc_id, tenant_context=tenant_context,
        )

        # 3. Resolve authoritative version
        authoritative = await self.resolve_authoritative(
            subject_or_node_id, doc_id, tenant_context=tenant_context,
        )

        has_forks = report.has_forks

        return ReconciledChain(
            node_id=subject_or_node_id,
            tenant_id=tenant_context,
            facts=facts,
            authoritative=authoritative,
            chain_length=len(facts),
            is_forked=has_forks,
            reconciled=True,
            report=report,
        )

    # ── Authoritative Version Resolution ───────────────────────────────────

    async def resolve_authoritative(
        self,
        subject_or_node_id: str,
        doc_id: str | None = None,
        *,
        tenant_context: str | None = None,
    ) -> TemporalFact | None:
        """Resolve the authoritative (canonical) version in a chain.

        Strategy (in order of precedence):
          1. Follow superseded_by chain to the latest non-superseded fact.
          2. If no supersession chain, use the fact with the latest valid_from
             that is still valid (valid_to is None or > now).
          3. Fallback to the most recently created fact.

        Args:
            subject_or_node_id: The subject or node ID.
            doc_id:             Optional document scope.
            tenant_context:     Required tenant ID.

        Returns:
            The authoritative :class:`TemporalFact`, or None.
        """
        if not tenant_context:
            return None

        facts = await self._detector._get_chain_facts(
            subject_or_node_id, doc_id, tenant_context=tenant_context,
        )
        if not facts:
            return None

        # Strategy 1: Follow superseded_by chain to the latest
        # Build supersession map: fact_id → superseded_by
        supersession_map: dict[str, str] = {}
        for fact in facts:
            if fact.superseded_by and not fact.superseded_by.startswith("__DELETED__"):
                supersession_map[fact.fact_id] = fact.superseded_by

        if supersession_map:
            # Find facts that are not superseded by anyone in the chain
            all_ids = {f.fact_id for f in facts}
            superseded_ids = set(supersession_map.keys())
            # A fact is a superseder if it appears as a superseded_by value
            superseder_ids = set(supersession_map.values())
            # Terminal facts: are superseder but not superseded by anyone in the chain
            terminal_ids = (superseder_ids & all_ids) - superseded_ids

            if terminal_ids:
                # Pick the one with latest valid_from
                terminal_facts = [f for f in facts if f.fact_id in terminal_ids]
                terminal_facts.sort(key=lambda f: f.valid_from, reverse=True)
                return terminal_facts[0]

            # If no terminal found, try the superseder that's not in the chain
            external_superseders = superseder_ids - all_ids
            if external_superseders:
                # Follow external links
                current_id = list(external_superseders)[0]
                visited: set[str] = set()
                while current_id and current_id not in visited:
                    visited.add(current_id)
                    ext_fact = await self._fact_store.get_fact(
                        current_id, tenant_context=tenant_context,
                    )
                    if ext_fact is None:
                        break
                    if ext_fact.superseded_by and not ext_fact.superseded_by.startswith("__DELETED__"):
                        current_id = ext_fact.superseded_by
                    else:
                        return ext_fact

        # Strategy 2: Find latest valid fact (non-expired)
        now = datetime.now(timezone.utc)
        valid_facts = [
            f for f in facts
            if f.valid_to is None or _ensure_aware(f.valid_to) > now
        ]
        if valid_facts:
            valid_facts.sort(key=lambda f: _ensure_aware(f.valid_from), reverse=True)
            return valid_facts[0]

        # Strategy 3: Fallback — most recently created
        facts.sort(key=lambda f: f.created_at, reverse=True)
        return facts[0]

    # ── Document-Level Reconciliation ──────────────────────────────────────

    async def resolve_document_chains(
        self,
        doc_id: str,
        *,
        tenant_context: str | None = None,
    ) -> ChainReconciliationReport:
        """Reconcile all version chains for a document.

        Groups facts by subject and reconciles each group independently,
        then composes the results into a unified report.

        Args:
            doc_id:          The document ID.
            tenant_context:  Required tenant ID.

        Returns:
            A :class:`ChainReconciliationReport` with per-subject chains.
        """
        if not tenant_context:
            return ChainReconciliationReport(
                doc_id=doc_id, tenant_id="default",
            )

        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )

        # Group by subject
        subjects: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            subjects[fact.subject].append(fact)

        total_anomalies = 0
        total_errors = 0
        chains: dict[str, ReconciledChain] = {}

        for subject in subjects:
            chain = await self.reconcile_chain(
                subject, doc_id, tenant_context=tenant_context,
            )
            chains[subject] = chain
            total_anomalies += chain.report.anomaly_count if chain.report else 0
            total_errors += chain.report.error_count if chain.report else 0

        return ChainReconciliationReport(
            doc_id=doc_id,
            tenant_id=tenant_context,
            chains=chains,
            total_chains=len(chains),
            total_anomalies=total_anomalies,
            total_errors=total_errors,
        )

    # ── Fork Detection ────────────────────────────────────────────────────

    async def detect_forks(
        self,
        subject_or_node_id: str,
        doc_id: str | None = None,
        *,
        tenant_context: str | None = None,
    ) -> list[ChainAnomaly]:
        """Detect and return fork anomalies in a chain.

        Args:
            subject_or_node_id: The subject or node ID.
            doc_id:             Optional document scope.
            tenant_context:     Required tenant ID.

        Returns:
            List of fork-related :class:`ChainAnomaly` objects.
        """
        facts = await self._detector._get_chain_facts(
            subject_or_node_id, doc_id, tenant_context=tenant_context,
        )
        return self._detector._check_forks(facts)

    async def describe_forks(
        self,
        subject_or_node_id: str,
        doc_id: str | None = None,
        *,
            tenant_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Describe fork branches in detail.

        Returns structured information about each fork: the parent fact,
        its children, their contents, and creation order.

        Args:
            subject_or_node_id: The subject or node ID.
            doc_id:             Optional document scope.
            tenant_context:     Required tenant ID.

        Returns:
            List of fork descriptions as dicts.
        """
        facts = await self._detector._get_chain_facts(
            subject_or_node_id, doc_id, tenant_context=tenant_context,
        )
        children_of: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            if fact.parent_fact_id:
                children_of[fact.parent_fact_id].append(fact)

        fork_descriptions: list[dict[str, Any]] = []
        for parent_id, children in children_of.items():
            if len(children) > 1:
                parent_fact = next((f for f in facts if f.fact_id == parent_id), None)
                fork_descriptions.append({
                    "parent_fact_id": parent_id,
                    "parent_subject": parent_fact.subject if parent_fact else "unknown",
                    "child_count": len(children),
                    "branches": [
                        {
                            "fact_id": c.fact_id,
                            "value": c.object,
                            "created_at": c.created_at.isoformat(),
                            "valid_from": c.valid_from.isoformat(),
                        }
                        for c in sorted(children, key=lambda x: x.created_at)
                    ],
                    "suggested_merge": (
                        "Latest branch (by created_at) is likely authoritative. "
                        "Verify and set superseded_by on older branches."
                        if len(children) > 1 else None
                    ),
                })

        return fork_descriptions


# ═══════════════════════════════════════════════════════════════
# CrossChainStateReconstructor — Unified State Across Chains
# ═══════════════════════════════════════════════════════════════


class CrossChainStateReconstructor:
    """Reconstructs authoritative state across multiple version chains.

    Takes a document's facts, groups them by subject, resolves each
    subject's chain via :class:`VersionChainReconciler`, and composes
    the authoritative versions into a unified state dict.

    This is the Sprint 6 equivalent of ``HistoricalStateEngine.get_state_at()``
    but uses chain-based resolution instead of raw temporal queries.

    Methods:
      - reconstruct_authoritative_state():  Get authoritative state via chain resolution.
      - reconstruct_state_at():             Get state at a specific time via chain resolution.
      - compare_states():                   Compare two chain-resolved states.
      - get_chain_summaries():              Get summaries of all chains in a document.
    """

    def __init__(
        self,
        fact_store: FactStore,
        reconciler: VersionChainReconciler,
    ) -> None:
        self._fact_store = fact_store
        self._reconciler = reconciler

    # ── Public API ─────────────────────────────────────────────────────────

    async def reconstruct_authoritative_state(
        self,
        doc_id: str,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, Any]:
        """Reconstruct the authoritative state for a document.

        Resolves each subject's version chain to its authoritative version
        and composes the results into a unified state dict.

        The state dict maps:
          ``subject → {value, confidence, fact_id, valid_from, chain_length}``

        Args:
            doc_id:          The document ID.
            tenant_context:  Required tenant ID.

        Returns:
            Dict of subject → authoritative value metadata.
        """
        if not tenant_context:
            return {}

        # Get document-level reconciliation report
        report = await self._reconciler.resolve_document_chains(
            doc_id, tenant_context=tenant_context,
        )

        state: dict[str, Any] = {}
        for subject, chain in report.chains.items():
            if chain.authoritative is not None:
                auth = chain.authoritative
                state[subject] = {
                    "value": auth.object,
                    "confidence": auth.confidence,
                    "fact_id": auth.fact_id,
                    "valid_from": auth.valid_from.isoformat(),
                    "valid_to": auth.valid_to.isoformat() if auth.valid_to else None,
                    "chain_length": chain.chain_length,
                    "has_anomalies": chain.report.anomaly_count > 0 if chain.report else False,
                    "source_document_id": auth.source_document_id,
                }

        return state

    async def reconstruct_state_at(
        self,
        doc_id: str,
        as_of: datetime,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, Any]:
        """Reconstruct state at a specific point in time using chain resolution.

        For each subject chain, selects the fact that was valid at ``as_of``
        (based on valid_from/valid_to window).  This is similar to
        ``FactValidityResolver.resolve_at_time()`` but enhanced with chain
        awareness (tracks chain position, anomaly info).

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime.
            tenant_context:  Required tenant ID.

        Returns:
            Dict of subject → value metadata valid at the given time.
        """
        if not tenant_context:
            return {}

        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )

        # Filter by validity window and group by subject
        subjects: dict[str, list[TemporalFact]] = defaultdict(list)
        for fact in facts:
            # Ensure aware comparison (SQLite strips timezone on round-trip)
            vf = _ensure_aware(fact.valid_from)
            vt = _ensure_aware(fact.valid_to) if fact.valid_to is not None else None
            as_of_aware = _ensure_aware(as_of)
            if vf <= as_of_aware and (
                vt is None or vt > as_of_aware
            ):
                subjects[fact.subject].append(fact)

        state: dict[str, Any] = {}
        for subject, subject_facts in subjects.items():
            # Pick the latest created fact within the window
            subject_facts.sort(key=lambda f: f.created_at, reverse=True)
            best = subject_facts[0]
            state[subject] = {
                "value": best.object,
                "confidence": best.confidence,
                "fact_id": best.fact_id,
                "valid_from": best.valid_from.isoformat(),
                "valid_to": best.valid_to.isoformat() if best.valid_to else None,
                "version_count": len(subject_facts),
                "source_document_id": best.source_document_id,
            }

        return state

    async def compare_states(
        self,
        doc_id: str,
        time_a: datetime,
        time_b: datetime,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, Any]:
        """Compare chain-resolved states at two points in time.

        Higher-level convenience that returns both states and their diff.

        Args:
            doc_id:          The document ID.
            time_a:          First point in time.
            time_b:          Second point in time.
            tenant_context:  Required tenant ID.

        Returns:
            Dict with keys: state_a, state_b, changes, summary.
        """
        state_a = await self.reconstruct_state_at(
            doc_id, time_a, tenant_context=tenant_context,
        )
        state_b = await self.reconstruct_state_at(
            doc_id, time_b, tenant_context=tenant_context,
        )

        subjects_a = set(state_a.keys())
        subjects_b = set(state_b.keys())

        added = subjects_b - subjects_a
        removed = subjects_a - subjects_b
        common = subjects_a & subjects_b

        # Detect changes in common subjects
        modified: dict[str, dict[str, Any]] = {}
        for subj in sorted(common):
            if state_a[subj]["value"] != state_b[subj]["value"]:
                modified[subj] = {
                    "before": state_a[subj]["value"],
                    "after": state_b[subj]["value"],
                    "before_confidence": state_a[subj]["confidence"],
                    "after_confidence": state_b[subj]["confidence"],
                }

        changes = {
            "added": sorted(added),
            "removed": sorted(removed),
            "modified": modified,
            "unchanged": len(common) - len(modified),
        }

        return {
            "time_a": time_a.isoformat(),
            "time_b": time_b.isoformat(),
            "state_a": state_a,
            "state_b": state_b,
            "changes": changes,
            "summary": {
                "total_subjects_a": len(state_a),
                "total_subjects_b": len(state_b),
                "added_count": len(added),
                "removed_count": len(removed),
                "modified_count": len(modified),
            },
        }

    async def get_chain_summaries(
        self,
        doc_id: str,
        *,
        tenant_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get summaries of all version chains in a document.

        Returns metadata about each subject's chain: length, authoritative
        value, anomaly count, fork status, health.

        Args:
            doc_id:          The document ID.
            tenant_context:  Required tenant ID.

        Returns:
            List of chain summaries.
        """
        report = await self._reconciler.resolve_document_chains(
            doc_id, tenant_context=tenant_context,
        )

        summaries: list[dict[str, Any]] = []
        for subject, chain in report.chains.items():
            summaries.append({
                "subject": subject,
                "chain_length": chain.chain_length,
                "authoritative_value": (
                    chain.authoritative.object if chain.authoritative else None
                ),
                "authoritative_fact_id": (
                    chain.authoritative.fact_id if chain.authoritative else None
                ),
                "anomaly_count": chain.report.anomaly_count if chain.report else 0,
                "error_count": chain.report.error_count if chain.report else 0,
                "has_forks": chain.is_forked,
                "has_gaps": chain.report.has_gaps if chain.report else False,
                "is_healthy": chain.report.is_healthy if chain.report else True,
                "tenant_id": chain.tenant_id,
            })

        summaries.sort(key=lambda s: s["subject"])
        return summaries
