"""
temporal/historical_state.py — Historical State Engine.

Computes deltas between document/fact states at different points in
time and provides methods to traverse state history efficiently.

PRINCIPLE 4 — O(log n) Validity Resolution.
  All temporal lookups delegate to FactStore's indexed queries on
  ``valid_from`` / ``valid_to`` via the ``ix_facts_valid`` index.

PRINCIPLE 5 — Lazy Snapshot Construction.
  State is resolved on demand.  No pre-built snapshots are created
  during ingestion.  Deltas are computed only when queried.

PRINCIPLE 15 — Immutable Snapshots.
  SnapshotDelta objects are frozen (immutable).  New deltas are
  created for subsequent time points rather than mutating existing ones.

Usage:
    engine = HistoricalStateEngine(fact_store, storage)
    delta = await engine.compute_delta("doc-123", t1, t2, tenant_context="tenant-a")
    state = await engine.get_state_at("doc-123", t1, tenant_context="tenant-a")
    range_deltas = await engine.compute_range("doc-123", t1, t3, tenant_context="tenant-a")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.fact_store import FactStore, TemporalFact
from apex_rag.temporal.fact_validity import FactValidityResolver
from apex_rag.temporal.snapshot_models import SnapshotDelta, StatePatch

logger = logging.getLogger("apex_rag.temporal.historical_state")


class HistoricalStateEngine:
    """Computes deltas between document/fact states at different points in time.

    Provides:
      - ``compute_delta()`` —  Diff between two points in time.
      - ``compute_range()`` — Sequence of deltas across a time range.
      - ``get_state_at()`` —  Full state reconstruction at a point in time.
      - ``get_fact_history()`` — History of changes for a specific subject.

    All methods delegate to :class:`FactValidityResolver` / :class:`FactStore`
    for indexed temporal queries (Principle 4).
    """

    def __init__(
        self,
        fact_store: FactStore,
        storage: ApexStorage,
    ) -> None:
        self._fact_store = fact_store
        self._storage = storage
        self._resolver = FactValidityResolver(fact_store)

    # ── State at a Point in Time ───────────────────────────────────────

    async def get_state_at(
        self,
        doc_id: str,
        as_of: datetime,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, Any]:
        """Reconstruct the full fact state at a specific point in time.

        PRINCIPLE 5 — Lazy Snapshot Construction.
        The state is built on demand by querying facts valid at ``as_of``.

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime.
            tenant_context:  Required tenant ID.

        Returns:
            A dict mapping ``subject → object`` for all facts valid at
            the given time.  If multiple facts share a subject, the
            most recently active one wins.
        """
        facts = await self._resolver.resolve_at_time(
            doc_id,
            as_of,
            tenant_context=tenant_context,
        )

        # Build a subject→value map (latest wins for duplicates)
        state: dict[str, Any] = {}
        for fact in facts:
            subject = fact.subject
            # If the subject already exists, prefer the fact with later created_at
            existing = state.get(subject)
            if existing is None or fact.created_at > existing.get(
                "_created_at", datetime.min.replace(tzinfo=timezone.utc)
            ):
                state[subject] = {
                    "value": fact.object,
                    "fact_id": fact.fact_id,
                    "confidence": fact.confidence,
                    "source_document_id": fact.source_document_id,
                    "_created_at": fact.created_at,
                }

        # Strip internal keys for clean output
        result: dict[str, Any] = {}
        for subject, data in state.items():
            result[subject] = {k: v for k, v in data.items() if not k.startswith("_")}

        return result

    async def get_raw_facts_at(
        self,
        doc_id: str,
        as_of: datetime,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Get the raw facts valid at a point in time.

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime.
            tenant_context:  Required tenant ID.

        Returns:
            List of facts valid at the given time.
        """
        return await self._resolver.resolve_at_time(
            doc_id,
            as_of,
            tenant_context=tenant_context,
        )

    # ── Delta Computation ──────────────────────────────────────────────

    async def compute_delta(
        self,
        doc_id: str,
        from_time: datetime,
        to_time: datetime,
        *,
        tenant_context: str | None = None,
    ) -> SnapshotDelta:
        """Compute the delta between two points in time.

        PRINCIPLE 4 — O(log n) lookups via FactStore indexed queries.

        Args:
            doc_id:          The document ID.
            from_time:       The earlier datetime (base).
            to_time:         The later datetime (target).
            tenant_context:  Required tenant ID.

        Returns:
            A :class:`SnapshotDelta` describing all changes between
            the two points.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError("tenant_context is required for compute_delta.")

        # Fetch facts at both points
        base_facts = await self._resolver.resolve_at_time(
            doc_id,
            from_time,
            tenant_context=tenant_context,
        )
        target_facts = await self._resolver.resolve_at_time(
            doc_id,
            to_time,
            tenant_context=tenant_context,
        )

        # Index by fact_id
        base_by_id: dict[str, TemporalFact] = {f.fact_id: f for f in base_facts}
        target_by_id: dict[str, TemporalFact] = {f.fact_id: f for f in target_facts}

        base_ids = set(base_by_id.keys())
        target_ids = set(target_by_id.keys())

        added_ids = target_ids - base_ids
        removed_ids = base_ids - target_ids
        common_ids = base_ids & target_ids

        # Detect modified subjects (same fact_id, different value)
        modified_subjects: dict[str, dict[str, Any]] = {}
        for fid in common_ids:
            b = base_by_id[fid]
            t = target_by_id[fid]
            if b.object != t.object or b.confidence != t.confidence:
                modified_subjects[b.subject] = {
                    "before": b.object,
                    "after": t.object,
                    "fact_id": fid,
                    "before_confidence": b.confidence,
                    "after_confidence": t.confidence,
                }

        return SnapshotDelta(
            doc_id=doc_id,
            tenant_id=tenant_context,
            base_as_of=from_time,
            target_as_of=to_time,
            added_fact_ids=added_ids,
            removed_fact_ids=removed_ids,
            modified_subjects=modified_subjects,
            metadata={
                "base_fact_count": len(base_facts),
                "target_fact_count": len(target_facts),
                "computed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def compute_range(
        self,
        doc_id: str,
        from_time: datetime,
        to_time: datetime,
        *,
        num_intervals: int = 5,
        tenant_context: str | None = None,
    ) -> list[SnapshotDelta]:
        """Compute a sequence of deltas across a time range, evenly spaced.

        Splits the range ``[from_time, to_time]`` into ``num_intervals``
        sub-intervals and computes a delta for each.  Useful for
        visualizing how state evolved over a period.

        Args:
            doc_id:           The document ID.
            from_time:        Start of the range.
            to_time:          End of the range.
            num_intervals:    Number of sub-intervals (default 5).
            tenant_context:   Required tenant ID.

        Returns:
            An ordered list of :class:`SnapshotDelta` objects, one per
            sub-interval.
        """
        if num_intervals < 2:
            raise ValueError("num_intervals must be >= 2")

        total_span_secs = (to_time - from_time).total_seconds()
        interval_secs = total_span_secs / (num_intervals - 1)
        interval_step = timedelta(seconds=interval_secs)

        deltas: list[SnapshotDelta] = []
        for i in range(num_intervals - 1):
            t1 = from_time + (interval_step * i) if i > 0 else from_time
            t2 = from_time + (interval_step * (i + 1))
            delta = await self.compute_delta(
                doc_id,
                t1,
                t2,
                tenant_context=tenant_context,
            )
            deltas.append(delta)

        return deltas

    # ── State Patch Construction ───────────────────────────────────────

    async def build_patch(
        self,
        doc_id: str,
        from_time: datetime,
        to_time: datetime,
        *,
        tenant_context: str | None = None,
    ) -> StatePatch:
        """Build a :class:`StatePatch` between two points in time.

        The patch contains the full delta between the two states and
        can be used to transform a base state into the target state.

        Args:
            doc_id:          The document ID.
            from_time:       The base datetime.
            to_time:         The target datetime.
            tenant_context:  Required tenant ID.

        Returns:
            A :class:`StatePatch` that transforms the base state to the
            target state.
        """
        delta = await self.compute_delta(
            doc_id,
            from_time,
            to_time,
            tenant_context=tenant_context,
        )
        return StatePatch(
            doc_id=doc_id,
            tenant_id=tenant_context or "default",
            base_as_of=from_time,
            target_as_of=to_time,
            deltas=[delta],
        )

    # ── Fact History ──────────────────────────────────────────────────

    async def get_fact_history(
        self,
        doc_id: str,
        subject: str,
        *,
        tenant_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get the full change history of a specific subject over time.

        Traces all versions of a fact with the given subject in
        chronological order.

        Args:
            doc_id:          The document ID.
            subject:         The fact subject to trace.
            tenant_context:  Required tenant ID.

        Returns:
            Ordered list of dicts, each representing a version of the
            fact with keys: value, confidence, valid_from, valid_to,
            created_at, fact_id.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError("tenant_context is required for get_fact_history.")

        facts = await self._fact_store.get_facts_by_document(
            doc_id,
            tenant_context=tenant_context,
        )
        subject_facts = [f for f in facts if f.subject == subject]
        subject_facts.sort(key=lambda f: f.valid_from)

        return [
            {
                "fact_id": f.fact_id,
                "value": f.object,
                "confidence": f.confidence,
                "valid_from": f.valid_from,
                "valid_to": f.valid_to,
                "created_at": f.created_at,
                "parent_fact_id": f.parent_fact_id,
            }
            for f in subject_facts
        ]

    # ── Utility ────────────────────────────────────────────────────────

    async def list_subjects(
        self,
        doc_id: str,
        *,
        as_of: datetime | None = None,
        tenant_context: str | None = None,
    ) -> list[str]:
        """List all distinct fact subjects for a document.

        Args:
            doc_id:          The document ID.
            as_of:           Optional — only include subjects active at
                             this time.
            tenant_context:  Required tenant ID.

        Returns:
            Sorted list of distinct subject names.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError("tenant_context is required for list_subjects.")

        if as_of is not None:
            facts = await self._resolver.resolve_at_time(
                doc_id,
                as_of,
                tenant_context=tenant_context,
            )
        else:
            facts = await self._fact_store.get_facts_by_document(
                doc_id,
                tenant_context=tenant_context,
            )

        subjects = sorted({f.subject for f in facts})
        return subjects

    async def compare_states(
        self,
        doc_id: str,
        time_a: datetime,
        time_b: datetime,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, Any]:
        """Compare two states and return a structured comparison.

        Higher-level convenience that returns both states and their delta
        in a single dict.

        Args:
            doc_id:          The document ID.
            time_a:          First point in time.
            time_b:          Second point in time.
            tenant_context:  Required tenant ID.

        Returns:
            Dict with keys: state_a, state_b, delta, summary.
        """
        state_a = await self.get_state_at(
            doc_id,
            time_a,
            tenant_context=tenant_context,
        )
        state_b = await self.get_state_at(
            doc_id,
            time_b,
            tenant_context=tenant_context,
        )
        delta = await self.compute_delta(
            doc_id,
            time_a,
            time_b,
            tenant_context=tenant_context,
        )

        # Summary
        subjects_a = set(state_a.keys())
        subjects_b = set(state_b.keys())
        summary = {
            "added_subjects": sorted(subjects_b - subjects_a),
            "removed_subjects": sorted(subjects_a - subjects_b),
            "common_subjects": sorted(subjects_a & subjects_b),
            "changed_count": delta.change_count,
        }

        return {
            "time_a": time_a.isoformat(),
            "time_b": time_b.isoformat(),
            "state_a": state_a,
            "state_b": state_b,
            "delta": delta.model_dump(),
            "summary": summary,
        }
