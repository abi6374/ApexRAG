"""
temporal/snapshot_models.py — SnapshotDelta & StatePatch data models.

Defines the core abstractions for delta-encoded state snapshots:

- **SnapshotDelta**: Represents the computed diff between two states
  at different points in time.  Captures added, removed, and modified
  facts/values so a full baseline can be reconstructed incrementally.

- **StatePatch**: A composable patch that can be applied to a base
  state to produce a new state.  Supports sequential application of
  multiple deltas (delta chain).

PRINCIPLE 5 — Lazy Snapshot Construction.
  Snapshots are NEVER built during ingestion.  They are computed on
  demand when queried.  Delta encoding makes this efficient by only
  storing changes between points in time.

PRINCIPLE 15 — Immutable Snapshots.
  Once created, a SnapshotDelta is never mutated.  New deltas are
  created for subsequent time points.

Usage:
    delta = SnapshotDelta(
        base_as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
        target_as_of=datetime(2025, 6, 1, tzinfo=timezone.utc),
        added_fact_ids={"fact-1", "fact-2"},
        removed_fact_ids={"fact-old"},
        modified_subjects={"Revenue": {"after": "$60M", "before": "$40M"}},
    )

    patch = StatePatch(base_state={})
    patch = await patch.apply(delta)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
# SnapshotDelta
# ═══════════════════════════════════════════════════════════════


class SnapshotDelta(BaseModel):
    """Immutable representation of the diff between two states in time.

    Records what changed between ``base_as_of`` and ``target_as_of``
    for a specific document/entity.  Designed to be serialised and
    stored so that full snapshots can be reconstructed by replaying
    deltas from a known baseline.

    Attributes:
        delta_id:         Unique identifier (UUID4).
        doc_id:           Document or entity this delta applies to.
        tenant_id:        Tenant isolation boundary.
        base_as_of:       The datetime of the base state (earlier).
        target_as_of:     The datetime of the target state (later).
        added_fact_ids:   Set of fact_ids that appeared between the two points.
        removed_fact_ids: Set of fact_ids that disappeared between the two points.
        modified_subjects: Dict of subject → {before, after} for subjects whose
                          value changed between the two points.
        changed_edges:    Dict of edge_id → {before_type, after_type} for
                          causal edges that changed type/strength.
        metadata:         Optional additional metadata (e.g. computation notes).
        created_at:       When this delta was computed.
    """

    delta_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    tenant_id: str = "default"
    base_as_of: datetime
    target_as_of: datetime
    added_fact_ids: set[str] = Field(default_factory=set)
    removed_fact_ids: set[str] = Field(default_factory=set)
    modified_subjects: dict[str, dict[str, Any]] = Field(default_factory=dict)
    changed_edges: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}  # Immutable (Principle 15)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def time_span_seconds(self) -> float:
        """The time span covered by this delta in seconds."""
        return (self.target_as_of - self.base_as_of).total_seconds()

    @property
    def is_empty(self) -> bool:
        """True if no changes were detected."""
        return (
            not self.added_fact_ids
            and not self.removed_fact_ids
            and not self.modified_subjects
            and not self.changed_edges
        )

    @property
    def change_count(self) -> int:
        """Total number of individual changes (adds + removes + modifications)."""
        return (
            len(self.added_fact_ids)
            + len(self.removed_fact_ids)
            + len(self.modified_subjects)
            + len(self.changed_edges)
        )

    # ── Merge ──────────────────────────────────────────────────────────

    def merge(self, other: SnapshotDelta) -> SnapshotDelta:
        """Merge another delta that follows sequentially in time.

        Creates a combined delta from ``self.base_as_of`` to
        ``other.target_as_of``.  Both deltas must be for the same
        document and tenant.

        Args:
            other: A subsequent delta (its ``base_as_of`` must equal
                   ``self.target_as_of``).

        Returns:
            A new merged SnapshotDelta.

        Raises:
            ValueError: If the deltas are not sequential or belong to
                        different documents/tenants.
        """
        if other.base_as_of != self.target_as_of:
            raise ValueError(
                f"Cannot merge: self.target_as_of={self.target_as_of} != "
                f"other.base_as_of={other.base_as_of}.  Deltas must be sequential."
            )
        if other.doc_id != self.doc_id:
            raise ValueError(f"Cannot merge: different doc_id ({self.doc_id} vs {other.doc_id}).")
        if other.tenant_id != self.tenant_id:
            raise ValueError(
                f"Cannot merge: different tenant_id ({self.tenant_id} vs {other.tenant_id})."
            )

        # Fact sets: remove any that were added then later removed
        net_added = (self.added_fact_ids | other.added_fact_ids) - other.removed_fact_ids
        net_removed = (self.removed_fact_ids | other.removed_fact_ids) - other.added_fact_ids

        # Modified subjects: later wins for overlapping keys
        merged_subjects = {}
        merged_subjects.update(self.modified_subjects)
        merged_subjects.update(other.modified_subjects)

        # Changed edges: later wins for overlapping keys
        merged_edges = {}
        merged_edges.update(self.changed_edges)
        merged_edges.update(other.changed_edges)

        return SnapshotDelta(
            doc_id=self.doc_id,
            tenant_id=self.tenant_id,
            base_as_of=self.base_as_of,
            target_as_of=other.target_as_of,
            added_fact_ids=net_added,
            removed_fact_ids=net_removed,
            modified_subjects=merged_subjects,
            changed_edges=merged_edges,
            metadata={
                "merged_from": [self.delta_id, other.delta_id],
                "original_spans": [
                    {"from": self.base_as_of.isoformat(), "to": self.target_as_of.isoformat()},
                    {"from": other.base_as_of.isoformat(), "to": other.target_as_of.isoformat()},
                ],
            },
        )


# ═══════════════════════════════════════════════════════════════
# StatePatch
# ═══════════════════════════════════════════════════════════════


class StatePatch(BaseModel):
    """A composable patch that transforms a base state via ordered deltas.

    ``StatePatch`` represents the ability to go from a known base state
    (e.g. at time T₀) to a target state (at time Tₙ) by applying a
    sequence of ``SnapshotDelta`` objects.

    PRINCIPLE 5 — Lazy Snapshot Construction.
    Patches are computed on demand and are never persisted ahead of time
    (though the underlying deltas may be cached).

    Attributes:
        patch_id:       Unique identifier (UUID4).
        doc_id:         Document or entity this patch applies to.
        tenant_id:      Tenant isolation boundary.
        base_as_of:     The datetime of the base state (earliest).
        target_as_of:   The datetime of the target state (latest).
        deltas:         Ordered list of deltas to apply sequentially.
        base_snapshot_id: Optional ID of the base snapshot this patch
                         was computed from.
        metadata:       Additional metadata.
        created_at:     When this patch was computed.
    """

    patch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    tenant_id: str = "default"
    base_as_of: datetime
    target_as_of: datetime
    deltas: list[SnapshotDelta] = Field(default_factory=list)
    base_snapshot_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def delta_count(self) -> int:
        """Number of individual deltas in this patch."""
        return len(self.deltas)

    @property
    def total_change_count(self) -> int:
        """Total number of individual changes across all deltas."""
        return sum(d.change_count for d in self.deltas)

    @property
    def merged_delta(self) -> SnapshotDelta | None:
        """The result of merging all deltas into a single combined delta.

        Returns None if there are no deltas.
        """
        if not self.deltas:
            return None
        result = self.deltas[0]
        for delta in self.deltas[1:]:
            result = result.merge(delta)
        return result

    # ── Application ────────────────────────────────────────────────────

    def apply_to_state(
        self,
        base_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply this patch's deltas to a base state dict.

        The ``base_state`` should be a snapshot of facts at ``base_as_of``.
        The result is the hypothetical state at ``target_as_of``.

        The state dict is expected to be a mapping of ``subject → value``
        (e.g. ``{"Revenue": "$40M", "Headcount": 500}``).

        Args:
            base_state: The state dictionary at ``base_as_of``.

        Returns:
            The transformed state dictionary at ``target_as_of``.
        """
        merged = self.merged_delta
        if merged is None:
            return dict(base_state)

        result = dict(base_state)

        # Remove deleted subjects
        for removed_id in merged.removed_fact_ids:
            # Remove any key whose value references the removed fact
            result.pop(removed_id, None)

        # Apply modifications
        for subject, changes in merged.modified_subjects.items():
            after = changes.get("after")
            if after is not None:
                result[subject] = after
            else:
                result.pop(subject, None)

        # Add new subjects
        for added_id in merged.added_fact_ids:
            # Mark as present; actual value would come from the fact store
            result[added_id] = "__present__"

        return result


# ═══════════════════════════════════════════════════════════════
# SnapshotManifest (lightweight metadata for caching)
# ═══════════════════════════════════════════════════════════════


class SnapshotManifest(BaseModel):
    """Lightweight metadata about a stored or computed snapshot.

    Used by ``SnapshotEngine`` to track what snapshots are available
    without loading the full snapshot data.

    Attributes:
        manifest_id:    Unique identifier.
        doc_id:         Document or entity this snapshot covers.
        tenant_id:      Tenant isolation boundary.
        snapshot_date:  The datetime this snapshot represents.
        delta_count:    Number of deltas from the previous baseline.
        fact_count:     Number of facts in this snapshot.
        is_full:        True if this is a full (not delta-based) snapshot.
        base_manifest_id: For delta-based snapshots, the manifest of the
                         baseline snapshot.
        metadata:       Additional info (e.g. computation time, size).
        created_at:     When this manifest was created.
    """

    manifest_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    tenant_id: str = "default"
    snapshot_date: datetime
    delta_count: int = 0
    fact_count: int = 0
    is_full: bool = True
    base_manifest_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}
