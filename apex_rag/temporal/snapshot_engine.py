"""
temporal/snapshot_engine.py — Snapshot Engine with Lazy Construction.

Manages the creation, retrieval, and caching of fact snapshots for
documents.  Supports both full snapshots (complete state at a point
in time) and delta-based snapshots (built incrementally from a
baseline).

PRINCIPLE 5 — Lazy Snapshot Construction.
  Snapshots are NEVER built during ingestion.  They are computed on
  demand when queried.  Delta encoding makes this efficient by only
  storing changes between points in time.

PRINCIPLE 15 — Immutable Snapshots.
  Once created, a snapshot is never mutated.  New snapshots are
  created for subsequent time points.

Architecture:
    ┌────────────────┐     ┌──────────────────┐
    │  Query(as_of)  │────▶│ SnapshotEngine    │
    └────────────────┘     │  .get_snapshot()  │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌────────────┐  ┌────────────┐  ┌──────────────┐
            │ Full       │  │ Delta      │  │ Lazy        │
            │ Snapshot   │  │ Snapshot   │  │ (on-demand) │
            └────────────┘  └────────────┘  └──────────────┘

Usage:
    engine = SnapshotEngine(historical_engine, fact_store)
    state = await engine.get_snapshot("doc-123", as_of=datetime(2025, 6, 1))
    manifest = await engine.create_snapshot("doc-123", as_of=datetime(2025, 6, 1))
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from apex_rag.ingestion.apex_storage import (
    ApexStorage,
    StateSnapshotRow,
)
from apex_rag.temporal.fact_store import FactStore
from apex_rag.temporal.historical_state import HistoricalStateEngine
from apex_rag.temporal.snapshot_models import SnapshotDelta, SnapshotManifest, StatePatch

logger = logging.getLogger("apex_rag.temporal.snapshot_engine")


class SnapshotEngine:
    """Snapshot management with lazy construction.

    Supports two modes:
      1. **Full snapshot**: Complete fact state at a point in time,
         serialised and persisted via ``StateSnapshotRow``.
      2. **Lazy / delta snapshot**: Built on demand from the current
         baseline by applying deltas.  Never persisted as a full copy.

    The engine chooses the most efficient strategy based on what's
    available:
      - If a full snapshot exists at or before the requested time,
        it's returned (or replayed forward via deltas).
      - Otherwise, a lazy snapshot is constructed from scratch via
        ``HistoricalStateEngine.get_state_at()``.
    """

    def __init__(
        self,
        historical_engine: HistoricalStateEngine,
        fact_store: FactStore,
        storage: ApexStorage,
    ) -> None:
        self._historical = historical_engine
        self._fact_store = fact_store
        self._storage = storage

        # In-memory cache: doc_id → {as_of → snapshot_data}
        self._cache: dict[str, dict[str, dict[str, Any]]] = {}
        # In-memory manifest registry: doc_id → list of manifests
        self._manifests: dict[str, list[SnapshotManifest]] = {}

    # ── Snapshot Retrieval ─────────────────────────────────────────────

    async def get_snapshot(
        self,
        doc_id: str,
        as_of: datetime,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, Any]:
        """Get the fact state snapshot at a specific point in time.

        PRINCIPLE 5 — Lazy Snapshot Construction.
        The snapshot is built on demand if not already cached or
        persisted.

        Strategy:
          1. Check in-memory cache (fastest).
          2. Check persisted ``StateSnapshotRow`` (persistent cache).
          3. If neither exists, compute a lazy snapshot from scratch
             via ``HistoricalStateEngine``.

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime.
            tenant_context:  Required tenant ID.

        Returns:
            Dict mapping ``subject → {value, confidence, fact_id}``.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError("tenant_context is required for get_snapshot.")

        cache_key = as_of.isoformat()

        # 1. Check in-memory cache
        doc_cache = self._cache.get(doc_id, {})
        if cache_key in doc_cache:
            logger.debug("Snapshot cache hit for %s @ %s", doc_id, cache_key)
            return dict(doc_cache[cache_key])

        # 2. Check persisted StateSnapshotRow
        persisted = await self._get_persisted_snapshot(doc_id, as_of)
        if persisted is not None:
            logger.debug("Persisted snapshot found for %s @ %s", doc_id, cache_key)
            self._cache.setdefault(doc_id, {})[cache_key] = persisted
            return dict(persisted)

        # 3. Compute lazy snapshot from scratch (Principle 5)
        logger.debug(
            "No cached/persisted snapshot for %s @ %s, building lazily...",
            doc_id,
            cache_key,
        )
        state = await self._historical.get_state_at(
            doc_id,
            as_of,
            tenant_context=tenant_context,
        )

        # Cache in memory
        self._cache.setdefault(doc_id, {})[cache_key] = state
        return dict(state)

    async def get_snapshot_between(
        self,
        doc_id: str,
        start: datetime,
        end: datetime,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, Any]:
        """Get the snapshot at the end of a range, using a delta from start.

        More efficient than calling ``get_snapshot()`` twice because
        it reuses the baseline if available.

        Args:
            doc_id:          The document ID.
            start:           Start of the range (baseline).
            end:             End of the range (target).
            tenant_context:  Required tenant ID.

        Returns:
            State dict at ``end``.
        """
        # Try to get baseline first
        baseline = None
        doc_cache = self._cache.get(doc_id, {})
        start_key = start.isoformat()
        if start_key in doc_cache:
            baseline = dict(doc_cache[start_key])

        if baseline is None:
            persisted = await self._get_persisted_snapshot(doc_id, start)
            if persisted is not None:
                baseline = dict(persisted)

        if baseline is not None:
            # Compute delta and apply to baseline
            delta = await self._historical.compute_delta(
                doc_id,
                start,
                end,
                tenant_context=tenant_context,
            )
            patch = StatePatch(
                doc_id=doc_id,
                tenant_id=tenant_context or "default",
                base_as_of=start,
                target_as_of=end,
                deltas=[delta],
            )
            result = patch.apply_to_state(baseline)
            # Cache result
            self._cache.setdefault(doc_id, {})[end.isoformat()] = result
            return result

        # Fallback: full lookup
        return await self.get_snapshot(doc_id, end, tenant_context=tenant_context)

    # ── Snapshot Creation ──────────────────────────────────────────────

    async def create_snapshot(
        self,
        doc_id: str,
        as_of: datetime | None = None,
        *,
        tenant_context: str | None = None,
        persist: bool = True,
    ) -> SnapshotManifest:
        """Create a full snapshot at a specific point in time.

        Optionally persists the snapshot to the database via
        ``StateSnapshotRow`` for long-term caching.

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime (defaults to now).
            tenant_context:  Required tenant ID.
            persist:         If True, persist the snapshot to the database.

        Returns:
            A :class:`SnapshotManifest` describing the created snapshot.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError("tenant_context is required for create_snapshot.")

        as_of = as_of or datetime.now(timezone.utc)

        # Build state
        state = await self._historical.get_state_at(
            doc_id,
            as_of,
            tenant_context=tenant_context,
        )

        # Update in-memory cache
        self._cache.setdefault(doc_id, {})[as_of.isoformat()] = state

        # Persist if requested
        if persist:
            await self._persist_snapshot(doc_id, as_of, state)

        # Create manifest
        manifest = SnapshotManifest(
            doc_id=doc_id,
            tenant_id=tenant_context,
            snapshot_date=as_of,
            fact_count=len(state),
            is_full=True,
            metadata={
                "source": "create_snapshot",
                "computed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._manifests.setdefault(doc_id, []).append(manifest)
        return manifest

    async def create_snapshot_from_delta(
        self,
        delta: SnapshotDelta,
        *,
        tenant_context: str | None = None,
        persist: bool = True,
    ) -> SnapshotManifest:
        """Create a snapshot by applying a delta to a known baseline.

        The delta's ``base_as_of`` must correspond to an existing
        snapshot (cached or persisted).  The result is the state at
        ``delta.target_as_of``.

        Args:
            delta:           The delta to apply.
            tenant_context:  Required tenant ID.
            persist:         If True, persist the resulting snapshot.

        Returns:
            A :class:`SnapshotManifest` for the new snapshot.
        """
        # Get baseline state
        baseline = await self.get_snapshot(
            delta.doc_id,
            delta.base_as_of,
            tenant_context=tenant_context,
        )

        # Apply delta
        patch = StatePatch(
            doc_id=delta.doc_id,
            tenant_id=tenant_context or delta.tenant_id,
            base_as_of=delta.base_as_of,
            target_as_of=delta.target_as_of,
            deltas=[delta],
        )
        new_state = patch.apply_to_state(baseline)

        # Cache
        self._cache.setdefault(delta.doc_id, {})[delta.target_as_of.isoformat()] = new_state

        # Persist if requested
        if persist:
            await self._persist_snapshot(delta.doc_id, delta.target_as_of, new_state)

        manifest = SnapshotManifest(
            doc_id=delta.doc_id,
            tenant_id=tenant_context or delta.tenant_id,
            snapshot_date=delta.target_as_of,
            fact_count=len(new_state),
            is_full=True,
            delta_count=1,
            metadata={
                "source": "create_snapshot_from_delta",
                "base_delta_id": delta.delta_id,
                "base_as_of": delta.base_as_of.isoformat(),
            },
        )
        self._manifests.setdefault(delta.doc_id, []).append(manifest)
        return manifest

    # ── Cache Management ───────────────────────────────────────────────

    async def invalidate_cache(
        self,
        doc_id: str | None = None,
    ) -> None:
        """Clear the in-memory snapshot cache.

        Args:
            doc_id: Optional — if provided, only clear cache for this
                    document.  Otherwise, clear all cached snapshots.
        """
        if doc_id:
            self._cache.pop(doc_id, None)
            self._manifests.pop(doc_id, None)
            logger.debug("Cache invalidated for doc %s", doc_id)
        else:
            self._cache.clear()
            self._manifests.clear()
            logger.debug("Full cache invalidated")

    async def list_manifests(
        self,
        doc_id: str,
    ) -> list[SnapshotManifest]:
        """List all available snapshot manifests for a document.

        Args:
            doc_id: The document ID.

        Returns:
            Ordered list of manifests (most recent first).
        """
        manifests = self._manifests.get(doc_id, [])
        manifests = list(manifests)
        manifests.sort(key=lambda m: m.snapshot_date, reverse=True)
        return manifests

    # ── Snapshot Deletion ──────────────────────────────────────────────

    async def delete_snapshot(
        self,
        doc_id: str,
        as_of: datetime,
    ) -> bool:
        """Delete a persisted snapshot from the database.

        Args:
            doc_id: The document ID.
            as_of:  The snapshot datetime to delete.

        Returns:
            True if a snapshot was found and deleted.
        """
        # Remove from in-memory cache
        doc_cache = self._cache.get(doc_id, {})
        doc_cache.pop(as_of.isoformat(), None)

        # Remove from database
        try:
            async with self._storage.session() as session:
                stmt = select(StateSnapshotRow).where(
                    StateSnapshotRow.doc_id == doc_id,
                    StateSnapshotRow.snapshot_date == as_of,
                )
                result = await session.execute(stmt)
                row = result.scalars().first()
                if row is None:
                    return False
                await session.delete(row)
                return True
        except Exception as exc:
            logger.error("Failed to delete snapshot: %s", exc)
            return False

    # ── Internal: Persistence ──────────────────────────────────────────

    async def _get_persisted_snapshot(
        self,
        doc_id: str,
        as_of: datetime,
    ) -> dict[str, Any] | None:
        """Fetch the closest persisted snapshot at or before ``as_of``.

        Uses the ``ix_state_snapshots_doc_date`` index for O(log n)
        lookup.

        Returns:
            The snapshot state dict, or None.
        """
        try:
            async with self._storage.session() as session:
                stmt = (
                    select(StateSnapshotRow)
                    .where(
                        StateSnapshotRow.doc_id == doc_id,
                        StateSnapshotRow.snapshot_date <= as_of,
                    )
                    .order_by(StateSnapshotRow.snapshot_date.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                row = result.scalars().first()
                if row is None or not row.snapshot_data:
                    return None

                data = json.loads(row.snapshot_data)
                if isinstance(data, dict):
                    return data
                return None
        except Exception as exc:
            logger.warning("Failed to fetch persisted snapshot: %s", exc)
            return None

    async def _persist_snapshot(
        self,
        doc_id: str,
        as_of: datetime,
        state: dict[str, Any],
    ) -> None:
        """Persist a snapshot to the database via ``StateSnapshotRow``."""
        try:
            async with self._storage.session() as session:
                row = StateSnapshotRow(
                    snapshot_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    snapshot_date=as_of,
                    snapshot_data=json.dumps(state, default=str),
                )
                session.add(row)
                logger.debug(
                    "Persisted snapshot for %s @ %s (%d facts)",
                    doc_id,
                    as_of.isoformat(),
                    len(state),
                )
        except Exception as exc:
            logger.warning("Failed to persist snapshot: %s", exc)
