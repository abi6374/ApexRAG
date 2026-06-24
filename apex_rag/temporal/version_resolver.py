"""
temporal/version_resolver.py — Version resolution and authoritative node selection.

Determines the correct version of a node for any given point in time,
resolves supersession chains, identifies authoritative versions, and
prevents stale or expired data from being retrieved.

Usage:
    resolver = VersionResolver(storage)
    version = await resolver.resolve_for_date(node_id, as_of=datetime(2025, 6, 1))
    authoritative = await resolver.resolve_authoritative(node_id)
    chain = await resolver.get_lineage_chain(node_id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apex_rag.ingestion.apex_storage import (
    ApexStorage,
    NodeVersionRow,
    VersionLineageRow,
)
from apex_rag.models.unified_models import (
    ASTNode,
    TemporalNodeVersion,
    VersionLineage,
)

logger = logging.getLogger("apex_rag.temporal.version_resolver")


class VersionResolver:
    """Resolves the correct node version for any temporal query.

    Provides deterministic, vectorless version resolution supporting:
      - **Latest version** — the most current active version
      - **As-of-date** — the version valid at a specific point in time
      - **Authoritative** — the authoritative version after resolving supersessions
      - **Lineage chain** — full ancestry traversal
      - **Validity filtering** — expiry, pending, and superseded suppression

    Args:
        storage: An :class:`ApexStorage` instance.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    # ── Public API ─────────────────────────────────────────────────────────

    async def resolve_latest(
        self,
        node_id: str,
        *,
        _tenant_id: str = "default",
    ) -> TemporalNodeVersion | None:
        """Resolve the latest current version of a node.

        Args:
            node_id:   The node ID to resolve.
            tenant_id: Tenant isolation boundary.

        Returns:
            The latest :class:`TemporalNodeVersion`, or ``None`` if no
            active version exists.
        """
        versions = await self._get_versions_for_node(node_id)
        current = [v for v in versions if v.is_current and v.validity_status == "ACTIVE"]
        if not current:
            # Fallback: use the highest version number
            if versions:
                latest = max(versions, key=lambda v: v.version_number)
                return self._row_to_temporal_version(latest)
            return None
        # Return the highest-versioned current entry
        return self._row_to_temporal_version(
            max(current, key=lambda v: v.version_number)
        )

    async def resolve_for_date(
        self,
        node_id: str,
        as_of: datetime,
        *,
        _tenant_id: str = "default",
    ) -> TemporalNodeVersion | None:
        """Resolve the node version that was active at a specific datetime.

        Uses the ``effective_from`` / ``effective_to`` validity window.
        If multiple versions overlap, the highest version number wins.

        Args:
            node_id: The node ID to resolve.
            as_of:   The target datetime.

        Returns:
            The :class:`TemporalNodeVersion` active at ``as_of``, or
            ``None`` if no version was active.
        """
        versions = await self._get_versions_for_node(node_id)
        active = [
            v for v in versions
            if v.effective_from <= as_of
            and (v.effective_to is None or v.effective_to > as_of)
        ]
        if not active:
            return None
        # Return the highest version number among active entries
        return self._row_to_temporal_version(
            max(active, key=lambda v: v.version_number)
        )

    async def resolve_authoritative(
        self,
        node_id: str,
        *,
        as_of: datetime | None = None,
        tenant_id: str = "default",
    ) -> TemporalNodeVersion | None:
        """Resolve the authoritative version by following the supersession chain.

        If a node has been superseded, this method follows the chain
        to the latest version that has not been superseded in turn.
        This prevents stale data from being returned.

        Args:
            node_id: The original node ID.
            as_of:   Optional — if provided, resolves the authoritative
                     version as of that date.

        Returns:
            The authoritative :class:`TemporalNodeVersion`.
        """
        if as_of is not None:
            return await self.resolve_for_date(node_id, as_of, tenant_id=tenant_id)

        # Follow the SUPERSEDES chain to the latest
        chain = await self._get_version_lineage(node_id)
        supersession_map: dict[str, str] = {}
        for entry in chain:
            if entry.lineage_type in ("SUPERSEDES", "REPLACED_BY"):
                supersession_map[entry.target_version_id or ""] = entry.source_version_id

        current_id = node_id
        visited: set[str] = set()
        while current_id in supersession_map and current_id not in visited:
            visited.add(current_id)
            current_id = supersession_map[current_id]

        return await self.resolve_latest(current_id, tenant_id=tenant_id)

    async def get_lineage_chain(
        self,
        node_id: str,
        *,
        include_superseded: bool = True,
    ) -> list[VersionLineage]:
        """Return the full version lineage chain for a node.

        Args:
            node_id:            The node ID.
            include_superseded: Include superseded entries in the chain.

        Returns:
            An ordered list of :class:`VersionLineage` objects.
        """
        rows = await self._get_version_lineage(node_id)
        result = []
        for row in rows:
            if not include_superseded and row.lineage_type == "SUPERSEDES":
                continue
            result.append(
                VersionLineage(
                    lineage_id=row.lineage_id,
                    node_id=row.node_id,
                    doc_id=row.doc_id,
                    tenant_id=row.tenant_id,
                    source_version_id=row.source_version_id,
                    target_version_id=row.target_version_id,
                    lineage_type=row.lineage_type,
                    strength=row.strength,
                    evidence=row.evidence,
                    created_at=row.created_at,
                )
            )
        return result

    async def get_version_history(
        self,
        node_id: str,
    ) -> list[TemporalNodeVersion]:
        """Return all versions of a node, ordered by version number.

        Args:
            node_id: The node ID.

        Returns:
            A list of :class:`TemporalNodeVersion` objects.
        """
        versions = await self._get_versions_for_node(node_id)
        versions.sort(key=lambda v: v.version_number)
        return [self._row_to_temporal_version(v) for v in versions]

    async def is_superseded(self, node_id: str) -> bool:
        """Check if a node has been superseded by a newer version.

        Args:
            node_id: The node ID to check.

        Returns:
            ``True`` if the node has been superseded.
        """
        latest = await self.resolve_latest(node_id)
        if latest is None:
            return False
        # If the latest version has a different version_id, the original is superseded
        versions = await self._get_versions_for_node(node_id)
        if not versions:
            return False
        return any(
            v.superseded_by is not None and v.node_id == node_id
            for v in versions
        )

    async def filter_expired(
        self,
        nodes: list[ASTNode],
        *,
        as_of: datetime | None = None,
    ) -> list[ASTNode]:
        """Filter out nodes that are expired or not yet effective.

        Args:
            nodes: List of AST nodes to filter.
            as_of: Reference datetime (defaults to now).

        Returns:
            Nodes that are valid at the reference datetime.
        """
        ref = as_of or datetime.now(timezone.utc)
        active = []
        for node in nodes:
            meta = await self._storage.get_temporal_metadata(node.node_id)
            if meta is None:
                # No temporal metadata = assumed always valid
                active.append(node)
                continue
            if meta.validity_status not in ("ACTIVE", "PENDING"):
                continue
            eff_from = meta.effective_from or datetime.min.replace(tzinfo=timezone.utc)
            eff_to = meta.effective_to
            if eff_from <= ref and (eff_to is None or eff_to > ref):
                active.append(node)
        return active

    # ── Internal helpers ───────────────────────────────────────────────

    async def _get_versions_for_node(self, node_id: str) -> list[NodeVersionRow]:
        """Fetch all version rows for a node from storage."""
        if not hasattr(self._storage, "get_node_versions"):
            return []
        versions = await self._storage.get_node_versions(node_id)
        return list(versions) if versions else []

    async def _get_version_lineage(self, node_id: str) -> list[VersionLineageRow]:
        """Fetch version lineage rows for a node."""
        if not hasattr(self._storage, "get_version_lineage"):
            return []
        lineage = await self._storage.get_version_lineage(node_id)
        return list(lineage) if lineage else []

    @staticmethod
    def _row_to_temporal_version(row: NodeVersionRow) -> TemporalNodeVersion:
        """Convert a NodeVersionRow to a TemporalNodeVersion domain model."""
        return TemporalNodeVersion(
            version_id=row.version_id,
            node_id=row.node_id,
            content=row.content or "",
            doc_id=row.doc_id,
            tenant_id=row.tenant_id,
            created_at=row.created_at,
            effective_from=row.effective_from,
            effective_to=row.effective_to,
            version_number=row.version_number,
            revision_number=row.revision_number,
            source_timestamp=row.source_timestamp,
            is_current=row.is_current,
            superseded_by=row.superseded_by,
            previous_version=row.previous_version,
            # Infer validity_status from is_current + effective_to
            validity_status=(
                "EXPIRED" if row.effective_to is not None and row.effective_to < datetime.now(timezone.utc)
                else "ACTIVE" if row.is_current
                else "SUPERSEDED"
            ),
        )
