"""
graph/dags/version_dag.py — VersionDAG Builder.

Creates version chain edges from node version history with
``projection=["version"]``.

Strategies (deterministic, no LLM calls):
    1. **VERSION_OF** — Version row → its parent node
    2. **SUPERSEDES** — Newer version → older version (version n+1 → version n)
    3. **REPLACED_BY** — Older version → newer version (reverse of supersedes)
    4. **SNAPSHOT_OF** — Snapshot → node it captures
    5. **HISTORICAL_PARENT** — Version → its historical ancestor

Usage:
    builder = VersionDagBuilder(storage)
    edges = await builder.build_from_versions(version_rows, doc_id="doc-123")
    for edge in edges:
        await storage.save_knowledge_edge(edge)
"""

from __future__ import annotations

import logging
from typing import Any

from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.models.unified_models import KnowledgeEdge

logger = logging.getLogger("apex_rag.graph.dags.version_dag")


class VersionDagBuilder:
    """Builds VersionDAG edges from node version history.

    All edges carry ``projection=["version"]`` with version metadata
    including version_number, version_id, and validity_status.
    """

    def __init__(self, storage: ApexStorage | None = None) -> None:
        self._storage = storage

    async def build_from_versions(
        self,
        version_rows: list[Any],
        *,
        doc_id: str,
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build VersionDAG edges from NodeVersionRow objects.

        Args:
            version_rows: List of NodeVersionRow objects for a document.
            doc_id:       The document ID.
            tenant_id:    Tenant isolation boundary (reserved).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["version"].
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        # Group versions by node_id
        versions_by_node: dict[str, list[Any]] = {}
        for vrow in version_rows:
            nid = getattr(vrow, "node_id", "")
            versions_by_node.setdefault(nid, []).append(vrow)

        for node_id, versions in versions_by_node.items():
            # Sort by version_number ascending
            versions.sort(key=lambda v: getattr(v, "version_number", 0) or 0)

            for i in range(len(versions)):
                v_current = versions[i]

                # VERSION_OF: version → node
                vid = getattr(v_current, "version_id", "")
                key_vof = (vid, node_id, "VERSION_OF")
                if key_vof not in seen:
                    seen.add(key_vof)
                    all_edges.append(
                        GraphEdge(
                            source_id=vid,
                            target_id=node_id,
                            relation_type=RelationType.VERSION_OF,
                            strength=1.0,
                            evidence=f"Version {getattr(v_current, 'version_number', '?')} of node {node_id[:8]}",
                            projections=["version"],
                            metadata={
                                "version_number": getattr(v_current, "version_number", 0),
                                "version_id": vid,
                            },
                        ).to_knowledge_edge()
                    )

                # SUPERSEDES / REPLACED_BY: chain pairs
                if i > 0:
                    v_prev = versions[i - 1]
                    prev_vid = getattr(v_prev, "version_id", "")
                    vnum_current = getattr(v_current, "version_number", 0)
                    vnum_prev = getattr(v_prev, "version_number", 0)

                    # SUPERSEDES: v_current → v_prev (newer supersedes older)
                    key_sup = (vid, prev_vid, "SUPERSEDES")
                    if key_sup not in seen:
                        seen.add(key_sup)
                        all_edges.append(
                            GraphEdge(
                                source_id=vid,
                                target_id=prev_vid,
                                relation_type=RelationType.SUPERSEDES,
                                strength=1.0,
                                evidence=f"Version {vnum_current} supersedes version {vnum_prev}",
                                projections=["version"],
                                metadata={
                                    "newer_version": vnum_current,
                                    "older_version": vnum_prev,
                                    "newer_version_id": vid,
                                    "older_version_id": prev_vid,
                                },
                            ).to_knowledge_edge()
                        )

                    # REPLACED_BY: v_prev → v_current
                    key_rep = (prev_vid, vid, "REPLACED_BY")
                    if key_rep not in seen:
                        seen.add(key_rep)
                        all_edges.append(
                            GraphEdge(
                                source_id=prev_vid,
                                target_id=vid,
                                relation_type=RelationType.REPLACED_BY,
                                strength=1.0,
                                evidence=f"Version {vnum_prev} replaced by version {vnum_current}",
                                projections=["version"],
                                metadata={
                                    "older_version": vnum_prev,
                                    "newer_version": vnum_current,
                                },
                            ).to_knowledge_edge()
                        )

                # SNAPSHOT_OF: if previous_version is set
                prev_version_id = getattr(v_current, "previous_version", None)
                if prev_version_id and prev_version_id != "":
                    key_snap = (vid, prev_version_id, "SNAPSHOT_OF")
                    if key_snap not in seen:
                        seen.add(key_snap)
                        all_edges.append(
                            GraphEdge(
                                source_id=vid,
                                target_id=prev_version_id,
                                relation_type=RelationType.SNAPSHOT_OF,
                                strength=0.9,
                                evidence=f"Version {getattr(v_current, 'version_number', '?')} "
                                f"is snapshot of {prev_version_id[:8]}",
                                projections=["version"],
                                metadata={"previous_version_id": prev_version_id},
                            ).to_knowledge_edge()
                        )

        logger.info(
            "VersionDAG: %d version groups → %d edges from doc %s",
            len(versions_by_node),
            len(all_edges),
            doc_id[:8],
        )
        return all_edges
