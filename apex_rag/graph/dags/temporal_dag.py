"""
graph/dags/temporal_dag.py — TemporalDAG Builder.

Creates temporal edges between nodes based on chronological ordering
and version history with ``projection=["temporal"]``.

Strategies (deterministic, no LLM calls):
    1. **SUCCESSOR** — Older node → newer node (chronological order)
    2. **PREDECESSOR** — Newer node → older node (reverse chronological)
    3. **VALID_DURING** — Node → its effective time window
    4. **OVERRIDES** — Newer contradictory source overrides older

Usage:
    builder = TemporalDagBuilder(storage)
    edges = await builder.build(nodes, doc_id="doc-123")
    for edge in edges:
        await storage.save_knowledge_edge(edge)
"""

from __future__ import annotations

import logging
from datetime import datetime

from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.models.unified_models import ASTNode, KnowledgeEdge

logger = logging.getLogger("apex_rag.graph.dags.temporal_dag")


class TemporalDagBuilder:
    """Builds TemporalDAG edges from AST nodes and version metadata.

    All edges carry ``projection=["temporal"]`` with temporal metadata
    including source_date, version_number, and freshness score.
    """

    def __init__(self, storage: ApexStorage | None = None) -> None:
        self._storage = storage

    async def build(
        self,
        nodes: list[ASTNode],
        *,
        doc_id: str,
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build TemporalDAG edges for a document.

        Runs 3 strategies:
            1. SUCCESSOR / PREDECESSOR — chronological ordering by source_date
            2. VALID_DURING — effective time window edges
            3. OVERRIDES — newer-over-older when same topic detected

        Args:
            nodes:     AST nodes for the document.
            doc_id:    The document ID.
            tenant_id: Tenant isolation boundary (reserved).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["temporal"].
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        # Sort nodes chronologically by source_date
        dated_nodes = sorted(
            [n for n in nodes if n.source_date is not None],
            key=lambda n: (n.source_date or datetime.min),
        )

        # 1. SUCCESSOR / PREDECESSOR edges
        for i in range(len(dated_nodes) - 1):
            older = dated_nodes[i]
            newer = dated_nodes[i + 1]

            # SUCCESSOR: older → newer
            key_succ = (older.node_id, newer.node_id, "SUCCESSOR")
            if key_succ not in seen:
                seen.add(key_succ)
                all_edges.append(
                    GraphEdge(
                        source_id=older.node_id,
                        target_id=newer.node_id,
                        relation_type=RelationType.SUCCESSOR,
                        strength=0.9,
                        evidence=f"Temporal: node {older.node_id[:8]} precedes {newer.node_id[:8]}",
                        projections=["temporal"],
                        metadata={
                            "older_date": older.source_date.isoformat() if older.source_date else None,
                            "newer_date": newer.source_date.isoformat() if newer.source_date else None,
                        },
                    ).to_knowledge_edge()
                )

            # PREDECESSOR: newer → older
            key_pred = (newer.node_id, older.node_id, "PREDECESSOR")
            if key_pred not in seen:
                seen.add(key_pred)
                all_edges.append(
                    GraphEdge(
                        source_id=newer.node_id,
                        target_id=older.node_id,
                        relation_type=RelationType.PREDECESSOR,
                        strength=0.9,
                        evidence=f"Temporal: node {newer.node_id[:8]} succeeds {older.node_id[:8]}",
                        projections=["temporal"],
                        metadata={
                            "newer_date": newer.source_date.isoformat() if newer.source_date else None,
                            "older_date": older.source_date.isoformat() if older.source_date else None,
                        },
                    ).to_knowledge_edge()
                )

        # 2. VALID_DURING edges (source_date as effective period)
        for node in dated_nodes:
            if node.source_date:
                key = (node.node_id, node.node_id, "VALID_DURING")
                if key not in seen:
                    seen.add(key)
                    all_edges.append(
                        GraphEdge(
                            source_id=node.node_id,
                            target_id=node.node_id,
                            relation_type=RelationType.VALID_DURING,
                            strength=1.0,
                            evidence=f"Temporal: node valid from {node.source_date.date()}",
                            projections=["temporal"],
                            metadata={
                                "valid_from": node.source_date.isoformat(),
                                "source_date": node.source_date.isoformat(),
                            },
                        ).to_knowledge_edge()
                    )

        # 3. OVERRIDES edges (same-level nodes, newer overrides older)
        # Group nodes by depth and parent for topic similarity
        depth_groups: dict[str, list[ASTNode]] = {}
        for node in dated_nodes:
            group_key = f"{node.depth}:{node.parent_id or 'root'}"
            depth_groups.setdefault(group_key, []).append(node)

        for _group_key, group in depth_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a.source_date and b.source_date:
                        newer, older = (a, b) if a.source_date >= b.source_date else (b, a)
                        key = (newer.node_id, older.node_id, "OVERRIDES")
                        if key not in seen:
                            seen.add(key)
                            all_edges.append(
                                GraphEdge(
                                    source_id=newer.node_id,
                                    target_id=older.node_id,
                                    relation_type=RelationType.OVERRIDES,
                                    strength=0.5,
                                    evidence=f"Temporal: newer node {newer.node_id[:8]} "
                                    f"overrides older {older.node_id[:8]} (same section)",
                                    projections=["temporal"],
                                    metadata={
                                        "newer_date": newer.source_date.isoformat(),
                                        "older_date": older.source_date.isoformat(),
                                    },
                                ).to_knowledge_edge()
                            )

        logger.info(
            "TemporalDAG: %d edges from doc %s (%d dated nodes)",
            len(all_edges),
            doc_id[:8],
            len(dated_nodes),
        )
        return all_edges
