"""
graph/dags/document_dag.py — DocumentDAG Builder.

Creates structural edges from the AST tree hierarchy with
``projection=["document"]``.

Strategies (deterministic, no LLM calls):
    1. **REFINES** — Parent → child (child adds detail to parent)
    2. **SUPPORTS** — Sibling → sibling (same-level topics reinforce)

This wraps the structural strategy from ``CausalGraphBuilder.build_structural()``
but tags all edges with the ``document`` projection.

Usage:
    builder = DocumentDagBuilder()
    edges = await builder.build(nodes, doc_id="doc-123")
    for edge in edges:
        await storage.save_knowledge_edge(edge)
"""

from __future__ import annotations

import logging
from typing import Any

from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.models.unified_models import ASTNode, KnowledgeEdge

logger = logging.getLogger("apex_rag.graph.dags.document_dag")


class DocumentDagBuilder:
    """Builds DocumentDAG edges from the AST tree structure.

    All produced edges carry ``projection=["document"]`` and are
    tagged with structural metadata (depth, relationship).
    """

    def __init__(self, storage: Any | None = None) -> None:
        self._storage = storage

    async def build(
        self,
        nodes: list[ASTNode],
        *,
        doc_id: str,
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build DocumentDAG edges from AST nodes.

        Args:
            nodes:     AST nodes for the document.
            doc_id:    The document ID.
            tenant_id: Tenant isolation boundary (reserved).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["document"].
        """
        all_edges: list[KnowledgeEdge] = []
        node_ids = {n.node_id for n in nodes}
        seen: set[tuple[str, str, str]] = set()

        # 1. REFINES edges (parent → child)
        for node in nodes:
            if node.parent_id and node.parent_id in node_ids:
                key = (node.parent_id, node.node_id, "REFINES")
                if key not in seen:
                    seen.add(key)
                    all_edges.append(
                        GraphEdge(
                            source_id=node.parent_id,
                            target_id=node.node_id,
                            relation_type=RelationType.REFINES,
                            strength=0.8,
                            evidence=f"Document structure: parent-child (depth {node.depth})",
                            projections=["document"],
                            metadata={"depth": node.depth, "relationship": "parent_child"},
                        ).to_knowledge_edge()
                    )

        # 2. SUPPORTS edges (siblings under same parent)
        parent_groups: dict[str | None, list[ASTNode]] = {}
        for node in nodes:
            pid = node.parent_id if node.parent_id else "ROOT"
            parent_groups.setdefault(pid, []).append(node)

        for _pid, group in parent_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    key = (a.node_id, b.node_id, "SUPPORTS")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=a.node_id,
                                target_id=b.node_id,
                                relation_type=RelationType.SUPPORTS,
                                strength=0.6,
                                evidence=f"Document structure: siblings at depth {a.depth}",
                                projections=["document"],
                                metadata={
                                    "depth": a.depth,
                                    "relationship": "sibling",
                                },
                            ).to_knowledge_edge()
                        )

        logger.info(
            "DocumentDAG: %d edges from doc %s",
            len(all_edges),
            doc_id[:8],
        )
        return all_edges
