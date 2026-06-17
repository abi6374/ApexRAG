import uuid
from collections import deque
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, Field
from apex_rag.models.unified_models import ASTNode, CausalEdge, EdgeType


class ReasoningChain(BaseModel):
    """
    A unified chain of causal and semantic connections showing a path of reasoning.
    """

    chain_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    edges: list[CausalEdge] = Field(default_factory=list)
    nodes: list[ASTNode] = Field(default_factory=list)
    score: float = 0.0
    contradictions: list[CausalEdge] = Field(default_factory=list)


class StorageInterface(Protocol):
    """Protocol for DB querying during graph reasoning."""

    async def get_edges_for_node(self, node_id: str) -> list[CausalEdge]:
        ...

    async def get_node(self, node_id: str) -> ASTNode | None:
        ...


class GraphReasoningEngine:
    """
    Evolved v3 Graph Reasoning Engine to perform BFS traversals, dependency tracing,
    contradiction discovery, and path scoring on the SRG.
    """

    def __init__(self, storage: StorageInterface) -> None:
        self.storage = storage

    async def build_reasoning_chain(
        self,
        seed_nodes: list[ASTNode],
        max_depth: int = 3,
        max_edges: int = 30,
    ) -> ReasoningChain:
        """
        Builds a comprehensive ReasoningChain by performing a BFS from seed nodes
        over non-obsolete edges, scoring paths, and capturing contradiction events.
        """
        chain_edges: list[CausalEdge] = []
        contradictions: list[CausalEdge] = []
        visited_nodes: set[str] = {n.node_id for n in seed_nodes}
        node_map: dict[str, ASTNode] = {n.node_id: n for n in seed_nodes}

        # BFS Queue holds (node_id, current_depth, path_score)
        queue: deque[tuple[str, int, float]] = deque()
        for seed in seed_nodes:
            queue.append((seed.node_id, 0, 1.0))

        seen_edge_ids: set[str] = set()

        while queue:
            node_id, depth, path_score = queue.popleft()
            if depth >= max_depth:
                continue

            edges = await self.storage.get_edges_for_node(node_id)
            for edge in edges:
                if edge.edge_id in seen_edge_ids:
                    continue

                # Identify target neighbor
                neighbor_id = edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
                
                # Check for contradictions
                if edge.edge_type in (EdgeType.CONTRADICTS, EdgeType.OVERRIDES):
                    contradictions.append(edge)
                    seen_edge_ids.add(edge.edge_id)
                    continue

                # Add to normal reasoning paths
                seen_edge_ids.add(edge.edge_id)
                chain_edges.append(edge)

                # Fetch neighbor details
                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    neighbor_node = await self.storage.get_node(neighbor_id)
                    if neighbor_node:
                        node_map[neighbor_id] = neighbor_node
                        # Update path score based on relationship strength
                        new_score = path_score * edge.strength
                        queue.append((neighbor_id, depth + 1, new_score))

        # Sort normal edges by strength/score
        chain_edges.sort(key=lambda e: e.strength, reverse=True)
        chain_edges = chain_edges[:max_edges]

        # Calculate final chain score as average of edge strengths (fallback to 1.0 if empty)
        final_score = (
            sum(e.strength for e in chain_edges) / len(chain_edges)
            if chain_edges
            else 1.0
        )

        return ReasoningChain(
            edges=chain_edges,
            nodes=list(node_map.values()),
            score=round(final_score, 4),
            contradictions=contradictions,
        )

    async def trace_dependencies(self, node_id: str, max_depth: int = 4) -> list[str]:
        """
        Traces nodes that the current node DEPENDS_ON, IMPORTS, or REFERENCES.
        """
        dependencies = []
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        visited = {node_id}

        while queue:
            curr_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            edges = await self.storage.get_edges_for_node(curr_id)
            for edge in edges:
                # We follow DEPENDS_ON and IMPORTS strictly forward from source to target
                if edge.source_node_id == curr_id and edge.edge_type in (EdgeType.DEPENDS_ON, EdgeType.IMPORTS, EdgeType.REFERENCES):
                    target = edge.target_node_id
                    if target not in visited:
                        visited.add(target)
                        dependencies.append(target)
                        queue.append((target, depth + 1))
        return dependencies
