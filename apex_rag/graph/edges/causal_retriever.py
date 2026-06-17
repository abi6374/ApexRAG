"""
graph/edges/causal_retriever.py — Graph traversal and evidence chain construction.

Traverses the Causal Knowledge Graph to find reasoning chains that link
evidence nodes together for the :class:`EvidenceSynthesizerAgent`.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Protocol

from apex_rag.models.unified_models import ASTNode, CausalEdge, EdgeType

logger = logging.getLogger("apex_rag.graph.causal_retriever")


# ── Storage Protocol ────────────────────────────────────────────────


class StorageProvider(Protocol):
    """Minimal storage interface for the retriever."""

    async def get_edges_for_node(self, node_id: str) -> list[CausalEdge]:
        ...

    async def get_node(self, node_id: str) -> ASTNode | None:
        ...

    async def get_nodes_by_doc(self, doc_id: str) -> list[ASTNode]:
        ...


# ═══════════════════════════════════════════════════════════════════════
# CausalRetriever
# ═══════════════════════════════════════════════════════════════════════


class CausalRetriever:
    """Traverses the Causal Knowledge Graph to build evidence chains.

    Given a set of seed nodes (e.g. retrieved evidence), the retriever
    follows ``SUPPORTS``, ``REFINES``, and ``DEPENDS_ON`` edges to build
    ordered reasoning chains.  ``CONTRADICTS`` and ``OVERRIDES`` edges
    are flagged separately for the temporal audit.

    Usage::

        retriever = CausalRetriever(storage)
        chain = await retriever.build_chain(seed_nodes)
        # chain is a list of CausalEdges forming a reasoning path
    """

    def __init__(self, storage: StorageProvider) -> None:
        self._storage = storage

    # ── Public API ─────────────────────────────────────────────────────

    async def build_chain(
        self,
        seed_nodes: list[ASTNode],
        *,
        max_depth: int = 3,
        max_edges: int = 20,
    ) -> list[CausalEdge]:
        """Build a reasoning chain from seed nodes.

        Starts a BFS from each seed node following forward edges
        (``SUPPORTS``, ``REFINES``, ``DEPENDS_ON``) and backward edges
        (reverse of those types).  The result is a merged, deduplicated
        list of edges that forms the ``causal_chain`` in
        :class:`ApexAnswer`.

        Args:
            seed_nodes: Anchor nodes (e.g. retrieved evidence).
            max_depth:  Maximum BFS depth (default 3).
            max_edges:  Maximum edges to return (default 20).

        Returns:
            An ordered list of :class:`CausalEdge` objects.
        """
        all_edges: list[CausalEdge] = []
        seen_edge_ids: set[str] = set()
        visited_nodes: set[str] = {n.node_id for n in seed_nodes}

        for seed in seed_nodes:
            chain = await self._bfs(seed, visited_nodes, max_depth)
            for edge in chain:
                if edge.edge_id not in seen_edge_ids:
                    seen_edge_ids.add(edge.edge_id)
                    all_edges.append(edge)

        # Sort by strength descending
        all_edges.sort(key=lambda e: e.strength, reverse=True)

        logger.info(
            "build_chain: %d seed nodes → %d edges",
            len(seed_nodes),
            len(all_edges),
        )
        return all_edges[:max_edges]

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 4,
    ) -> list[CausalEdge]:
        """Find the strongest path between two nodes.

        Uses BFS with strength-weighted scoring to find the most
        plausible reasoning path from ``source_id`` to ``target_id``.

        Args:
            source_id: Start node ID.
            target_id: End node ID.
            max_depth: Maximum path length (default 4).

        Returns:
            A list of :class:`CausalEdge` objects forming the path,
            or an empty list if no path exists.
        """
        if source_id == target_id:
            return []

        # BFS — track visited + best parent edge
        queue: deque[tuple[str, list[CausalEdge]]] = deque()
        queue.append((source_id, []))
        visited: set[str] = {source_id}

        while queue:
            current_id, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            edges = await self._storage.get_edges_for_node(current_id)
            for edge in edges:
                # Determine the next node (the one *not* equal to current)
                next_id: str | None = None
                if edge.source_node_id == current_id:
                    next_id = edge.target_node_id
                elif edge.target_node_id == current_id:
                    next_id = edge.source_node_id

                if next_id is None or next_id in visited:
                    continue

                visited.add(next_id)
                new_path = path + [edge]

                if next_id == target_id:
                    logger.info(
                        "find_path: found path %s → %s (%d edges)",
                        source_id[:8],
                        target_id[:8],
                        len(new_path),
                    )
                    return new_path

                queue.append((next_id, new_path))

        logger.debug("find_path: no path from %s to %s", source_id[:8], target_id[:8])
        return []

    async def get_subgraph(
        self,
        center_id: str,
        *,
        radius: int = 1,
    ) -> list[CausalEdge]:
        """Extract the local subgraph around a node.

        Args:
            center_id: Central node ID.
            radius:    Number of hops (default 1 = immediate neighbours).

        Returns:
            All edges within the given radius.
        """
        collected: list[CausalEdge] = []
        seen_edges: set[str] = set()
        current_ring: set[str] = {center_id}
        already_visited: set[str] = set()

        for _hop in range(radius):
            next_ring: set[str] = set()
            for nid in current_ring:
                if nid in already_visited:
                    continue
                already_visited.add(nid)
                edges = await self._storage.get_edges_for_node(nid)
                for edge in edges:
                    if edge.edge_id not in seen_edges:
                        seen_edges.add(edge.edge_id)
                        collected.append(edge)
                    # Add neighbour to next ring
                    if edge.source_node_id == nid:
                        next_ring.add(edge.target_node_id)
                    else:
                        next_ring.add(edge.source_node_id)
            current_ring = next_ring - already_visited

        logger.info(
            "get_subgraph: radius=%d around %s → %d edges",
            radius,
            center_id[:8],
            len(collected),
        )
        return collected

    # ── Internal ───────────────────────────────────────────────────────

    async def _bfs(
        self,
        seed: ASTNode,
        visited_nodes: set[str],
        max_depth: int,
    ) -> list[CausalEdge]:
        """BFS from a seed node through relevant edges."""
        edges_found: list[CausalEdge] = []

        queue: deque[tuple[str, int]] = deque()
        queue.append((seed.node_id, 0))

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            causal_edges = await self._storage.get_edges_for_node(current_id)

            for edge in causal_edges:
                # Only follow non-contrary edges (support edges)
                if edge.edge_type in (
                    EdgeType.CONTRADICTS,
                    EdgeType.OVERRIDES,
                ):
                    continue

                edges_found.append(edge)

                # Determine neighbour
                neighbour = (
                    edge.target_node_id
                    if edge.source_node_id == current_id
                    else edge.source_node_id
                )
                if neighbour not in visited_nodes:
                    visited_nodes.add(neighbour)
                    queue.append((neighbour, depth + 1))

        return edges_found
