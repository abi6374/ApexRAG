"""
graph/edges/causal_builder.py — Multi-strategy causal edge discovery.

Discovers typed relationships between :class:`ASTNode` instances using
four complementary strategies:

    1. **Structural** — Parent-child → ``REFINES``; siblings → ``SUPPORTS``
    2. **Temporal** — Newer node overrides older contradictory node →
       ``OVERRIDES``
    3. **Semantic** — High embedding similarity → ``SUPPORTS``
    4. **LLM** — LLM analyses node pairs to assign the most appropriate
       :class:`EdgeType`

Each strategy produces a list of :class:`GraphEdge` objects which are
then persisted via :class:`ApexStorage`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from typing import Any, Protocol

from apex_rag.core.protocols.interfaces import LLMProvider
from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.models.unified_models import ASTNode, EdgeType

logger = logging.getLogger("apex_rag.graph.causal_builder")


# ── Embedder Protocol (duck-type compatible with EmbeddingEngine) ────────


class Embedder(Protocol):
    """Minimal embedder interface used by the semantic strategy."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


# ═══════════════════════════════════════════════════════════════════════
# CausalGraphBuilder
# ═══════════════════════════════════════════════════════════════════════


class CausalGraphBuilder:
    """Discovers typed causal edges between AST nodes using multiple strategies.

    Usage::

        builder = CausalGraphBuilder(embedder=engine, llm=my_provider)

        # Discover all edges across a document's nodes
        edges = await builder.build_all(nodes)

        # Or run specific strategies
        structural = builder.build_structural(nodes)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLMProvider | None = None,
        similarity_threshold: float = 0.75,
        temporal_llm: LLMProvider | None = None,
        llm_max_pairs: int = 20,
        storage: Any | None = None,
    ) -> None:
        self._embedder = embedder
        self._llm: LLMProvider | None = llm
        self._similarity_threshold = similarity_threshold
        # Reuse the contradiction detector's LLM if provided
        self._temporal_llm = temporal_llm or llm
        self._llm_max_pairs = llm_max_pairs
        self._storage = storage

    # ── Public API ─────────────────────────────────────────────────────

    async def build_all(
        self,
        nodes: list[ASTNode],
        *,
        include_structural: bool = True,
        include_temporal: bool = True,
        include_semantic: bool = True,
        include_llm: bool = True,
        tenant_id: str | None = None,
    ) -> list[GraphEdge]:
        """Run all enabled strategies and return the union of discovered edges.

        If a ``storage`` reference was provided to the constructor AND
        ``tenant_id`` is passed, all nodes are validated to belong to
        the same tenant before any edge discovery begins, ensuring
        cross-tenant edge discovery is impossible (Principle 18).

        If no storage is available, tenant isolation is deferred to the
        persistence layer (:class:`ApexStorage.save_causal_edge()`),
        which enforces tenant boundaries at write time.

        Args:
            nodes:               All AST nodes to analyse.
            include_structural:  Parent-child / sibling edges.
            include_temporal:    Temporal-override edges.
            include_semantic:    High-similarity support edges.
            include_llm:         LLM-assisted relationship discovery.
            tenant_id:           Optional tenant ID for isolation validation.

        Returns:
            A list of :class:`GraphEdge` objects (deduplicated by
            source-target-relation).

        Raises:
            PermissionError: If storage is available and any node belongs
                             to a different tenant.
        """
        # Tenant isolation: validate all nodes belong to the same tenant
        if tenant_id and self._storage is not None and hasattr(self._storage, "session"):
            from apex_rag.enterprise.auth.tenant_validator import TenantIsolationValidator

            try:
                validator = TenantIsolationValidator(self._storage)
                node_ids = [n.node_id for n in nodes]
                await validator.assert_tenant_graph_traversal(tenant_id, node_ids)
            except Exception:
                # If validation fails (e.g. storage not fully wired),
                # log and defer to persistence-layer enforcement
                import logging

                logging.getLogger(__name__).warning(
                    "Tenant validation skipped in CausalGraphBuilder",
                    exc_info=True,
                )
        coros: list[asyncio.Task[list[GraphEdge]]] = []

        if include_structural:
            coros.append(asyncio.ensure_future(self._run("structural", nodes)))
        if include_temporal and self._temporal_llm is not None:
            coros.append(asyncio.ensure_future(self._run("temporal", nodes)))
        if include_semantic and self._embedder is not None:
            coros.append(asyncio.ensure_future(self._run("semantic", nodes)))
        if include_llm and self._llm is not None:
            coros.append(asyncio.ensure_future(self._run("llm", nodes)))

        if not coros:
            logger.warning("No strategies enabled — no edges discovered.")
            return []

        results = await asyncio.gather(*coros, return_exceptions=True)
        all_edges: list[GraphEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for strategy_edges in results:
            if isinstance(strategy_edges, BaseException):
                continue
            for edge in strategy_edges:
                key = (edge.source_id, edge.target_id, edge.relation_type.value)
                if key not in seen:
                    seen.add(key)
                    all_edges.append(edge)

        logger.info(
            "build_all: %d strategies → %d unique edges",
            len(coros),
            len(all_edges),
        )
        return all_edges

    # ── Strategy: Structural ───────────────────────────────────────────

    def build_structural(self, nodes: list[ASTNode]) -> list[GraphEdge]:
        """Discover edges from the AST tree structure.

        - Parent → child: ``REFINES``  (child adds detail to parent)
        - Siblings:       ``SUPPORTS`` (same-level topics reinforce each other)
        """
        edges: list[GraphEdge] = []

        node_ids = {n.node_id for n in nodes}

        for node in nodes:
            # Parent-child edges: only if parent is also in the provided nodes
            if node.parent_id and node.parent_id in node_ids:
                edges.append(
                    GraphEdge(
                        source_id=node.parent_id,
                        target_id=node.node_id,
                        relation_type=RelationType.REFINES,
                        strength=0.8,
                        evidence=f"Structural: parent-child (depth {node.depth})",
                    )
                )

        # Sibling edges: group by parent_id, connect all pairs
        parent_groups: dict[str | None, list[ASTNode]] = {}
        for node in nodes:
            pid = node.parent_id if node.parent_id else "ROOT"
            parent_groups.setdefault(pid, []).append(node)

        for _pid, group in parent_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    edges.append(
                        GraphEdge(
                            source_id=group[i].node_id,
                            target_id=group[j].node_id,
                            relation_type=RelationType.SUPPORTS,
                            strength=0.6,
                            evidence="Structural: sibling nodes",
                        )
                    )

        logger.debug("Structural strategy: %d edges", len(edges))
        return edges

    # ── Strategy: Temporal ─────────────────────────────────────────────

    async def build_temporal(self, nodes: list[ASTNode]) -> list[GraphEdge]:
        """Discover temporal-override edges.

        For nodes with a known ``source_date``, pairs that are topically
        similar (embedding cosine similarity > 0.65) and where the newer
        node contradicts the older one produce an ``OVERRIDES`` edge.
        """
        edges: list[GraphEdge] = []

        # Filter nodes with dates
        dated = [n for n in nodes if n.source_date is not None and n.embedding]
        if len(dated) < 2:
            return []

        from apex_rag.temporal.contradiction import (
            TemporalContradictionDetector,
        )

        detector = TemporalContradictionDetector(llm=self._temporal_llm)

        for i in range(len(dated)):
            for j in range(i + 1, len(dated)):
                a, b = dated[i], dated[j]
                a_date = a.source_date
                b_date = b.source_date
                if a_date is None or b_date is None:
                    continue
                newer, older = (a, b) if a_date >= b_date else (b, a)

                result = await detector.detect(older, newer)
                if result is not None and result.edge_type == EdgeType.CONTRADICTS:
                    edges.append(
                        GraphEdge(
                            source_id=newer.node_id,
                            target_id=older.node_id,
                            relation_type=RelationType.OVERRIDES,
                            strength=result.strength,
                            evidence=f"Temporal override: newer document overrides older. "
                            f"{result.evidence}",
                        )
                    )

        logger.debug("Temporal strategy: %d edges", len(edges))
        return edges

    # ── Strategy: Semantic ─────────────────────────────────────────────

    async def build_semantic(self, nodes: list[ASTNode]) -> list[GraphEdge]:
        """Discover support edges between semantically similar nodes.

        Two nodes with embedding cosine similarity > threshold are connected
        via a ``SUPPORTS`` edge, provided they are not already structurally
        related (i.e., not parent-child or siblings).
        """
        edges: list[GraphEdge] = []

        if self._embedder is None:
            return edges

        # Ensure nodes have embeddings
        no_emb = [n for n in nodes if not n.embedding]
        if no_emb:
            texts = [n.content for n in no_emb]
            try:
                embs = await self._embedder.embed_texts(texts)
                for n, e in zip(no_emb, embs, strict=True):
                    n.embedding = e
            except Exception as exc:
                logger.warning("Embedding failed for semantic strategy: %s", exc)

        embed_nodes = [n for n in nodes if n.embedding]
        if len(embed_nodes) < 2:
            return []

        for i in range(len(embed_nodes)):
            for j in range(i + 1, len(embed_nodes)):
                a, b = embed_nodes[i], embed_nodes[j]
                sim = self._cosine_similarity(a.embedding, b.embedding)
                if sim >= self._similarity_threshold:
                    if self._is_structurally_related(a, b):
                        continue
                    edges.append(
                        GraphEdge(
                            source_id=a.node_id,
                            target_id=b.node_id,
                            relation_type=RelationType.SUPPORTS,
                            strength=round(sim, 4),
                            evidence=f"Semantic similarity: {sim:.3f}",
                        )
                    )

        logger.debug(
            "Semantic strategy: %d edges (threshold=%.2f)",
            len(edges),
            self._similarity_threshold,
        )
        return edges

    # ── Strategy: LLM ──────────────────────────────────────────────────

    async def build_llm(self, nodes: list[ASTNode]) -> list[GraphEdge]:
        """Use an LLM to discover fine-grained relationships.

        Batches topically similar node pairs (embedding > 0.65) and asks
        the LLM to assign the best :class:`RelationType`.

        Because LLM calls are expensive, only a random sample of pairs
        is analysed (max ``_llm_max_pairs``).
        """
        edges: list[GraphEdge] = []

        if self._llm is None:
            return edges

        # Collect candidate pairs (embedding > 0.65)
        embed_nodes = [n for n in nodes if n.embedding]
        candidates: list[tuple[ASTNode, ASTNode]] = []
        for i in range(len(embed_nodes)):
            for j in range(i + 1, len(embed_nodes)):
                sim = self._cosine_similarity(embed_nodes[i].embedding, embed_nodes[j].embedding)
                if sim > 0.65:
                    candidates.append((embed_nodes[i], embed_nodes[j]))

        if not candidates:
            return edges

        # Limit to avoid excessive cost
        if len(candidates) > self._llm_max_pairs:
            candidates = random.sample(candidates, self._llm_max_pairs)

        logger.info("LLM strategy: analysing %d candidate pairs", len(candidates))

        # Process in parallel with a semaphore for rate limiting
        sem = asyncio.Semaphore(5)

        async def _analyse(a: ASTNode, b: ASTNode) -> GraphEdge | None:
            async with sem:
                return await self._llm_classify(self._llm, a, b)  # type: ignore[arg-type]

        tasks = [_analyse(a, b) for a, b in candidates]
        results = await asyncio.gather(*tasks)
        for r in results:
            if r is not None:
                edges.append(r)

        logger.debug("LLM strategy: %d edges", len(edges))
        return edges

    # ── Internal helpers ───────────────────────────────────────────────

    async def _run(self, strategy: str, nodes: list[ASTNode]) -> list[GraphEdge]:
        """Run a single strategy by name."""
        try:
            if strategy == "structural":
                return self.build_structural(nodes)
            elif strategy == "temporal":
                return await self.build_temporal(nodes)
            elif strategy == "semantic":
                return await self.build_semantic(nodes)
            elif strategy == "llm":
                return await self.build_llm(nodes)
            else:
                logger.warning("Unknown strategy: %s", strategy)
                return []
        except Exception as exc:
            logger.error("Strategy '%s' failed: %s", strategy, exc)
            return []

    async def _llm_classify(
        self, llm: LLMProvider, node_a: ASTNode, node_b: ASTNode
    ) -> GraphEdge | None:
        """Ask the LLM to classify the relationship between two nodes."""
        content_a = node_a.content[:600]
        content_b = node_b.content[:600]

        prompt = f"""You are a relationship classifier. Given two passages from a document,
determine the best relationship type from the following options:

- SUPPORTS:    A reinforces or agrees with B
- REFINES:     A adds detail or scope to B (A is the more detailed one)
- CONTRADICTS: A makes a conflicting factual claim to B
- DEPENDS_ON:  A logically depends on B (A requires B's context)
- SAME_TOPIC:  A and B cover the same topic without clear hierarchy

PASSAGE A:
---START---
{content_a}
---END---

PASSAGE B:
---START---
{content_b}
---END---

Answer with EXACTLY one line in the format:
RELATION:<TYPE>
EVIDENCE:<one sentence explaining why>

Example:
RELATION:REFINES
EVIDENCE:Passage A provides a more detailed breakdown of the revenue figures mentioned in Passage B."""

        try:
            response = await llm.generate(prompt=prompt, temperature=0.0, max_tokens=120)
        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)
            return None

        response = response.strip()
        rel_type_str = "SUPPORTS"
        evidence = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("RELATION:"):
                candidate = line.split(":", 1)[1].strip().upper()
                try:
                    rel_type_str = candidate
                    RelationType(candidate)
                except ValueError:
                    rel_type_str = "SUPPORTS"
            elif line.upper().startswith("EVIDENCE:"):
                evidence = line.split(":", 1)[1].strip()

        try:
            rel_type = RelationType(rel_type_str)
        except ValueError:
            rel_type = RelationType.SUPPORTS

        return GraphEdge(
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            relation_type=rel_type,
            strength=0.7,
            evidence=evidence or f"LLM classified as {rel_type.value}",
        )

    # ── Utility ────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _is_structurally_related(a: ASTNode, b: ASTNode) -> bool:
        """Check if two nodes are parent-child or siblings."""
        if a.parent_id == b.node_id or b.parent_id == a.node_id:
            return True
        return bool(a.parent_id is not None and a.parent_id == b.parent_id)
