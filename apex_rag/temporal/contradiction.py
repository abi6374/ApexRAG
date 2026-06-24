"""
temporal/contradiction.py — Temporal contradiction detection between AST nodes.

Detects when two ASTNodes make conflicting factual claims using a three-step
process:

    **Step 1** — Topical similarity:
        Cosine similarity of embeddings must be > 0.65 (topically related)
        but the raw content must differ.

    **Step 2** — Negation heuristic:
        One node contains negation phrases (``"not"``, ``"no longer"``,
        ``"was repealed"``, ``"has been revised"``) relative to a shared
        entity found in both nodes.

    **Step 3** — LLM confirmation:
        Prompt the LLM: "Do these two passages make contradictory factual
        claims? Answer YES or NO with one sentence of reasoning."

If all three steps pass, a ``CausalEdge`` of type ``CONTRADICTS`` is returned.
"""

from __future__ import annotations

import logging
import math
import re

from apex_rag.core.protocols.interfaces import LLMProvider  # noqa: TC001
from apex_rag.models.unified_models import ASTNode, CausalEdge, EdgeType

logger = logging.getLogger("apex_rag.temporal.contradiction")


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

_SIMILARITY_THRESHOLD = 0.65

_NEGATION_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bno longer\b", re.IGNORECASE),
    re.compile(r"\bwas repealed\b", re.IGNORECASE),
    re.compile(r"\bhas been revised\b", re.IGNORECASE),
    re.compile(r"\bwas rescinded\b", re.IGNORECASE),
    re.compile(r"\bhas been overturned\b", re.IGNORECASE),
    re.compile(r"\bis no longer valid\b", re.IGNORECASE),
    re.compile(r"\bis hereby\s+(revoked|cancelled|terminated)\b", re.IGNORECASE),
    re.compile(r"\bcontradict(s|ed)?\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\bhowever\b", re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════
# TemporalContradictionDetector
# ═══════════════════════════════════════════════════════════════


class TemporalContradictionDetector:
    """Detects contradictory factual claims between two ASTNodes.

    Uses a three-step pipeline:

        1. **Cosine similarity** — embeddings must be > 0.65
        2. **Negation heuristic** — one node contains negation phrases
           relative to a shared entity
        3. **LLM confirmation** — the LLM confirms the contradiction

    Usage::

        detector = TemporalContradictionDetector(llm=my_provider)
        edge = await detector.detect(node_a, node_b)
        if edge is not None:
            print(f"CONTRADICTS: {edge.evidence}")
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    # ── Public API ─────────────────────────────────────────────────────────

    async def detect(self, node_a: ASTNode, node_b: ASTNode) -> CausalEdge | None:
        """Run the three-step contradiction detection pipeline.

        Args:
            node_a: The first AST node.
            node_b: The second AST node.

        Returns:
            A ``CausalEdge`` with type ``CONTRADICTS`` if a contradiction is
            detected, or ``None`` if the nodes are consistent.
        """
        # Step 1: Topical similarity
        if not self._step_1_similarity(node_a, node_b):
            return None

        # Step 2: Negation heuristic
        if not self._step_2_negation_heuristic(node_a, node_b):
            return None

        # Step 3: LLM confirmation
        if self._llm is not None:
            confirmed = await self._step_3_llm_confirm(node_a, node_b)
            if confirmed is None:
                return None
            return confirmed

        # If no LLM is available, rely on steps 1 and 2
        return CausalEdge(
            source_node_id=node_a.node_id,
            target_node_id=node_b.node_id,
            edge_type=EdgeType.CONTRADICTS,
            strength=self._cosine_similarity(node_a.embedding, node_b.embedding),
            evidence="Negation heuristic detected contradictory language.",
        )

    # ── Step 1: Cosine similarity ──────────────────────────────────────────

    def _step_1_similarity(self, node_a: ASTNode, node_b: ASTNode) -> bool:
        """Check that embeddings are similar enough to be topically related."""
        emb_a = node_a.embedding
        emb_b = node_b.embedding

        if not emb_a or not emb_b:
            # No embeddings available — skip similarity check
            return True

        sim = self._cosine_similarity(emb_a, emb_b)

        if sim <= _SIMILARITY_THRESHOLD:
            logger.debug(
                "Step 1 FAIL: similarity %.3f <= %.2f for %s vs %s",
                sim,
                _SIMILARITY_THRESHOLD,
                node_a.node_id[:8],
                node_b.node_id[:8],
            )
            return False

        # Also ensure they're not identical (same content = not a contradiction)
        if node_a.content.strip().lower() == node_b.content.strip().lower():
            logger.debug("Step 1 FAIL: identical content")
            return False

        logger.debug("Step 1 PASS: similarity = %.3f", sim)
        return True

    # ── Step 2: Negation heuristic ─────────────────────────────────────────

    def _step_2_negation_heuristic(self, node_a: ASTNode, node_b: ASTNode) -> bool:
        """Check that one node contains negation phrases relative to a shared entity.

        A "shared entity" is any word (>= 4 chars, alphabetical) that appears
        in both nodes' content.
        """
        content_a = node_a.content.lower()
        content_b = node_b.content.lower()

        # Find shared entities (common words >= 4 chars)
        words_a = set(re.findall(r"\b([a-z]{4,})\b", content_a))
        words_b = set(re.findall(r"\b([a-z]{4,})\b", content_b))
        shared = words_a & words_b

        if not shared:
            logger.debug("Step 2 FAIL: no shared entities")
            return False

        # Check if either node contains negation phrases
        a_has_negation = any(p.search(content_a) for p in _NEGATION_PHRASES)
        b_has_negation = any(p.search(content_b) for p in _NEGATION_PHRASES)

        if not a_has_negation and not b_has_negation:
            logger.debug("Step 2 FAIL: no negation phrases in either node")
            return False

        logger.debug(
            "Step 2 PASS: shared entities=%d, negations: a=%s b=%s",
            len(shared),
            a_has_negation,
            b_has_negation,
        )
        return True

    # ── Step 3: LLM confirmation ───────────────────────────────────────────

    async def _step_3_llm_confirm(
        self,
        node_a: ASTNode,
        node_b: ASTNode,
    ) -> CausalEdge | None:
        """Use the LLM to confirm whether the two nodes contradict each other."""
        if self._llm is None:
            return None

        content_a = node_a.content[:1000]
        content_b = node_b.content[:1000]

        prompt = f"""You are a contradiction detection assistant.  Determine whether the
following two passages make contradictory factual claims.

Answer with EXACTLY one line in this format:
YES|<reason in one sentence>
NO|<reason in one sentence>

Passage A:
---START---
{content_a}
---END---

Passage B:
---START---
{content_b}
---END---"""

        try:
            response = await self._llm.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=80,
            )
        except Exception as exc:
            logger.warning("LLM contradiction check failed: %s", exc)
            return None

        response = response.strip()

        if response.upper().startswith("YES"):
            # Extract reasoning
            evidence = response.split("|", 1)[1].strip() if "|" in response else response
            sim = self._cosine_similarity(node_a.embedding, node_b.embedding)
            logger.info("Step 3 PASS: %s", evidence)
            return CausalEdge(
                source_node_id=node_a.node_id,
                target_node_id=node_b.node_id,
                edge_type=EdgeType.CONTRADICTS,
                strength=sim,
                evidence=evidence,
            )

        logger.debug("Step 3 FAIL: LLM says NO")
        return None

    # ── Batch detection: detect_all ─────────────────────────────────────────

    async def detect_all(self, nodes: list[ASTNode]) -> list[CausalEdge]:
        """Run contradiction detection across all node pairs within topic clusters.

        Clusters nodes by embedding cosine similarity > 0.65, then runs
        ``detect()`` on every pair within each cluster.  This keeps the
        number of comparisons at O(k\u00b2) where k is the cluster size,
        instead of O(n\u00b2) where n is the total number of nodes.

        Args:
            nodes: All ASTNodes to check for contradictions.

        Returns:
            A list of ``CausalEdge`` objects (type ``CONTRADICTS``) for
            every confirmed contradiction found.
        """
        if len(nodes) < 2:
            return []

        # 1. Cluster by embedding similarity > 0.65
        clusters = self._cluster_by_similarity(nodes)

        logger.info(
            "Cluster stats: %d nodes → %d cluster(s) (avg %.1f nodes/cluster)",
            len(nodes),
            len(clusters),
            sum(len(c) for c in clusters) / max(len(clusters), 1),
        )

        # 2. Run detect() on every pair within each cluster
        edges: list[CausalEdge] = []
        for cluster in clusters:
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    edge = await self.detect(cluster[i], cluster[j])
                    if edge is not None:
                        edges.append(edge)

        return edges

    # ── Clustering helper ────────────────────────────────────────────────────

    def _cluster_by_similarity(self, nodes: list[ASTNode]) -> list[list[ASTNode]]:
        """Greedy single-linkage clustering by embedding similarity > 0.65."""
        if not nodes:
            return []

        # Pre-filter: only nodes with embeddings can be clustered
        embed_nodes = [n for n in nodes if n.embedding]
        if not embed_nodes:
            # No embeddings available — assign each node to its own cluster
            return [[n] for n in nodes]

        assigned = [False] * len(embed_nodes)
        clusters: list[list[ASTNode]] = []

        for i in range(len(embed_nodes)):
            if assigned[i]:
                continue

            # Start a new cluster with this node
            cluster: list[ASTNode] = [embed_nodes[i]]
            assigned[i] = True

            # Greedy expansion: add any unassigned node that is similar
            # to ANY node already in the cluster
            changed = True
            while changed:
                changed = False
                for j in range(len(embed_nodes)):
                    if assigned[j]:
                        continue
                    # Check similarity to any node already in the cluster
                    for cn in cluster:
                        sim = self._cosine_similarity(cn.embedding, embed_nodes[j].embedding)
                        if sim > _SIMILARITY_THRESHOLD:
                            cluster.append(embed_nodes[j])
                            assigned[j] = True
                            changed = True
                            break

            clusters.append(cluster)

        # Add nodes without embeddings (each as its own cluster)
        no_embed_nodes = [n for n in nodes if not n.embedding]
        for n in no_embed_nodes:
            clusters.append([n])

        return clusters

    # ── Utility ────────────────────────────────────────────────────────────

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
