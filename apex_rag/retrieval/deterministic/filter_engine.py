import re
from collections.abc import Sequence
from datetime import datetime

from apex_rag.models.unified_models import ASTNode, NodeType


class StructuralFilterEngine:
    """
    Evolved v3 Structural Filter Engine that performs deterministic, multi-factor,
    vector-less filtering of AST nodes to narrow down search spaces.
    Target: 100,000 nodes -> <100 candidate nodes.
    """

    def __init__(self, stop_words: set[str] | None = None) -> None:
        self.stop_words = stop_words or {
            "the", "is", "in", "and", "to", "a", "of", "for", "on", "with", "as", "by", "an", "at", "it", "this"
        }

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b\w{3,}\b", text.lower())
        return [w for w in words if w not in self.stop_words]

    def score_node(
        self,
        query: str,
        node: ASTNode,
        page_index_map: dict[str, list[int]] | None = None,
    ) -> float:
        """
        Compute a hybrid structural/deterministic score for a node.
        """
        score = 0.0
        query_lower = query.lower()
        content_lower = node.content.lower()

        query_tokens = set(self._tokenize(query))
        node_tokens = self._tokenize(node.content)

        if not query_tokens or not node_tokens:
            return 0.0

        # 1. Exact Phrase Match (high weight)
        if query_lower in content_lower:
            score += 50.0

        # 2. Heading Relevance
        if node.node_type == NodeType.HEADING:
            heading_matches = sum(1 for tok in query_tokens if tok in content_lower)
            score += heading_matches * 15.0

        # 3. Entity / Keyword Overlap
        overlap = query_tokens.intersection(set(node_tokens))
        score += len(overlap) * 5.0

        # 4. Keyword Density
        density = len(overlap) / max(len(node_tokens), 1)
        score += density * 10.0

        # 5. Page Index Relevance
        if page_index_map and node.page_number is not None:
            for term, pages in page_index_map.items():
                if term.lower() in query_lower and node.page_number in pages:
                    score += 25.0

        # 6. Section Hierarchy Relevance
        # Nodes that are higher in structural path or refine queried titles get a slight boost
        if hasattr(node, "path") and node.path:
            score += max(0.0, 5.0 - (len(node.path.split(".")) * 0.5))

        # 7. Temporal Relevance
        if node.source_date is not None:
            # Assumed relative to current date (newer is slightly prioritized if equal matching)
            days = (datetime.now(node.source_date.tzinfo or None) - node.source_date).days
            decay = 1.0 / (1.0 + (0.0001 * max(0, days)))
            score *= (0.8 + 0.2 * decay)

        return score

    def filter_candidates(
        self,
        query: str,
        nodes: Sequence[ASTNode],
        page_index_map: dict[str, list[int]] | None = None,
        max_candidates: int = 80,
    ) -> list[ASTNode]:
        """
        Filters and ranks ASTNodes deterministically, returning at most max_candidates.
        """
        scored = []
        for node in nodes:
            score = self.score_node(query, node, page_index_map)
            if score > 0.0:
                scored.append((score, node))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n[1] for n in scored[:max_candidates]]
