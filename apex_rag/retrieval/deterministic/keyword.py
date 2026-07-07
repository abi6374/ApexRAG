import re
from collections import Counter

from apex_rag.core.protocols.interfaces import DeterministicRetriever
from apex_rag.models.unified_models import ASTNode, NodeType


class KeywordDeterministicRetriever(DeterministicRetriever):
    """
    A basic implementation of Deterministic pre-filtering using keyword frequency
    and heading overlap.
    """

    def __init__(self) -> None:
        # Basic stop words to ignore
        self.stop_words = {
            "the",
            "is",
            "in",
            "and",
            "to",
            "a",
            "of",
            "for",
            "on",
            "with",
            "as",
            "by",
        }

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.stop_words]

    async def retrieve(self, query: str, root_node: ASTNode, top_k: int = 5) -> list[ASTNode]:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        # Flatten the AST to score sections and paragraphs
        # Exclude the root node itself from candidates to prevent navigation loops
        all_nodes = self._flatten_ast(root_node)
        all_nodes = [n for n in all_nodes if n.node_id != root_node.node_id]

        scored_nodes = []
        for node in all_nodes:
            score = self._score_node(query_tokens, node)
            if score > 0:
                scored_nodes.append((score, node))

        # Sort by score descending
        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        # Return top_k nodes
        return [n[1] for n in scored_nodes[:top_k]]

    def _score_node(self, query_tokens: set[str], node: ASTNode) -> float:
        node_tokens = self._tokenize(node.content)
        if not node_tokens:
            return 0.0

        token_counts = Counter(node_tokens)
        score = 0.0

        for q_token in query_tokens:
            if q_token in token_counts:
                # Heading matches get a massive boost (structural scoring)
                if node.node_type == NodeType.HEADING:
                    score += token_counts[q_token] * 5.0
                else:
                    score += token_counts[q_token] * 1.0

        # Normalize by node length to prevent long nodes from unfairly winning
        return score / max(len(node_tokens), 1)

    def _flatten_ast(self, node: ASTNode) -> list[ASTNode]:
        nodes = []
        # We only want to rank substantive nodes, not the root Document container usually
        if node.node_type in (NodeType.HEADING, NodeType.PARAGRAPH, NodeType.TABLE, NodeType.LIST):
            nodes.append(node)

        for child in node.children:
            if isinstance(child, ASTNode):
                nodes.extend(self._flatten_ast(child))
            # String children (node IDs) are skipped — they are not in-memory nodes
            # and require a separate DB fetch for full resolution
        return nodes

    def filter_candidates(
        self,
        query: str,
        nodes: list[ASTNode],
        page_index_map: dict[str, list[int]] | None = None,
        max_candidates: int = 80,
    ) -> list[ASTNode]:
        """Evolved v3 filter method leveraging StructuralFilterEngine."""
        from apex_rag.retrieval.deterministic.filter_engine import StructuralFilterEngine

        engine = StructuralFilterEngine(stop_words=self.stop_words)
        return engine.filter_candidates(
            query=query,
            nodes=nodes,
            page_index_map=page_index_map,
            max_candidates=max_candidates,
        )
