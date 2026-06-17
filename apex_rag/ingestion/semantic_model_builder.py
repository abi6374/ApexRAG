"""
semantic_model_builder.py — LLM-based signpost generation for AST navigation.

After parsing, every non‑leaf node needs a concise signpost so that the
:class:`ASTNavigatorAgent` can decide which branch to descend into without
reading the full content.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any, Protocol

from apex_rag.models.unified_models import ASTNode, NodeType

logger = logging.getLogger("apex_rag.semantic")


# ═══════════════════════════════════════════════════════════════
# Signpost Provider Protocol
# ═══════════════════════════════════════════════════════════════


class SignpostProvider(Protocol):
    """Protocol for LLM providers that can generate signpost summaries."""

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        ...


# ═══════════════════════════════════════════════════════════════
# Prompt template
# ═══════════════════════════════════════════════════════════════

_SIGNPOST_PROMPT = """\
You are a document indexing assistant.  Generate a concise, information-dense \
signpost (exactly 2 sentences) for the following section so that a search agent \
can decide whether to navigate into it.

Section heading: {heading}
Section content preview (first 500 chars):
{content_preview}

Signpost (2 sentences, max 60 words total):"""


# ═══════════════════════════════════════════════════════════════
# SemanticModelBuilder
# ═══════════════════════════════════════════════════════════════


class SemanticModelBuilder:
    """Generates 2-sentence signpost summaries for every non‑leaf AST node.

    Non‑leaf nodes are those that have children — they represent branches
    in the document's hierarchy that the navigator must decide about.

    Usage::

        builder = SemanticModelBuilder(llm=my_provider)
        signposts = await builder.build_signposts(nodes)
        # signposts: dict[node_id -> "Two sentence summary."]
    """

    def __init__(
        self,
        llm: SignpostProvider,
        max_concurrent: int = 8,
        max_content_chars: int = 500,
    ) -> None:
        self._llm = llm
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_content_chars = max_content_chars

    # ── Public API ─────────────────────────────────────────────────────────

    async def build_signposts(
        self,
        nodes: list[ASTNode],
        *,
        force: bool = False,
    ) -> dict[str, str]:
        """Generate signposts for all non‑leaf nodes.

        Args:
            nodes: Flat list of :class:`ASTNode` objects.
            force: If ``True``, also generate signposts for leaf nodes.

        Returns:
            A dictionary mapping ``node_id → signpost text``.
        """
        non_leaf_nodes = [
            n for n in nodes
            if n.children and (force or n.node_type in {NodeType.HEADING, NodeType.LIST})
        ]

        if not non_leaf_nodes:
            logger.info("No non‑leaf nodes to signpost.")
            return {}

        tasks = [self._generate_signpost(node) for node in non_leaf_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signposts: dict[str, str] = {}
        for node, result in zip(non_leaf_nodes, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "Signpost generation failed for node %s: %s",
                    node.node_id[:8],
                    result,
                )
                signposts[node.node_id] = node.content[:120]  # fallback
            else:
                signposts[node.node_id] = result  # type: ignore[assignment]

        return signposts

    async def build_signpost(self, node: ASTNode) -> str:
        """Generate a signpost for a single node.

        Args:
            node: The AST node to summarise.

        Returns:
            A 2-sentence signpost string.
        """
        return await self._generate_signpost(node)

    # ── Streaming ──────────────────────────────────────────────────────────

    async def stream_signposts(
        self,
        nodes: list[ASTNode],
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Yield ``(node_id, signpost)`` pairs as they complete.

        Useful for real‑time progress display during ingestion.
        """
        for node in nodes:
            if not node.children:
                continue
            try:
                signpost = await self._generate_signpost(node)
                yield node.node_id, signpost
            except Exception as exc:
                logger.warning("Stream signpost failed for %s: %s", node.node_id[:8], exc)
                yield node.node_id, node.content[:120]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _node_preview(self, node: ASTNode) -> tuple[str, str]:
        """Build a preview string for the signpost prompt."""
        heading = node.content[:200] if node.content else "(no title)"

        # Collect child content for context
        child_texts: list[str] = []
        # To avoid reading actual child content (which we might not have),
        # we rely on the node's own content.
        content_preview = node.content[: self._max_content_chars]
        if not content_preview.strip():
            content_preview = "(no content preview available)"

        return heading, content_preview

    async def _generate_signpost(self, node: ASTNode) -> str:
        """Call the LLM to produce a signpost for one node."""
        heading, content_preview = self._node_preview(node)

        prompt = _SIGNPOST_PROMPT.format(
            heading=heading,
            content_preview=content_preview,
        )

        async with self._semaphore:
            response = await self._llm.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=100,
            )

        signpost = response.strip()
        if not signpost:
            return node.content[:120]
        return signpost
