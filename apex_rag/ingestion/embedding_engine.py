"""
embedding_engine.py — Async batched embedding generation for AST nodes.

Uses the provider's ``embed()`` method (or falls back to a simple
hash‑based fingerprint when no embedding model is available).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Sequence
from typing import Protocol

from apex_rag.models.unified_models import ASTNode

logger = logging.getLogger("apex_rag.embedding")


# ═══════════════════════════════════════════════════════════════
# Embedder Protocol
# ═══════════════════════════════════════════════════════════════


class Embedder(Protocol):
    """Protocol for providers that can generate text embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


# ═══════════════════════════════════════════════════════════════
# EmbeddingEngine
# ═══════════════════════════════════════════════════════════════


class EmbeddingEngine:
    """Async batched embedding generator for AST nodes.

    Embeds the ``content`` of each node in batches, reducing API calls
    for large document trees.

    If no embedder is provided, a deterministic fingerprint (SHA‑256 hash
    of content seeded as floats) is used instead — useful for testing
    and offline environments.

    Usage::

        engine = EmbeddingEngine(embedder=my_provider, batch_size=16)
        await engine.embed_nodes(nodes)
        # nodes now have their ``embedding`` field populated
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        batch_size: int = 16,
        dimension: int = 384,
    ) -> None:
        self._embedder = embedder
        self._batch_size = batch_size
        self._dimension = dimension

    # ── Public API ─────────────────────────────────────────────────────────

    async def embed_nodes(self, nodes: Sequence[ASTNode]) -> list[ASTNode]:
        """Embed all nodes in batches, mutating each node's ``embedding`` field.

        Args:
            nodes: The list of AST nodes to embed.

        Returns:
            The same list of nodes with their ``embedding`` field populated.
        """
        if not nodes:
            return list(nodes)

        # Collect node content for embedding
        texts = [node.content for node in nodes]

        if self._embedder is not None:
            embeddings = await self._batch_embed(texts)
        else:
            embeddings = [self._fingerprint(t) for t in texts]

        for node, emb in zip(nodes, embeddings, strict=True):
            node.embedding = emb

        return list(nodes)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed arbitrary texts (not node‑bound).

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors (one per text).
        """
        if self._embedder is not None:
            return await self._batch_embed(texts)
        return [self._fingerprint(t) for t in texts]

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, returning a flat list of vectors."""
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]

        # Run batches concurrently (but the embedder may have its own rate limit)
        coros = [self._embed_single_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*coros)

        # Flatten
        result: list[list[float]] = []
        for batch in batch_results:
            result.extend(batch)
        return result

    async def _embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch of texts using the provider."""
        assert self._embedder is not None
        try:
            return await self._embedder.embed(texts)
        except Exception as exc:
            logger.warning("Batch embedding failed (%d texts): %s", len(texts), exc)
            # Fall back to fingerprints for the failed batch
            return [self._fingerprint(t) for t in texts]

    def _fingerprint(self, text: str) -> list[float]:
        """Deterministic hash‑based embedding fingerprint.

        Produces a ``dimension``-length vector by seeding Python's hash
        with consecutive SHA‑256 digest bytes.  This is **not** semantically
        meaningful — it just gives a unique, repeatable vector per text.
        """
        result: list[float] = []
        counter = 0
        while len(result) < self._dimension:
            extended = hashlib.sha256(f"{text}:{counter}".encode()).hexdigest()
            for i in range(0, len(extended), 8):
                chunk = extended[i : i + 8]
                # Map hex to float in [-1, 1]
                val = int(chunk, 16) / (16**8) * 2.0 - 1.0
                result.append(val)
            counter += 1
        return result[: self._dimension]
