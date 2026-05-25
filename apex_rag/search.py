"""
search.py — Hybrid Search Engine for ApexRAG.

Combines THREE retrieval strategies for maximum accuracy:
  1. **Agentic Navigation** — LLM-guided structural tree walking (primary)
  2. **Vector Similarity** — Semantic embeddings via sentence-transformers (optional)
  3. **Keyword BM25** — Full-text search via SQLite FTS5 (always-on)

This is the secret sauce that makes ApexRAG unique: it's the only RAG library
that unifies structural, semantic, AND keyword search in a single agentic pipeline.

Usage::

    from apex_rag.search import HybridSearch

    searcher = HybridSearch(storage)
    results = await searcher.hybrid_search("What is Q3 revenue?", doc_id)
"""

from __future__ import annotations

import logging
from typing import Any

from apex_rag.storage import DocumentNode, StorageEngine

# ---------------------------------------------------------------------------
# Embeddings Engine (Optional — uses sentence-transformers)
# ---------------------------------------------------------------------------


class EmbeddingsEngine:
    """
    Lightweight wrapper around sentence-transformers for local embeddings.

    This is **completely optional** — if sentence-transformers is not installed,
    the library degrades gracefully to FTS5 + agentic search only.

    Usage::

        engine = EmbeddingsEngine()
        vectors = await engine.encode(["text1", "text2"])
        similarity = await engine.similarity(query_vec, doc_vecs)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._available = False
        self._logger = logging.getLogger("apex_rag.search")

    async def ensure_loaded(self) -> bool:
        """Lazy-load the embedding model. Returns True if available."""
        if self._model is not None:
            return True
        try:
            # Lazy import — sentence-transformers is an optional dependency
            # Run model loading in a thread to avoid blocking
            import asyncio

            from sentence_transformers import SentenceTransformer

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self._model_name, device=self._device),
            )
            self._available = True
            self._logger.info(
                "EmbeddingsEngine ready: model=%s device=%s dim=%d",
                self._model_name,
                self._device,
                self._model.get_sentence_embedding_dimension(),
            )
            return True
        except ImportError:
            self._available = False
            self._logger.warning(
                "sentence-transformers not installed. "
                "Install: pip install apex-rag[vectors]"
            )
            return False
        except Exception as exc:
            self._available = False
            self._logger.warning(
                "Failed to load embeddings model: %s. "
                "Falling back to FTS5 + agentic search.",
                exc,
            )
            return False

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into embedding vectors.

        Returns:
            List of float vectors, or empty list if embeddings unavailable.
        """
        loaded = await self.ensure_loaded()
        if not loaded or self._model is None:
            return []

        import asyncio
        loop = asyncio.get_event_loop()
        embeddings: list[list[float]] = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, show_progress_bar=False).tolist(),
        )
        return embeddings

    async def similarity(
        self,
        query_vec: list[float],
        doc_vecs: list[list[float]],
    ) -> list[float]:
        """
        Compute cosine similarity between a query vector and document vectors.

        Args:
            query_vec:  The query embedding (from encode()).
            doc_vecs:   List of document embeddings.

        Returns:
            List of similarity scores (0–1), same order as doc_vecs.
        """
        if not query_vec or not doc_vecs:
            return []

        import numpy as np

        q = np.array(query_vec, dtype=np.float32)
        d = np.array(doc_vecs, dtype=np.float32)

        # Cosine similarity
        q_norm = q / np.linalg.norm(q)
        d_norm = d / np.linalg.norm(d, axis=1, keepdims=True)
        scores: list[float] = np.dot(d_norm, q_norm).tolist()
        return scores

    @property
    def is_available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# Hybrid Search — combines FTS5 + Vector + Agentic
# ---------------------------------------------------------------------------


class HybridSearch:
    """
    Three-tier hybrid search engine.

    Tiers (tried in order):
      1. **Semantic Cache** — Instant replay of previous exact/substring matches
      2. **Vector Search** — Cosine similarity on sentence embeddings (if available)
      3. **FTS5 Keyword** — SQLite full-text search (always available)
      4. **Agentic Navigation** — LLM-guided structural tree walking (final)

    Args:
        storage:       StorageEngine instance.
        embeddings:    Optional EmbeddingsEngine for vector search.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingsEngine | None = None,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings

    # -- Vector Search ----------------------------------------------------

    async def vector_search(
        self,
        query: str,
        doc_id: str,
        *,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[tuple[DocumentNode, float]]:
        """
        Search document nodes by vector similarity.

        Args:
            query:     Natural-language query.
            doc_id:    Target document.
            top_k:     Maximum results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of (DocumentNode, score) tuples, sorted by score descending.
        """
        if not self._embeddings or not self._embeddings.is_available:
            return []

        async with self._storage.session() as session:
            nodes = await self._storage.get_full_tree(session, doc_id)

        if not nodes:
            return []

        # Encode query and all leaf nodes (leaf nodes have content)
        leaf_nodes = [n for n in nodes if n.is_leaf and n.content]
        if not leaf_nodes:
            return []

        # Batch encode all contents
        contents = [n.content or "" for n in leaf_nodes]
        all_embeddings = await self._embeddings.encode(contents)

        if not all_embeddings:
            return []

        query_vec = (await self._embeddings.encode([query]))[0]
        scores = await self._embeddings.similarity(query_vec, all_embeddings)

        # Filter and sort
        results = []
        for node, score in zip(leaf_nodes, scores, strict=True):
            if score >= min_score:
                results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # -- Combined Search --------------------------------------------------

    async def hybrid_rank(
        self,
        query: str,
        doc_id: str,
        *,
        vector_weight: float = 0.4,
        keyword_weight: float = 0.3,
        structural_weight: float = 0.3,
    ) -> list[tuple[DocumentNode, float]]:
        """
        Rank all leaf nodes by a weighted combination of vector, keyword,
        and structural scores.

        This provides a unified ranking that can be used either:
        - As a standalone search (fast but less accurate than agentic)
        - To pre-filter candidates for the agent (reducing LLM calls)

        Args:
            query:              Natural-language query.
            doc_id:             Target document.
            vector_weight:      Weight for vector similarity score (0–1).
            keyword_weight:     Weight for keyword/BM25 score (0–1).
            structural_weight:  Weight for structural position score (0–1).

        Returns:
            List of (DocumentNode, combined_score) sorted by score descending.
        """
        async with self._storage.session() as session:
            nodes = await self._storage.get_full_tree(session, doc_id)

        if not nodes:
            return []

        leaf_nodes = [n for n in nodes if n.is_leaf and n.content]
        if not leaf_nodes:
            return []

        # 1. Vector scores
        vector_scores: dict[int, float] = {}
        if self._embeddings and self._embeddings.is_available and vector_weight > 0:
            contents = [n.content or "" for n in leaf_nodes]
            all_embeddings = await self._embeddings.encode(contents)
            if all_embeddings:
                query_vec = (await self._embeddings.encode([query]))[0]
                sim_scores = await self._embeddings.similarity(query_vec, all_embeddings)
                for node, score in zip(leaf_nodes, sim_scores, strict=True):
                    vector_scores[node.id] = score

        # 2. Keyword / BM25 scores
        keyword_scores: dict[int, float] = {}
        if keyword_weight > 0:
            query_terms = query.lower().split()
            for node in leaf_nodes:
                score = 0.0
                title_lower = node.title.lower()
                content_lower = (node.content or "").lower()
                for term in query_terms:
                    if term in title_lower:
                        score += 3.0  # Title match is highly relevant
                    count = content_lower.count(term)
                    score += count * 0.5  # Frequency in content
                if score > 0:
                    keyword_scores[node.id] = score

        # 3. Structural scores (position in tree — earlier nodes get bonus)
        structural_scores: dict[int, float] = {}
        if structural_weight > 0:
            total = len(leaf_nodes)
            for i, node in enumerate(leaf_nodes):
                # Earlier nodes (higher in the doc) get higher scores
                structural_scores[node.id] = 1.0 - (i / max(total, 1))

        # Normalize each score set to [0, 1]
        def _normalize(scores: dict[int, float]) -> dict[int, float]:
            if not scores:
                return scores
            max_val = max(scores.values())
            if max_val == 0:
                return scores
            return {k: v / max_val for k, v in scores.items()}

        vector_scores = _normalize(vector_scores)
        keyword_scores = _normalize(keyword_scores)
        structural_scores = _normalize(structural_scores)

        # Combine
        all_ids = {n.id for n in leaf_nodes}
        combined: dict[int, float] = {}
        for nid in all_ids:
            combined[nid] = (
                vector_scores.get(nid, 0) * vector_weight
                + keyword_scores.get(nid, 0) * keyword_weight
                + structural_scores.get(nid, 0) * structural_weight
            )

        ranked = sorted(
            [(n, combined[n.id]) for n in leaf_nodes if combined.get(n.id, 0) > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    # -- Search helper for global query -----------------------------------

    async def vector_search_global(
        self,
        query: str,
        *,
        top_k_docs: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find the most relevant documents for a query using vector search.

        Uses the root-level summaries of each document for comparison.

        Args:
            query:        Natural-language query.
            top_k_docs:   Maximum documents to return.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if not self._embeddings or not self._embeddings.is_available:
            return []

        async with self._storage.session() as session:
            doc_ids = await self._storage.list_documents(session)
            if not doc_ids:
                return []

            # Get root node summaries
            docs_summaries = []
            for did in doc_ids:
                roots = await self._storage.get_children(
                    session, parent_id=None, doc_id=did
                )
                if roots:
                    summary = roots[0].summary or roots[0].title
                    docs_summaries.append((did, summary))
                else:
                    docs_summaries.append((did, ""))

            if not docs_summaries:
                return []

        # Encode query and document summaries
        query_vec = (await self._embeddings.encode([query]))[0]
        doc_texts = [s for _, s in docs_summaries]
        doc_embeddings = await self._embeddings.encode(doc_texts)

        scores = await self._embeddings.similarity(query_vec, doc_embeddings)

        results = [
            (did, score)
            for (did, _), score in zip(docs_summaries, scores, strict=True)
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k_docs]
