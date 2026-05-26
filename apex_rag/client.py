"""
client.py — Primary user-facing API for ApexRAG.

`ApexIndex` is the single entry point for all ApexRAG operations.
It is designed to be:
  - Thread-safe via asyncio locks (compatible with FastAPI, Starlette, etc.)
  - Context-manager friendly for clean resource management
  - Rich error hierarchy via `apex_rag.exceptions`
  - Hybrid search with optional vector embeddings
  - pip-installable as a library with zero required config

Typical library usage::

    pip install apex-rag

    from apex_rag import ApexIndex

    async with await ApexIndex.create() as index:
        doc_id = await index.ingest("report.pdf")
        result = await index.query("What is the revenue?", doc_id)
        print(result.content)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from types import TracebackType
from typing import Any

from typing_extensions import Self

from apex_rag.exceptions import (
    DocumentNotFoundError,
)
from apex_rag.ingestion.legacy import IngestionEngine, Summariser
from apex_rag.navigation import AggregatorAgent, NavigationAgent, NavigationResult
from apex_rag.providers import AsyncLLM, OllamaProvider
from apex_rag.search import EmbeddingsEngine, HybridSearch
from apex_rag.storage import StorageEngine
from apex_rag.utils import ReasoningTrace, logger


class ApexIndex:
    """
    Production-ready, thread-safe facade for the ApexRAG library.

    All mutating operations are protected by an asyncio.Lock, ensuring safe
    concurrent usage from multiple FastAPI request handlers.

    Quick start::

        from apex_rag import ApexIndex

        async with await ApexIndex.create() as index:
            doc_id = await index.ingest("annual_report.pdf")
            result = await index.query("What was Q3 revenue?", doc_id)
            tree   = await index.get_tree(doc_id)
            idx    = await index.get_page_index(doc_id)

    Args:
        storage:        StorageEngine instance.
        ingestor:       IngestionEngine instance.
        agent:          NavigationAgent instance.
        trace_enabled:  Whether to print the colored reasoning trace.
    """

    def __init__(
        self,
        storage: StorageEngine,
        ingestor: IngestionEngine,
        agent: NavigationAgent,
        aggregator: AggregatorAgent,
        *,
        trace_enabled: bool = True,
        embeddings: EmbeddingsEngine | None = None,
    ) -> None:
        self._storage = storage
        self._ingestor = ingestor
        self._agent = agent
        self._aggregator = aggregator
        self._search = HybridSearch(storage, embeddings=embeddings)
        self._embeddings = embeddings
        self._trace_enabled = trace_enabled
        self._lock = asyncio.Lock()

    # -- Factory ------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        *,
        db_url: str = "sqlite+aiosqlite:///apex.db",
        model: str | AsyncLLM = "llama3.1",
        summariser_model: str | AsyncLLM | None = None,
        verifier_model: str | AsyncLLM | None = None,
        aggregator_model: str | AsyncLLM | None = None,
        ollama_host: str = "http://localhost:11434",
        max_concurrent_summaries: int = 4,
        parser_backend: str = "markitdown",
        trace_enabled: bool = True,
        verify_leaves: bool = True,
        db_echo: bool = False,
    ) -> ApexIndex:
        """
        Async factory — initialises all sub-components and ensures DB schema.

        Args:
            db_url:                    SQLAlchemy async URL.
                                       SQLite:   ``sqlite+aiosqlite:///./apex.db``
                                       Postgres: ``postgresql+asyncpg://user:pass@host/db``
            ollama_host:               Local Ollama server URL.
            model:                     Ollama model for navigation decisions.
            summariser_model:          Model for ingestion summaries (defaults to `model`).
            verifier_model:            Model for leaf verification (defaults to `model`).
                                       Use a smaller/faster model here (e.g. phi3).
            aggregator_model:          Model for multi-document synthesis.
            max_concurrent_summaries:  Ingestion parallelism (tune to GPU VRAM).
            parser_backend:            "markitdown" | "docling" | "plaintext".
            trace_enabled:             Print live color-coded navigation trace.
            verify_leaves:             Verify every candidate leaf with an LLM call.
                                       Disable to trade accuracy for speed.
            db_echo:                   Log SQL queries (dev only).
        """
        storage = await StorageEngine.create(db_url, echo=db_echo)

        def _resolve_llm(m: str | AsyncLLM | None, fallback: AsyncLLM | None = None) -> AsyncLLM:
            if isinstance(m, AsyncLLM):
                return m
            if isinstance(m, str):
                return OllamaProvider(model=m, host=ollama_host)
            if fallback is not None:
                return fallback
            return OllamaProvider(model="llama3.1", host=ollama_host)

        nav_model = _resolve_llm(model)
        summ_model = _resolve_llm(summariser_model, fallback=nav_model)
        verif_model = _resolve_llm(verifier_model, fallback=nav_model)
        aggr_model = _resolve_llm(aggregator_model, fallback=nav_model)

        summariser = Summariser(
            llm=summ_model,
            max_concurrent=max_concurrent_summaries,
        )

        ingestor = IngestionEngine(
            storage=storage,
            summariser=summariser,
            parser_backend=parser_backend,
        )

        trace = ReasoningTrace(enabled=trace_enabled)
        agent = NavigationAgent(
            storage=storage,
            model=nav_model,
            verifier_model=verif_model,
            verify_leaves=verify_leaves,
            trace=trace,
        )

        # Optional: EmbeddingsEngine
        embeddings = None
        try:
            embeddings = EmbeddingsEngine()
            # Non-blocking attempt to load
            await embeddings.ensure_loaded()
        except Exception:
            embeddings = None

        aggregator = AggregatorAgent(model=aggr_model)

        instance = cls(
            storage,
            ingestor,
            agent,
            aggregator,
            trace_enabled=trace_enabled,
            embeddings=embeddings,
        )
        logger.info(
            "ApexIndex ready | db=%s | verify=%s | vectors=%s",
            db_url.split("?")[0],
            verify_leaves,
            "enabled" if embeddings and embeddings.is_available else "disabled",
        )
        return instance

    # -- Context manager support -------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Release DB connection pool and any other resources."""
        await self._storage.dispose()
        logger.info("ApexIndex closed.")

    # -- Ingestion API ------------------------------------------------------

    async def ingest(
        self,
        file_path: str | Path,
        *,
        doc_id: str | None = None,
        synthesize_summaries: bool = True,
    ) -> str:
        """
        Ingest a document file into the ApexRAG decision tree.

        Converts -> parses -> persists -> generates Semantic Map summaries
        -> builds page index. All steps are async and cancellation-safe.

        Args:
            file_path:            Path to the document (PDF, DOCX, MD, HTML, TXT).
            doc_id:               Override auto-generated ID (SHA-256 hash).
            synthesize_summaries: Call Ollama for summaries. Set False for tests.

        Returns:
            doc_id — use this for all subsequent query/tree/index calls.
        """
        async with self._lock:
            return await self._ingestor.ingest(
                file_path,
                doc_id=doc_id,
                synthesize_summaries=synthesize_summaries,
            )

    async def ingest_text(
        self,
        text: str,
        *,
        doc_id: str,
        synthesize_summaries: bool = True,
    ) -> str:
        """
        Ingest raw Markdown/plain text — no file required.

        Useful for:
        - Programmatic document creation
        - Unit testing without physical files
        - Streaming ingestion from pipelines

        Args:
            text:                 Raw Markdown or plain text.
            doc_id:               Required unique identifier.
            synthesize_summaries: Generate LLM summaries for tree nodes.

        Returns:
            The doc_id passed in.
        """
        async with self._lock:
            return await self._ingestor.ingest_text(
                text,
                doc_id=doc_id,
                synthesize_summaries=synthesize_summaries,
            )

    async def ingest_many(
        self,
        items: list[tuple[str, str | Path]],
        *,
        synthesize_summaries: bool = True,
    ) -> list[str]:
        """
        Batch-ingest multiple files/texts in parallel.

        Each item is either:
          - ``(doc_id, file_path)`` for file ingestion
          - ``(doc_id, text_content)`` for raw text ingestion

        All items are ingested concurrently using asyncio.gather, which
        significantly speeds up bulk-loading workflows.

        Args:
            items:                 List of ``(doc_id, path_or_text)`` tuples.
            synthesize_summaries: Generate LLM summaries for all items.

        Returns:
            List of ``doc_id`` strings in the same order as ``items``.

        Example::

            doc_ids = await index.ingest_many([
                ("doc1", "report.pdf"),
                ("doc2", Path("memo.md")),
                ("doc3", "# Manual Inline\nContent here"),
            ])
        """
        async with self._lock:
            tasks = []
            for doc_id, source in items:
                if isinstance(source, str) and not Path(source).exists():
                    # Treat as raw text
                    tasks.append(
                        self._ingestor.ingest_text(
                            source,
                            doc_id=doc_id,
                            synthesize_summaries=synthesize_summaries,
                        )
                    )
                else:
                    path = Path(source) if isinstance(source, str) else source
                    tasks.append(
                        self._ingestor.ingest(
                            path,
                            doc_id=doc_id,
                            synthesize_summaries=synthesize_summaries,
                        )
                    )
            return await asyncio.gather(*tasks)

    # -- Query API ----------------------------------------------------------

    async def query(
        self,
        question: str,
        doc_id: str,
        *,
        root_node_id: int | None = None,
        event_queue: asyncio.Queue[Any] | None = None,
        hybrid: bool = False,
    ) -> NavigationResult | None:
        """
        Navigate the document tree to answer `question`.

        The agent recursively walks the structural tree, makes LLM-guided
        decisions at each level, verifies the answer at the leaf, and
        returns the exact section content — no hallucinated blending.

        When ``hybrid=True``, the search combines:
          1. Vector similarity (semantic embeddings — requires sentence-transformers)
          2. Keyword / BM25 matches (SQLite FTS5)
          3. Agentic navigation (LLM-guided structural tree walking)

        Args:
            question:     Natural-language query.
            doc_id:       Target document (returned by ingest()).
            root_node_id: Restrict to a subtree (optional).
            event_queue:  Optional asyncio.Queue for real-time status updates.
            hybrid:       Enable hybrid search (vector + keyword + agentic).

        Returns:
            NavigationResult with .content, .path, .trace, .verified,
            or None if the answer could not be found.
        """
        if hybrid and self._embeddings and self._embeddings.is_available:
            # Hybrid ranking: combine vector + keyword + structural scores
            # to pre-filter candidates, then enrich the query with top section titles
            rankings = await self._search.hybrid_rank(question, doc_id)
            if rankings:
                # Add top section titles as context to guide the agent's navigation
                top_titles = [n.title for n, _ in rankings[:5]]
                hint_text = "; ".join(top_titles)
                enriched_question = (
                    f"{question}\n\n[Hybrid search suggests these sections: {hint_text}]"
                )
                logger.info(
                    "Hybrid: %d candidates ranked, enriching query with top sections",
                    len(rankings),
                )
                return await self._agent.find(
                    query=enriched_question,
                    doc_id=doc_id,
                    root_node_id=root_node_id,
                    event_queue=event_queue,
                )

        return await self._agent.find(
            query=question,
            doc_id=doc_id,
            root_node_id=root_node_id,
            event_queue=event_queue,
        )

    async def query_stream(
        self,
        question: str,
        doc_id: str,
        *,
        event_queue: asyncio.Queue[Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the navigation process token by token.

        Yields SSE-style JSON events as the agent:
        - Enters nodes
        - Evaluates children
        - Makes choices
        - Verifies leaves
        - Produces the final answer

        Args:
            question:    Natural-language query.
            doc_id:      Target document.
            event_queue: Optional external queue to feed events into.

        Yields:
            JSON strings with keys: event, node_id, title, path, etc.
        """
        # Create or use the provided event queue
        q: asyncio.Queue[Any] = event_queue or asyncio.Queue()

        # Launch the query in a background task
        task = asyncio.create_task(self.query(question, doc_id, event_queue=q))

        # Yield events as they arrive
        while True:
            done, _ = await asyncio.wait(
                [asyncio.create_task(q.get()), task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            while not q.empty():
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
            if task.done():
                break

    async def query_global(
        self,
        question: str,
        *,
        event_queue: asyncio.Queue[Any] | None = None,
        synthesize: bool = True,
        hybrid: bool = False,
    ) -> NavigationResult | str | None:
        """
        Query across ALL indexed documents.

        First, the agent identifies the most relevant documents based on their
        top-level summaries, then performs a structural search within each
        candidate until an answer is found.

        When ``hybrid=True``, vector similarity is used to rank documents
        before agentic navigation (requires sentence-transformers).

        Args:
            question:    Natural-language query.
            event_queue: Optional asyncio.Queue for real-time status updates.
            synthesize:  If True, use AggregatorAgent to synthesize a final answer.
            hybrid:      Enable vector-based document ranking.

        Returns:
            NavigationResult, synthesized string, or None.
        """
        if hybrid and self._embeddings and self._embeddings.is_available:
            # Use vector search to find relevant docs first
            candidates = await self._search.vector_search_global(question, top_k_docs=3)
            if candidates:
                for doc_id, _ in candidates:
                    result = await self._agent.find(
                        query=question,
                        doc_id=doc_id,
                        event_queue=event_queue,
                    )
                    if result:
                        if synthesize:
                            if event_queue:
                                await event_queue.put({"event": "synthesize_start"})
                            answer = await self._aggregator.synthesize(question, [result])
                            if event_queue:
                                await event_queue.put(
                                    {"event": "synthesize_done", "answer": answer}
                                )
                            return answer
                        return result

        result = await self._agent.find_global(
            query=question,
            event_queue=event_queue,
        )

        if result and synthesize:
            if event_queue:
                await event_queue.put({"event": "synthesize_start"})
            answer = await self._aggregator.synthesize(question, [result])
            if event_queue:
                await event_queue.put({"event": "synthesize_done", "answer": answer})
            return answer

        return result

    # -- Tree & Index API ---------------------------------------------------

    async def get_tree(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Return the complete document tree as a list of node dicts.

        Ordered by LTree path (depth-first). Used by the FastAPI index page
        to render the expandable tree UI.

        Returns:
            List of node dicts (see DocumentNode.to_dict()).
        """
        async with self._storage.session() as session:
            nodes = await self._storage.get_full_tree(session, doc_id)
            return [n.to_dict() for n in nodes]

    async def export_tree(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Export the document tree as a nested JSON structure (PageIndex format).

        This perfectly matches the original PageIndex output format, where
        each node contains a 'nodes' list of its children.

        Returns:
            List of root node dicts, each with nested 'nodes'.
        """
        async with self._storage.session() as session:
            flat_nodes = await self._storage.get_full_tree(session, doc_id)

        if not flat_nodes:
            return []

        # Convert to PageIndex dict format
        node_dicts: dict[int, dict[str, Any]] = {}
        for n in flat_nodes:
            node_dicts[n.id] = {
                "node_id": str(n.id),
                "title": n.title,
                "summary": n.summary,
                "start_index": n.page_start,
                "end_index": n.page_end,
                "content": n.content if n.content else "",
                "image_data": n.image_data,
                "nodes": [],
                "_parent_id": n.parent_id,  # Temporary for building tree
            }

        # Build the nested structure
        roots: list[dict[str, Any]] = []
        for n in flat_nodes:
            current = node_dicts[n.id]
            pid = current.pop("_parent_id", None)

            if pid is None:
                roots.append(current)
            else:
                if pid in node_dicts:
                    node_dicts[pid]["nodes"].append(current)

        return roots

    async def get_page_index(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Return the book-style alphabetical page index for a document.

        Each entry maps a section heading to its page range and node_id.
        Use this to build an index page like a textbook's back-of-book index.

        Returns:
            List of PageIndexEntry dicts sorted alphabetically by term.
        """
        async with self._storage.session() as session:
            entries = await self._storage.get_page_index(session, doc_id)
            return [e.to_dict() for e in entries]

    async def search_index(self, doc_id: str, query: str) -> list[dict[str, Any]]:
        """
        Full-text search over the page index terms (case-insensitive).

        Args:
            doc_id: Target document.
            query:  Search string (partial match).

        Returns:
            Matching PageIndexEntry dicts.
        """
        async with self._storage.session() as session:
            entries = await self._storage.search_page_index(session, doc_id, query)
            return [e.to_dict() for e in entries]

    async def get_stats(self, doc_id: str) -> dict[str, Any]:
        """
        Return aggregate statistics for a document.

        Returns:
            Dict with keys: doc_id, total_nodes, max_depth, leaf_count.
        """
        async with self._storage.session() as session:
            return await self._storage.get_document_stats(session, doc_id)

    # -- Management API -----------------------------------------------------

    async def delete(self, doc_id: str) -> int:
        """
        Delete all tree nodes and page index entries for a document.

        Args:
            doc_id: The document ID returned by ingest().

        Returns:
            Number of DocumentNodes deleted.

        Raises:
            DocumentNotFoundError: If the doc_id does not exist.
        """
        async with self._lock, self._storage.session() as session:
            docs: list[str] = list(await self._storage.list_documents(session))
            if doc_id not in docs:
                raise DocumentNotFoundError(
                    message=f"Document '{doc_id}' not found.",
                    hint="Use index.list_documents() to see available documents.",
                )
            return await self._storage.delete_document(session, doc_id)

    async def list_documents(self) -> list[str]:
        """Return all doc_ids currently stored in the index."""
        async with self._storage.session() as session:
            results = await self._storage.list_documents(session)
            return list(results)

    async def get_document(self, doc_id: str, node_id: int) -> dict[str, Any] | None:
        """
        Fetch a specific node by its primary key.

        Args:
            doc_id:  Document ID the node belongs to (for validation).
            node_id: Primary key of the DocumentNode.

        Returns:
            Node dict or None if not found or doc_id mismatch.

        Raises:
            DocumentNotFoundError: If the node doesn't exist.
        """
        async with self._storage.session() as session:
            node = await self._storage.get_node(session, node_id)
            if node is None or node.doc_id != doc_id:
                raise DocumentNotFoundError(
                    message=f"Node {node_id} not found in document '{doc_id}'.",
                    hint="Verify the doc_id and node_id. Use get_tree() to list available nodes.",
                )
            return node.to_dict()
