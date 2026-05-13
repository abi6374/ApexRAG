"""
client.py — Primary user-facing API for ApexRAG.

`ApexIndex` is the single entry point for all ApexRAG operations.
It is designed to be:
  - Thread-safe via asyncio locks (compatible with FastAPI, Starlette, etc.)
  - Context-manager friendly for clean resource management
  - Simple: six main methods — ingest(), ingest_text(), query(), delete(),
    get_tree(), get_page_index()
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
from pathlib import Path
from types import TracebackType
from typing import Any, Self, Sequence

from apex_rag.ingestion import IngestionEngine, Summariser
from apex_rag.navigation import NavigationAgent, NavigationResult
from apex_rag.providers import AsyncLLM, OllamaProvider
from apex_rag.storage import DocumentNode, PageIndexEntry, StorageEngine
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
        *,
        trace_enabled: bool = True,
    ) -> None:
        self._storage = storage
        self._ingestor = ingestor
        self._agent = agent
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
        ollama_host: str = "http://localhost:11434",
        max_concurrent_summaries: int = 4,
        parser_backend: str = "markitdown",
        trace_enabled: bool = True,
        verify_leaves: bool = True,
        db_echo: bool = False,
    ) -> "ApexIndex":
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

        instance = cls(storage, ingestor, agent, trace_enabled=trace_enabled)
        logger.info(
            "ApexIndex ready | db=%s | verify=%s",
            db_url.split("?")[0],
            verify_leaves,
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

        Converts → parses → persists → generates Semantic Map summaries
        → builds page index. All steps are async and cancellation-safe.

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

    # -- Query API ----------------------------------------------------------

    async def query(
        self,
        question: str,
        doc_id: str,
        *,
        root_node_id: int | None = None,
    ) -> NavigationResult | None:
        """
        Navigate the document tree to answer `question`.

        The agent recursively walks the structural tree, makes LLM-guided
        decisions at each level, verifies the answer at the leaf, and
        returns the exact section content — no hallucinated blending.

        Args:
            question:     Natural-language query.
            doc_id:       Target document (returned by ingest()).
            root_node_id: Restrict to a subtree (optional).

        Returns:
            NavigationResult with .content, .path, .trace, .verified,
            or None if the answer could not be found.
        """
        return await self._agent.find(
            query=question,
            doc_id=doc_id,
            root_node_id=root_node_id,
        )

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
        node_dicts = {}
        for n in flat_nodes:
            node_dicts[n.id] = {
                "node_id": str(n.id),
                "title": n.title,
                "summary": n.summary,
                "start_index": n.page_start,
                "end_index": n.page_end,
                "content": n.content if n.content else "",
                "nodes": [],
                "_parent_id": n.parent_id # Temporary for building tree
            }

        # Build the nested structure
        roots = []
        for n in flat_nodes:
            current = node_dicts[n.id]
            pid = current.pop("_parent_id")
            
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
        """
        async with self._lock:
            async with self._storage.session() as session:
                return await self._storage.delete_document(session, doc_id)

    async def list_documents(self) -> list[str]:
        """Return all doc_ids currently stored in the index."""
        async with self._storage.session() as session:
            results = await self._storage.list_documents(session)
            return list(results)
