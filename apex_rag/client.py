"""
client.py -- Primary user-facing API for ApexRAG.

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
import inspect
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import networkx as nx
from typing_extensions import Self

# Agents
from apex_rag.agents.apex_orchestrator import ApexOrchestrator
from apex_rag.agents.audit.conformal_wrapper import ConformalWrapperAgent
from apex_rag.agents.audit.temporal_audit import TemporalAuditAgent
from apex_rag.agents.critic.agent import EvaluationCriticAgent
from apex_rag.agents.planner.agent import QueryPlannerAgent
from apex_rag.agents.synthesizer.agent import EvidenceSynthesizerAgent
from apex_rag.enterprise.auth.models import TenantContext

# Exceptions & Utilities
from apex_rag.exceptions import DocumentNotFoundError
from apex_rag.graph.edges.causal_builder import CausalGraphBuilder

# Ingestion
from apex_rag.ingestion.apex_parser import ApexParser
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.ingestion.embedding_engine import EmbeddingEngine

# Legacy / To be deprecated
from apex_rag.ingestion.semantic_model_builder import SemanticModelBuilder

# Unified Models
from apex_rag.models.unified_models import ApexAnswer, NodeType
from apex_rag.providers import (
    AnthropicProvider,
    AsyncLLM,
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvier,
)
from apex_rag.retrieval.agentic.navigator import ASTNavigationAgent
from apex_rag.utils import logger

# Legacy / To be deprecated & re-exported models
from apex_rag.ingestion.legacy import IngestionEngine  # noqa: F401
from apex_rag.navigation import AggregatorAgent, NavigationAgent, NavigationResult  # noqa: F401
from apex_rag.search import EmbeddingsEngine, HybridSearch  # noqa: F401
from apex_rag.storage import StorageEngine  # noqa: F401
from apex_rag.models.unified_models import ASTNode, EvidencePacket  # noqa: F401



class ApexIndex:
    """
    World-class, thread-safe facade for the ApexRAG research-grade library.

    Provides a unified entry point for temporally-aware, causally-linked,
    and uncertainty-quantified RAG.  ApexIndex orchestrates AST-based
    ingestion, knowledge graph construction, and conformal retrieval.

    Typical library usage::

        from apex_rag import ApexIndex

        async with await ApexIndex.create(provider="openai") as index:
            await index.ingest_file("annual_report.pdf")
            answer = await index.query("What was Q3 revenue?", coverage=0.90)
            print(answer.answer_text)

    Args:
        storage:          Unified :class:`ApexStorage` instance.
        parser:           :class:`ApexParser` for AST conversion.
        embedder:         :class:`EmbeddingEngine` for vector search.
        summariser:       :class:`SemanticModelBuilder` for AST signposts.
        graph_builder:    :class:`CausalGraphBuilder` for KGs.
        orchestrator:     :class:`ApexOrchestrator` for multi-agent loops.
    """

    def __init__(
        self,
        storage: ApexStorage,
        parser: ApexParser,
        embedder: EmbeddingEngine,
        summariser: SemanticModelBuilder,
        graph_builder: CausalGraphBuilder,
        orchestrator: ApexOrchestrator,
        *,
        llm: AsyncLLM | None = None,
        trace_enabled: bool = True,
    ) -> None:
        self._storage = storage
        self._parser = parser
        self._embedder = embedder
        self._summariser = summariser
        self._graph_builder = graph_builder
        self._orchestrator = orchestrator
        self._llm = llm

        self._trace_enabled = trace_enabled
        self._lock = asyncio.Lock()

    # -- Factory ------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        *,
        provider: str = "ollama",
        model: str | AsyncLLM | None = None,
        db_url: str = "sqlite+aiosqlite:///apex_rag.db",
        trace_enabled: bool = True,
        db_echo: bool = False,
        **kwargs: Any,
    ) -> ApexIndex:
        """
        Async factory - initialises all unified components and ensures DB schema.

        Args:
            provider:      LLM provider name ("openai", "anthropic", "groq", "ollama").
            model:         Specific model ID (e.g. "gpt-4o", "claude-3-5-sonnet").
            db_url:        SQLAlchemy async connection URL for ApexStorage.
            trace_enabled: Print colored reasoning traces.
            db_echo:       Log SQL queries (dev only).
            **kwargs:      Passed to the provider and orchestrator constructors.
        """
        # 1. Resolve Provider
        llm: AsyncLLM

        if hasattr(model, "generate") and hasattr(model, "embed"):
            llm = model  # type: ignore[assignment]
        elif hasattr(provider, "generate") and hasattr(provider, "embed"):
            llm = provider # type: ignore
        else:
            p_name = str(provider).lower()
            if p_name == "openai":
                llm = OpenAIProvider(model=model or "gpt-4o-mini", **kwargs)
            elif p_name == "anthropic":
                llm = AnthropicProvider(model=model or "claude-3-5-sonnet-20240620", **kwargs)
            elif p_name == "groq":
                llm = GroqProvider(model=model or "llama3-70b-8192", **kwargs)
            elif p_name == "gemini":
                llm = GeminiProvider(model=model or "gemini-1.5-flash", **kwargs)
            elif p_name == "openrouter":
                llm = OpenRouterProvier(model=model or "meta-llama/llama-3-70b-instruct", **kwargs)
            else:
                llm = OllamaProvider(model=model or "llama3.1", **kwargs)

        # 2. Initialise Unified Storage
        storage = await ApexStorage.create(db_url, echo=db_echo)

        # 3. Initialise Ingestion Engines
        parser = ApexParser()
        embedder = EmbeddingEngine(embedder=llm)
        summariser = SemanticModelBuilder(llm=llm)
        graph_builder = CausalGraphBuilder(embedder=embedder, llm=llm)

        # 4. Initialise Agents
        planner = QueryPlannerAgent(llm=llm)

        # ASTNavigationAgent requires a retriever and verifier
        from apex_rag.retrieval.deterministic.keyword import KeywordDeterministicRetriever
        from apex_rag.retrieval.verification.strict_verifier import StrictLeafVerifier

        retriever = KeywordDeterministicRetriever()
        verifier = StrictLeafVerifier(llm=llm)

        navigator = ASTNavigationAgent(
            storage=storage,
            model=llm,
            retriever=retriever,
            verifier=verifier
        )

        critic = EvaluationCriticAgent(llm=llm)
        synthesizer = EvidenceSynthesizerAgent(llm=llm)
        temporal_auditor = TemporalAuditAgent()
        conformal_wrapper = ConformalWrapperAgent(coverage_level=0.90)

        orchestrator = ApexOrchestrator(
            planner=planner,
            navigator=navigator,
            critic=critic,
            synthesizer=synthesizer,
            temporal_auditor=temporal_auditor,
            conformal_wrapper=conformal_wrapper,
        )

        return cls(
            storage=storage,
            parser=parser,
            embedder=embedder,
            summariser=summariser,
            graph_builder=graph_builder,
            orchestrator=orchestrator,
            llm=llm,
            trace_enabled=trace_enabled,
        )

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

    async def ingest_file(
        self,
        file_path: str | Path,
        *,
        doc_id: str | None = None,
        source_date: datetime | None = None,
        synthesize_summaries: bool = True,
    ) -> str:
        """
        Ingest a file into the ApexRAG AST-based four-layer architecture.

        Pipeline:
            1. Parse -> Universal AST Nodes
            2. Signpost -> 2-sentence summaries for navigation
            3. Embed -> Async batched vector embeddings
            4. Causal -> Automatic edge discovery (Supports/Overrides/etc.)
            5. Store -> Persist to ApexStorage

        Args:
            file_path:   Path to PDF, DOCX, MD, or PY file.
            doc_id:      Override auto-generated document ID.
            source_date: Optional authorship date (extracted from meta if None).

        Returns:
            The document ID.
        """
        async with self._lock:
            # 1. Parse
            nodes = await self._parser.parse_file(file_path, doc_id=doc_id)
            resolved_doc_id = nodes[0].doc_id

            # 2. Signpost (Summarise)
            if synthesize_summaries:
                signposts = await self._summariser.build_signposts(nodes)
                for node in nodes:
                    if node.node_id in signposts:
                        node.content = f"{signposts[node.node_id]}\n\n{node.content}"

            # 3. Embed
            await self._embedder.embed_nodes(nodes)

            # 4. Causal Edge Discovery
            edges = await self._graph_builder.build_all(nodes)

            # 5. Persistence (Save nodes first to satisfy foreign keys)
            await self._storage.save_nodes(nodes)
            for edge in edges:
                await self._storage.save_causal_edge(edge.to_causal_edge())

            # 6. Page Index Generation (Book-style index for headings)
            page_entries = []
            for node in nodes:
                if node.node_type == NodeType.HEADING:
                    page_entries.append({
                        "node_id": node.node_id,
                        "doc_id": resolved_doc_id,
                        "term": node.content,
                        "page_number": node.page_number
                    })
            if page_entries:
                result = self._storage.save_page_index_entries(page_entries)
                if inspect.isawaitable(result):
                    await result

            logger.info("Ingested document %s: %d nodes", resolved_doc_id, len(nodes))
            return resolved_doc_id

    async def ingest(
        self,
        file_path: str | Path,
        *,
        doc_id: str | None = None,
        source_date: datetime | None = None,
        synthesize_summaries: bool = True,
    ) -> str:
        """Backward-compatible alias for :meth:`ingest_file`."""
        return await self.ingest_file(
            file_path,
            doc_id=doc_id,
            source_date=source_date,
            synthesize_summaries=synthesize_summaries,
        )

    async def ingest_text(
        self,
        text: str,
        *,
        doc_id: str,
        source_date: datetime | None = None,
        synthesize_summaries: bool = True,
    ) -> str:
        """
        Ingest raw Markdown/plain text into the AST architecture.

        Args:
            text:        Raw Markdown or plain text.
            doc_id:      Required unique identifier.
            source_date: Optional authorship date.

        Returns:
            The document ID.
        """
        async with self._lock:
            # 1. Parse
            nodes = self._parser.parse_markdown(
                text, doc_id=doc_id, source_date=source_date
            )

            # 2. Signpost
            if synthesize_summaries:
                signposts = await self._summariser.build_signposts(nodes)
                for node in nodes:
                    if node.node_id in signposts:
                        node.content = f"{signposts[node.node_id]}\n\n{node.content}"

            # 3. Embed
            await self._embedder.embed_nodes(nodes)

            # 4. Causal Edge Discovery
            edges = await self._graph_builder.build_all(nodes)

            # 5. Persistence (Save nodes first to satisfy foreign keys)
            await self._storage.save_nodes(nodes)
            for edge in edges:
                await self._storage.save_causal_edge(edge.to_causal_edge())

            # 6. Page Index Generation (Book-style index for headings)
            page_entries = []
            for node in nodes:
                if node.node_type == NodeType.HEADING:
                    page_entries.append({
                        "node_id": node.node_id,
                        "doc_id": doc_id,
                        "term": node.content,
                        "page_number": node.page_number
                    })
            if page_entries:
                result = self._storage.save_page_index_entries(page_entries)
                if inspect.isawaitable(result):
                    await result

            logger.info("Ingested text %s: %d nodes", doc_id, len(nodes))
            return doc_id

    async def ingest_many(
        self,
        items: list[tuple[str, str | Path]],
    ) -> list[str]:
        """
        Batch-ingest multiple files/texts concurrently.

        Args:
            items: List of (doc_id, path_or_text) tuples.

        Returns:
            List of document IDs.
        """
        tasks = []
        for doc_id, source in items:
            if isinstance(source, str) and not Path(source).exists():
                tasks.append(self.ingest_text(source, doc_id=doc_id))
            else:
                tasks.append(self.ingest_file(source, doc_id=doc_id))
        return list(await asyncio.gather(*tasks))

    # -- Query API ----------------------------------------------------------

    async def query(
        self,
        question: str,
        doc_id: str,
        *,
        coverage: float = 0.90,
        domain: str = "general",
        ablation_mode: bool = False,
        root_node_id: str | int | None = None,
        event_queue: asyncio.Queue[Any] | None = None,
        tenant_context: TenantContext | None = None,
    ) -> ApexAnswer:
        """
        Query the index with a mathematically rigorous confidence guarantee.

        Args:
            question: Natural-language query.
            doc_id:   The document to search.
            coverage: Target conformal coverage level (e.g. 0.90).
            domain:   Strategic domain for freshness decay ("legal", "financial", etc.).
            ablation_mode: Run in Baseline C mode (AST only).

        Returns:
            An :class:`ApexAnswer` containing the text, evidence, and coverage info.
        """
        # Set target coverage on the orchestrator's conformal wrapper
        self._orchestrator.conformal_wrapper.coverage_level = coverage

        if event_queue is not None:
            await event_queue.put({"event": "start", "doc_id": doc_id, "question": question})

        answer = await self._orchestrator.run(
            query=question,
            doc_id=doc_id,
            domain=domain,
            ablation_mode=ablation_mode,
            tenant_context=tenant_context,
        )

        if answer is None:
            # Return an empty answer object rather than None for API consistency
            return ApexAnswer(
                answer_text="I could not find enough verified evidence to answer your query.",
                query=question,
            )

        if event_queue is not None:
            await event_queue.put({"event": "done", "doc_id": doc_id})

        return answer

    async def stream_query(
        self,
        question: str,
        doc_id: str,
        *,
        domain: str = "general",
        tenant_id: str = "default",
        tenant_context: TenantContext | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream answer tokens as they are generated.

        Args:
            question: Natural-language query.
            doc_id:   The document to search.
            domain:   Strategic domain for freshness decay.

        Yields:
            Token chunks from the LLM.
        """
        ctx = tenant_context
        if ctx is None and tenant_id:
            ctx = TenantContext(tenant_id=tenant_id, user_id="inferred-user-id", roles=["Guest"])

        async for chunk in self._orchestrator.stream(
            query=question,
            doc_id=doc_id,
            domain=domain,
            tenant_context=ctx,
        ):
            yield chunk

    async def query_global(
        self,
        question: str,
        *,
        synthesize: bool = True,
        coverage: float = 0.90,
        domain: str = "general",
        event_queue: asyncio.Queue[Any] | None = None,
    ) -> ApexAnswer | None:
        """Query all indexed documents and return the first answer with evidence."""
        docs = await self.list_documents()
        best_answer: ApexAnswer | None = None

        for doc_id in docs:
            if event_queue is not None:
                await event_queue.put({"event": "searching", "doc_id": doc_id})
            answer = await self.query(
                question,
                doc_id,
                coverage=coverage,
                domain=domain,
                event_queue=None,
            )
            if answer.evidence_packets:
                return answer
            if best_answer is None:
                best_answer = answer

        return best_answer

    # -- Research & Explainability API --------------------------------------

    async def get_causal_graph(self) -> nx.DiGraph:
        """
        Construct and return the full Causal Knowledge Graph.

        Returns:
            A NetworkX Directed Graph where nodes are ASTNode IDs and edges
            are typed causal relationships.
        """
        edges = await self._storage.get_all_edges()
        graph = nx.DiGraph()

        for edge in edges:
            graph.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                edge_id=edge.edge_id,
                type=edge.edge_type,
                strength=edge.strength,
                evidence=edge.evidence,
            )

        logger.info("Generated Causal Graph: %d edges", len(edges))
        return graph

    async def calibrate(self, calibration_file: str | Path) -> float:
        """
        Update the conformal prediction layer with a new calibration set.

        Args:
            calibration_file: Path to a JSON file containing
                             (query, correct_node_id) pairs.

        Returns:
            The new q_hat threshold value.
        """
        path = Path(calibration_file)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        data = json.loads(path.read_text())
        # The ConformalWrapperAgent needs alpha scores (nonconformity)
        # to compute the quantile. This usually requires running the retriever.
        # For simplicity, we delegate to the wrapper's calibrate method if it exists,
        # otherwise we log a warning.
        if hasattr(self._orchestrator.conformal_wrapper, "calibrate_from_data"):
             return await self._orchestrator.conformal_wrapper.calibrate_from_data(data)

        logger.warning("Calibration not supported by current wrapper implementation.")
        return 0.0

    async def explain(self, node_id: str) -> dict[str, Any]:
        """
        Return the full temporal and causal context for a specific ASTNode.

        Args:
            node_id: The UUID4 identifier of the node.

        Returns:
            A dictionary containing node content, freshness, and all
            incoming/outgoing causal edges.
        """
        node = await self._storage.get_node(node_id)
        if not node:
            raise DocumentNotFoundError(f"Node {node_id} not found.")

        temporal = await self._storage.get_temporal_metadata(node_id)
        edges = await self._storage.get_edges_for_node(node_id)

        return {
            "node": node.model_dump(),
            "temporal": temporal.model_dump() if temporal else None,
            "edges": [e.model_dump() for e in edges],
        }

    # -- Tree & Index API (To be refactored) --------------------------------

    async def get_tree(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Return the complete document tree as a list of node dicts.

        Ordered depth-first. Used by UIs to render expandable trees.

        Returns:
            List of node dicts.
        """
        nodes = await self._storage.get_nodes_by_doc(doc_id)
        # Sort by depth and then by some order if possible,
        # but for now we just return them.
        return [n.model_dump() for n in nodes]

    async def export_tree(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Export the document tree as a nested JSON structure.

        Each node contains a 'children_nodes' list of its children.

        Returns:
            List of root node dicts, each with nested 'children_nodes'.
        """
        flat_nodes = await self._storage.get_nodes_by_doc(doc_id)
        if not flat_nodes:
            return []

        node_map = {n.node_id: n.model_dump() for n in flat_nodes}
        for n in node_map.values():
            n["children_nodes"] = []

        roots = []
        for n in node_map.values():
            parent_id = n.get("parent_id")
            if parent_id is None:
                roots.append(n)
            elif parent_id in node_map:
                node_map[parent_id]["children_nodes"].append(n)
            else:
                # Parent not in this doc (shouldn't happen) or root
                roots.append(n)

        return roots

    async def get_page_index(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Return the book-style alphabetical page index for a document.

        Each entry maps a section heading to its page range and node_id.
        Use this to build an index page like a textbook's back-of-book index.

        Returns:
            List of PageIndexEntry dicts sorted alphabetically by term.
        """
        return await self._storage.get_page_index_entries(doc_id)

    async def search_index(self, doc_id: str, query: str) -> list[dict[str, Any]]:
        """
        Full-text search over the page index terms (case-insensitive).

        Args:
            doc_id: Target document.
            query:  Search string (partial match).

        Returns:
            Matching PageIndexEntry dicts.
        """
        return await self._storage.search_page_index(doc_id, query)

    # -- Management API -----------------------------------------------------

    async def get_stats(self, doc_id: str) -> dict[str, Any]:
        """
        Return aggregate statistics for a document.

        Returns:
            Dict with keys: doc_id, total_nodes, max_depth, leaf_count.
        """
        return await self._storage.get_document_stats(doc_id)

    async def delete(self, doc_id: str) -> int:
        """
        Delete all tree nodes, temporal meta, and causal edges for a document.

        Args:
            doc_id: The document ID returned by ingest_file().

        Returns:
            Number of nodes deleted.
        """
        async with self._lock:
            return await self._storage.delete_document(doc_id)

    async def list_documents(self) -> list[str]:
        """Return all doc_ids currently stored in the index."""
        return await self._storage.list_document_ids()

    async def get_nodes(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Fetch all nodes for a specific document.

        Returns:
            List of node dictionaries.
        """
        nodes = await self._storage.get_nodes_by_doc(doc_id)
        return [n.model_dump() for n in nodes]
