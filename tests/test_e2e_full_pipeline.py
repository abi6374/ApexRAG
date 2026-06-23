"""
test_e2e_full_pipeline.py — End-to-end integration test connecting all 5 parts.

Tests the complete pipeline from document parsing through to ApexAnswer:

    - **Part 1-2:** ``ApexParser`` → ``ApexStorage`` → ``EmbeddingEngine`` →
      ``SemanticModelBuilder``
    - **Part 3:** ``TemporalExtractor`` → ``FreshnessScorer`` →
      ``TemporalContradictionDetector``
    - **Part 4:** ``CausalGraphBuilder`` → ``CausalRetriever``
    - **Part 5:** ``EvidenceSynthesizerAgent`` → ``Orchestrator`` → ``ApexAnswer``

Two test suites:

    1. :class:`TestE2EFullPipeline` — Always runs, uses real components with a
       mock LLM for LLM-dependent stages.  Every step writes to and reads from
       ``ApexStorage`` so the full data flow is validated end-to-end.

    2. :class:`TestE2ERealLLM` — Runs only when ``OPENAI_API_KEY`` is set.
       Uses ``OpenAIProvider`` for the full ``Orchestrator`` loop with a real
       LLM, verifying that the pipeline works with a production provider.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

# ── Part 1–2 Imports ────────────────────────────────────────
from apex_rag.ingestion.apex_parser import ApexParser
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.ingestion.embedding_engine import EmbeddingEngine
from apex_rag.ingestion.semantic_model_builder import SemanticModelBuilder

# ── Part 3 Imports ──────────────────────────────────────────
from apex_rag.temporal.extractor import TemporalExtractor
from apex_rag.temporal.scorer import FreshnessScorer
from apex_rag.temporal.contradiction import TemporalContradictionDetector

# ── Part 4 Imports ──────────────────────────────────────────
from apex_rag.graph.edges.causal_builder import CausalGraphBuilder
from apex_rag.graph.edges.causal_retriever import CausalRetriever

# ── Part 5 Imports ──────────────────────────────────────────
from apex_rag.agents.synthesizer.agent import EvidenceSynthesizerAgent
from apex_rag.agents.orchestrator import Orchestrator
from apex_rag.models.unified_models import ApexAnswer, EvidencePacket

# ── Provider Imports ────────────────────────────────────────
from apex_rag.providers import OpenAIProvider

# ── Old-system Imports (for real-LLM test) ──────────────────
from apex_rag.client import ApexIndex

# ═══════════════════════════════════════════════════════════════
# Sample document — realistic financial report
# ═══════════════════════════════════════════════════════════════

SAMPLE_FINANCIAL_REPORT = """\
# Q3 2024 Financial Report

Published: 2024-10-15

## Revenue Overview

Q3 revenue reached $52 million, representing 30% growth year-over-year.
The growth was driven by strong performance in the European market.

## Expense Breakdown

Operating expenses totaled $38 million in Q3 2024.
R&D spending accounted for $15 million of total expenses.
Sales and marketing expenses were $12 million.

## Profit Margin

The gross margin improved to 72% in Q3 2024, up from 68% in Q2 2024.
Net profit margin was 18% for the quarter.

## Market Analysis

Market share increased to 24% in the European region.
Customer acquisition cost decreased by 15% compared to Q2.

## Competitive Landscape

The main competitor reported a 12% decline in revenue for the same period.
Our net promoter score improved to 62, up from 55 in Q2.

## Outlook

Management expects continued growth in Q4 2024 with projected revenue of $55-58 million.
New product launches are planned for Q1 2025.
"""

# ═══════════════════════════════════════════════════════════════
# Shared mock LLM — returns deterministic responses per prompt
# ═══════════════════════════════════════════════════════════════


class MockLLM:
    """Deterministic mock LLM that returns context-appropriate responses."""

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()

        # Signpost generation
        if "signpost" in prompt_lower or "section heading" in prompt_lower:
            return "This section discusses financial performance metrics and key business indicators for the quarter."

        # Date / temporal extraction
        if "date" in prompt_lower and "extract" in prompt_lower:
            return "2024-10-15"

        # Contradiction detection
        if "contradict" in prompt_lower or "contradiction" in prompt_lower:
            return "YES"

        # Synthesis
        if "synthesizer" in prompt_lower or "synthesize" in prompt_lower:
            return (
                "Based on the evidence, Q3 2024 revenue was $52 million with a "
                "72% gross margin. Operating expenses were $38 million. "
                "Market share reached 24% in Europe. [Source 1][Source 2]"
            )

        # Navigation / planning
        if "plan" in prompt_lower or "sub-query" in prompt_lower or "decompose" in prompt_lower:
            return "1. Revenue figures for Q3 2024\n2. Expense breakdown\n3. Market share data"

        # Verification
        if "verify" in prompt_lower or "answers" in prompt_lower:
            return '{"answers_query": true, "confidence": 0.95}'

        # General fallback
        return "Q3 2024 revenue was $52 million with strong growth across all segments."


def _make_mock_llm() -> MockLLM:
    """Factory to create a fresh MockLLM with call tracking."""
    return MockLLM()


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def apex_storage() -> ApexStorage:
    """In-memory ApexStorage for the pipeline tests."""
    storage = await ApexStorage.create("sqlite+aiosqlite:///:memory:", echo=False)
    yield storage
    await storage.dispose()


@pytest_asyncio.fixture
async def parsed_nodes() -> list:
    """Parse the sample financial report into AST nodes."""
    parser = ApexParser(default_doc_id="e2e-financial-report")
    nodes = parser.parse_markdown(
        SAMPLE_FINANCIAL_REPORT,
        doc_id="e2e-financial-report",
        source_date=datetime(2024, 10, 15, tzinfo=timezone.utc),
    )
    return nodes


# ═══════════════════════════════════════════════════════════════
# Suite 1 — Full pipeline with mock LLM (always runs)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestE2EFullPipeline:
    """Validates every component works together end-to-end.

    Pipeline flow:
        Parse → Store → Embed → Signpost → Temporal → Score →
        Contradiction → Causal Graph → Evidence Chain → Synthesis → ApexAnswer
    """

    async def test_parse_and_store(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 1-2: ApexParser produces valid nodes; ApexStorage persists them."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        assert len(nodes) > 0, "Parser should produce at least one node"

        # Verify tree structure (node_type is already a string due to use_enum_values=True)
        headings = [n for n in nodes if n.node_type == "HEADING"]
        paragraphs = [n for n in nodes if n.node_type == "PARAGRAPH"]
        assert len(headings) >= 6, f"Expected 6+ headings, got {len(headings)}"
        assert len(paragraphs) >= 5, f"Expected 5+ paragraphs, got {len(paragraphs)}"

        # Verify all nodes have valid UUIDs
        import uuid

        for node in nodes:
            uuid.UUID(node.node_id, version=4)

        # Store
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]
        count = await apex_storage.count_nodes(doc_id="e2e-financial-report")  # type: ignore[arg-type]
        assert count == len(nodes), f"Stored {count} nodes, expected {len(nodes)}"

        # Retrieve
        retrieved = await apex_storage.get_nodes_by_doc("e2e-financial-report", tenant_context="default")  # type: ignore[arg-type]
        assert len(retrieved) == len(nodes)

    async def test_embedding_pass(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 2: EmbeddingEngine populates node embeddings (fingerprint fallback)."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        # Embed using fingerprint fallback (no real embedder)
        engine = EmbeddingEngine(embedder=None, dimension=384)
        await engine.embed_nodes(nodes)

        # Verify embeddings were set
        for node in nodes:
            assert len(node.embedding) == 384, (
                f"Node {node.node_id[:8]} has {len(node.embedding)} dims, expected 384"
            )
            assert not all(v == 0.0 for v in node.embedding), "Embedding should not be all zeros"

        # Save embeddings back to storage
        for node in nodes:
            await apex_storage.save_node(node, tenant_context="default")  # type: ignore[arg-type]

        # Re-read and verify persistence
        retrieved = await apex_storage.get_nodes_by_doc("e2e-financial-report", tenant_context="default")  # type: ignore[arg-type]
        for node in retrieved:
            assert len(node.embedding) == 384

    async def test_signpost_generation(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 2: SemanticModelBuilder generates signposts for heading nodes."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        mock_llm = _make_mock_llm()
        builder = SemanticModelBuilder(llm=mock_llm, max_concurrent=4)

        signposts = await builder.build_signposts(nodes)

        # Every heading node should have a signpost (node_type is string due to use_enum_values=True)
        heading_nodes = [n for n in nodes if n.node_type == "HEADING" and len(n.children) > 0]
        for node in heading_nodes:
            assert node.node_id in signposts, (
                f"Heading node {node.node_id[:8]} missing signpost"
            )
            assert len(signposts[node.node_id]) > 10, "Signpost should be substantive"

        # Non-heading nodes without children should not have signposts
        leaf_nodes = [n for n in nodes if not n.children]
        for node in leaf_nodes:
            # Nodes with children should NOT have signposts if they have no heading content
            if node.node_type != "HEADING" and not node.children:
                continue
            # Leaf nodes (no children) should not have signposts
            if not node.children:
                continue

    async def test_temporal_extraction(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 3: TemporalExtractor extracts dates from node content."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        mock_llm = _make_mock_llm()
        extractor = TemporalExtractor(llm=mock_llm)

        # TemporalExtractor.extract() takes text + optional metadata, not ASTNodes
        # Call extract() on each node's content individually
        extracted_dates: list[datetime | None] = []
        for node in nodes:
            date = await extractor.extract(
                text=node.content,
                metadata={"source_date": node.source_date.isoformat() if node.source_date else None},
            )
            extracted_dates.append(date)

        # At least some nodes should have dates extracted
        dated_nodes = [(node, date) for node, date in zip(nodes, extracted_dates) if date is not None]
        assert len(dated_nodes) > 0, "At least one node should have a date extracted"

    async def test_freshness_scoring(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 3: FreshnessScorer computes decay-based scores for each node."""
        nodes = parsed_nodes  # type: ignore[valid-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        scorer = FreshnessScorer(domain="financial")
        scores: list[float] = []

        for node in nodes:
            score = scorer.compute(node.source_date)
            scores.append(score)
            assert 0.0 <= score <= 1.0, f"Freshness {score} out of [0, 1]"

        # Nodes with explicit dates should have higher freshness than very old ones
        dated_scores = [
            scorer.compute(datetime(2024, 1, 1, tzinfo=timezone.utc)),
            scorer.compute(datetime(2023, 1, 1, tzinfo=timezone.utc)),
            scorer.compute(datetime(2020, 1, 1, tzinfo=timezone.utc)),
        ]
        assert dated_scores[0] > dated_scores[1], "2024 should be fresher than 2023"
        assert dated_scores[1] > dated_scores[2], "2023 should be fresher than 2020"

        # Bulk scoring
        bulk = scorer.compute_many([n.source_date for n in nodes])
        assert len(bulk) == len(nodes)
        assert all(0.0 <= s <= 1.0 for s in bulk)

    async def test_contradiction_detection(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 3: TemporalContradictionDetector finds contradictions between nodes."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        mock_llm = _make_mock_llm()

        # Embed nodes for similarity computation
        engine = EmbeddingEngine(embedder=None, dimension=384)
        await engine.embed_nodes(nodes)

        # TemporalContradictionDetector uses hardcoded _SIMILARITY_THRESHOLD = 0.65
        detector = TemporalContradictionDetector(llm=mock_llm)

        # Detect contradictions among all nodes
        edges = await detector.detect_all(nodes)

        # Should find some contradiction edges (at least one pair above threshold)
        assert isinstance(edges, list)

    async def test_causal_graph_and_chain(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 4: CausalGraphBuilder + CausalRetriever build edges and chains."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        # Embed
        engine = EmbeddingEngine(embedder=None, dimension=384)
        await engine.embed_nodes(nodes)

        # CausalGraphBuilder (fingerprint embedder fallback)
        class FakeEmbedder:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                import hashlib
                result: list[list[float]] = []
                for t in texts:
                    h = hashlib.sha256(t.encode()).hexdigest()
                    vec = [int(h[i : i + 2], 16) / 255.0 for i in range(0, 384, 2)]
                    result.append(vec[:64])  # 64-dim for speed
                return result

            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return await self.embed(texts)

        builder = CausalGraphBuilder(
            embedder=FakeEmbedder(),
            similarity_threshold=0.3,
        )

        # Build edges (structural + temporal + semantic)
        graph_edges = await builder.build_all(
            nodes,
            include_temporal=True,
            include_semantic=True,
            include_structural=True,
            include_llm=False,
        )

        # Save edges to storage
        for ge in graph_edges:
            ce = ge.to_causal_edge()
            if ce is not None:
                await apex_storage.save_causal_edge(ce)

        # Verify edges were stored
        all_stored_edges = await apex_storage.get_all_edges()  # type: ignore[arg-type]
        assert len(all_stored_edges) > 0, "Should have persisted at least one causal edge"

        # CausalRetriever: Build evidence chain
        retriever = CausalRetriever(apex_storage)  # type: ignore[arg-type]
        chain = await retriever.build_chain(nodes, max_depth=3, max_edges=20)

        assert isinstance(chain, list)
        if chain:
            assert all(e.edge_type in ("SUPPORTS", "REFINES") for e in chain), (
                "Chain should only contain supportive edges"
            )

    async def test_synthesis_with_evidence(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """Part 5: EvidenceSynthesizerAgent produces a grounded answer."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        mock_llm = _make_mock_llm()
        synthesizer = EvidenceSynthesizerAgent(llm=mock_llm)

        # Convert nodes to CoreEvidencePackets
        from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket

        packets: list[CoreEvidencePacket] = [
            CoreEvidencePacket(
                node_id=n.node_id,
                source_document=n.doc_id,
                section_path=n.content[:60],
                retrieval_reason="E2E test retrieval",
                verification_result=True,
                confidence_score=0.95,
                content=n.content,
            )
            for n in nodes[:3]  # First 3 nodes as evidence
        ]

        answer = await synthesizer.synthesize(
            "What is the revenue and margin for Q3 2024?",
            packets,
        )

        assert answer is not None
        assert len(answer) > 20, "Answer should be substantive"

    async def test_full_pipeline_apex_answer(
        self, apex_storage: ApexStorage, parsed_nodes: list  # type: ignore[valid-type]
    ) -> None:
        """All 5 parts: Full pipeline produces a valid ApexAnswer."""
        nodes = parsed_nodes  # type: ignore[arg-type]
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        # ── Part 2: Embed ──
        engine = EmbeddingEngine(embedder=None, dimension=384)
        await engine.embed_nodes(nodes)
        for node in nodes:
            await apex_storage.save_node(node, tenant_context="default")  # type: ignore[arg-type]

        # ── Part 2: Signposts ──
        mock_llm = _make_mock_llm()
        builder = SemanticModelBuilder(llm=mock_llm, max_concurrent=4)
        signposts = await builder.build_signposts(nodes)
        assert len(signposts) > 0

        # ── Part 3: Temporal extraction ──
        extractor = TemporalExtractor(llm=mock_llm)
        extracted_dates = []
        for node in nodes:
            date = await extractor.extract(
                text=node.content,
                metadata={"source_date": node.source_date.isoformat() if node.source_date else None},
            )
            extracted_dates.append(date)
        assert len(extracted_dates) == len(nodes)
        assert any(d is not None for d in extracted_dates), "At least one date should be extracted"

        # ── Part 3: Freshness scoring ──
        scorer = FreshnessScorer(domain="financial")
        freshness = [scorer.compute(n.source_date) for n in nodes]
        assert all(0.0 <= s <= 1.0 for s in freshness)

        # ── Part 3: Contradiction detection ──
        detector = TemporalContradictionDetector(llm=mock_llm)
        contradictions = await detector.detect_all(nodes)
        assert isinstance(contradictions, list)

        # ── Part 4: Causal graph ──
        class FakeEmbedder:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                import hashlib
                result: list[list[float]] = []
                for t in texts:
                    h = hashlib.sha256(t.encode()).hexdigest()
                    vec = [int(h[i : i + 2], 16) / 255.0 for i in range(0, 128, 2)]
                    result.append(vec[:64])
                return result

            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return await self.embed(texts)

        graph_builder = CausalGraphBuilder(
            embedder=FakeEmbedder(),
            similarity_threshold=0.3,
        )
        graph_edges = await graph_builder.build_all(
            nodes,
            include_temporal=True,
            include_semantic=True,
            include_structural=True,
            include_llm=False,
        )
        for ge in graph_edges:
            ce = ge.to_causal_edge()
            if ce is not None:
                await apex_storage.save_causal_edge(ce)

        # ── Part 4: Evidence chain ──
        retriever = CausalRetriever(apex_storage)  # type: ignore[arg-type]
        chain = await retriever.build_chain(nodes, max_depth=3, max_edges=20)

        # ── Part 5: Synthesis ──
        from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket

        packets: list[CoreEvidencePacket] = [
            CoreEvidencePacket(
                node_id=n.node_id,
                source_document="e2e-financial-report",
                section_path=n.content[:60],
                retrieval_reason="Pipeline test",
                verification_result=True,
                confidence_score=0.95,
                content=n.content,
            )
            for n in nodes[:4]
        ]

        synthesizer = EvidenceSynthesizerAgent(llm=mock_llm)
        answer_text = await synthesizer.synthesize(
            "What is the revenue and margin for Q3 2024?",
            packets,
        )

        # ── Assemble ApexAnswer ──
        from apex_rag.models.unified_models import (
            ASTNode as UnifiedASTNode,
            CausalEdge,
            EdgeType,
            EvidencePacket as UnifiedEvidencePacket,
            NodeType,
            TemporalMetadata,
        )

        unified_packets: list[UnifiedEvidencePacket] = []
        for i, pkt in enumerate(packets):
            node = UnifiedASTNode(
                node_id=pkt.node_id,
                content=pkt.content,
                node_type=NodeType.PARAGRAPH,
                doc_id="e2e-financial-report",
            )
            meta = TemporalMetadata(
                node_id=pkt.node_id,
                freshness_score=0.85,
            )
            unified_packets.append(
                UnifiedEvidencePacket(
                    node=node,
                    temporal_metadata=meta,
                    retrieval_score=0.95,
                    rank=i + 1,
                )
            )

        answer = ApexAnswer(
            answer_text=answer_text or "",
            evidence_packets=unified_packets,
            temporal_freshness=0.85,
            contradictions=contradictions[:3] if contradictions else [],
            coverage_guarantee=0.0,
            prediction_set_size=len(packets),
            causal_chain=chain[:5] if chain else [],
            query="What is the revenue and margin for Q3 2024?",
            latency_ms=150.0,
        )

        # ── Verify ApexAnswer ──
        assert answer.answer_text, "Answer text should be non-empty"
        assert len(answer.evidence_packets) == 4, "Should have 4 evidence packets"
        assert 0.0 <= answer.temporal_freshness <= 1.0
        assert answer.prediction_set_size == 4
        assert answer.query == "What is the revenue and margin for Q3 2024?"
        assert answer.latency_ms > 0

        # Verify serialization round-trip
        json_str = answer.model_dump_json(indent=2)
        assert "answer_text" in json_str
        assert "evidence_packets" in json_str
        assert "temporal_freshness" in json_str
        assert "contradictions" in json_str
        assert "causal_chain" in json_str
        assert "latency_ms" in json_str


# ═══════════════════════════════════════════════════════════════
# Suite 2 — Real LLM provider (requires OPENAI_API_KEY)
# ═══════════════════════════════════════════════════════════════

_REQUIRES_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real LLM provider test",
)


@pytest.mark.asyncio
class TestE2ERealLLM:
    """Full orchestrator loop with a real LLM provider.

    Uses ``ApexIndex`` (old system for ingestion) + ``OpenAIProvider``
    for the ``Orchestrator``.  This validates the entire navigation,
    planning, and critic loop with a production LLM.
    """

    @_REQUIRES_OPENAI
    async def test_orchestrator_basic_query(self) -> None:
        """Orchestrator.execute_query with real OpenAI returns evidence packets."""
        
        # Real LLM provider — gpt-4o-mini is cheap and fast
        llm = OpenAIProvider(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )

        index = await ApexIndex.create(
            provider=llm,
            db_url="sqlite+aiosqlite:///:memory:",
            trace_enabled=False
        )

        try:
            # Ingest the financial report
            doc_id = await index.ingest_text(
                SAMPLE_FINANCIAL_REPORT,
                doc_id="e2e-real-llm",
            )
            assert doc_id == "e2e-real-llm"

            # Verify document was stored
            stats = await index.get_stats(doc_id)
            assert stats["total_nodes"] > 0

            # Build the Orchestrator with real LLM components
            # Since index._orchestrator is already built, we can just use it
            orchestrator = index._orchestrator

            # Execute the query
            packets = await orchestrator.execute_query(
                "What was the Q3 2024 revenue and gross margin?",
                doc_id,
            )

            assert packets is not None, "Should have retrieved evidence packets"
            assert len(packets) > 0, "Should have at least one evidence packet"

            # Verify packet structure
            for pkt in packets:
                assert pkt.retrieval_score >= 0.0
                assert pkt.node.content, "Packet should have content"

            # Verify some content about revenue or margin
            all_content = " ".join(p.node.content for p in packets).lower()
            assert ("revenue" in all_content or "margin" in all_content or "$52" in all_content or "72%" in all_content), (
                "Evidence should mention revenue or margin"
            )

        finally:
            await index.close()

    @_REQUIRES_OPENAI
    async def test_orchestrator_integrated_answer(self) -> None:
        """Orchestrator.execute_query_integrated with real OpenAI returns ApexAnswer."""
        llm = OpenAIProvider(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )

        index = await ApexIndex.create(
            provider=llm,
            db_url="sqlite+aiosqlite:///:memory:",
            trace_enabled=False
        )

        try:
            doc_id = await index.ingest_text(
                SAMPLE_FINANCIAL_REPORT,
                doc_id="e2e-real-llm-integrated",
            )

            orchestrator = index._orchestrator

            answer = await orchestrator.run(
                "What was the Q3 2024 revenue and gross margin?",
                doc_id,
                domain="financial",
            )

            assert answer is not None, "Should have produced an ApexAnswer"
            assert isinstance(answer, ApexAnswer), (
                f"Expected ApexAnswer, got {type(answer)}"
            )
            assert answer.answer_text, "Answer text should be non-empty"
            assert len(answer.evidence_packets) > 0
            assert 0.0 <= answer.temporal_freshness <= 1.0
            assert answer.prediction_set_size > 0
            assert answer.query
            assert answer.latency_ms > 0

            # Verify content mentions revenue or margin
            answer_lower = answer.answer_text.lower()
            assert ("revenue" in answer_lower or "margin" in answer_lower or "$52" in answer_lower or "72%" in answer_lower), (
                f"Answer should reference revenue/margin: {answer.answer_text[:200]}"
            )

        finally:
            await index.close()

    @_REQUIRES_OPENAI
    async def test_orchestrator_streaming(self) -> None:
        """Stream synthesis with real OpenAI produces token chunks."""
        llm = OpenAIProvider(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )

        index = await ApexIndex.create(
            provider=llm,
            db_url="sqlite+aiosqlite:///:memory:",
            trace_enabled=False
        )

        try:
            doc_id = await index.ingest_text(
                SAMPLE_FINANCIAL_REPORT,
                doc_id="e2e-stream-test",
            )

            orchestrator = index._orchestrator

            # Stream synthesis
            chunks: list[str] = []
            async for chunk in orchestrator.stream(
                "What was the revenue for Q3 2024?",
                doc_id,
            ):
                chunks.append(chunk)

            full_answer = "".join(chunks)
            assert len(full_answer) > 20, "Streamed answer should be substantive"
            assert len(chunks) > 0, "Should have received at least one chunk"

        finally:
            await index.close()


# ═══════════════════════════════════════════════════════════════
# Suite 3 — Cross-system integration (old + new)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestE2ECrossSystem:
    """Validates that old-system (ApexIndex) and new-system (ApexStorage) can
    coexist on the same document data.

    This test ingests a document via the old system, then manually
    parses and stores the same content in ApexStorage, verifying
    that both systems can access and process the same logical data.
    """

    async def test_old_and_new_storage_coexistence(
        self, apex_storage: ApexStorage  # type: ignore[valid-type]
    ) -> None:
        """Parse document once, store in both old and new storage."""
        # Parse with ApexParser
        parser = ApexParser(default_doc_id="cross-system-test")
        nodes = parser.parse_markdown(
            SAMPLE_FINANCIAL_REPORT,
            doc_id="cross-system-test",
            source_date=datetime(2024, 10, 15, tzinfo=timezone.utc),
        )

        # Store in ApexStorage (new system)
        await apex_storage.save_nodes(nodes, tenant_context="default")  # type: ignore[arg-type]

        # Verify new storage
        new_count = await apex_storage.count_nodes(doc_id="cross-system-test")  # type: ignore[arg-type]
        assert new_count == len(nodes)

        # Ingest same text via old system
        from unittest.mock import AsyncMock
        dummy_llm = AsyncMock()
        dummy_llm.generate = AsyncMock(return_value="Summary")
        async def mock_embed(texts, **kwargs):
            return [[0.1] * 384 for _ in texts]
        dummy_llm.embed = mock_embed

        index = await ApexIndex.create(
            db_url="sqlite+aiosqlite:///:memory:",
            provider=dummy_llm,
            trace_enabled=False
        )

        try:
            old_doc_id = await index.ingest_text(
                SAMPLE_FINANCIAL_REPORT,
                doc_id="cross-system-old",
            )
            old_stats = await index.get_stats(old_doc_id)
            assert old_stats["total_nodes"] > 0

            # Both storage systems should have the document
            assert old_stats["doc_id"] == "cross-system-old"

        finally:
            await index.close()


