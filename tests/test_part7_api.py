"""
tests/test_part7_api.py — Verification of the Part 7 Public API refactor.

Tests the new unified ApexIndex factory, advanced ingestion pipeline,
and uncertainty-quantified querying.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import networkx as nx

from apex_rag import ApexIndex
from apex_rag.models.unified_models import ASTNode, ApexAnswer, EvidencePacket, NodeType, TemporalMetadata


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Mocked response")
    llm.embed = AsyncMock(return_value=[[0.1] * 384])
    # Mock stream_generate as an async generator
    async def _stream(*args, **kwargs):
        yield "Mocked "
        yield "stream "
        yield "response"
    llm.stream_generate = _stream
    return llm


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.save_nodes = AsyncMock()
    storage.save_causal_edge = AsyncMock()
    storage.get_node = AsyncMock(return_value=None)
    storage.get_nodes_by_doc = AsyncMock(return_value=[])
    storage.get_all_edges = AsyncMock(return_value=[])
    storage.dispose = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_apex_index_create_factory(mock_llm) -> None:
    """Test the one-line setup via provider string."""
    with patch("apex_rag.client.OpenAIProvider", return_value=mock_llm):
        with patch("apex_rag.ingestion.apex_storage.ApexStorage.create") as mock_storage_create:
            mock_storage_create.return_value = AsyncMock()
            
            index = await ApexIndex.create(provider="openai", api_key="sk-test")
            
            assert isinstance(index, ApexIndex)
            assert index._orchestrator is not None
            # Verify provider was resolved correctly (internally)
            # Since we patched OpenAIProvider, it should have been called


@pytest.mark.asyncio
async def test_ingest_file_pipeline(mock_llm) -> None:
    """Test the full ingestion pipeline (Parse -> Signpost -> Embed -> Causal -> Store)."""
    # Create a dummy file
    dummy_file = Path("test_doc.md")
    dummy_file.write_text("# Section 1\nContent 1")

    try:
        # Mocking components
        mock_storage = MagicMock()
        mock_storage.save_nodes = AsyncMock()
        mock_storage.save_causal_edge = AsyncMock()
        mock_storage.dispose = AsyncMock()

        # We need to mock ApexParser to return at least one node
        node1 = ASTNode(
            node_id="00000000-0000-4000-8000-000000000001",
            content="Section 1",
            node_type=NodeType.HEADING,
            doc_id="test-doc",
        )
        
        with patch("apex_rag.ingestion.apex_parser.ApexParser.parse_file", return_value=[node1]):
            with patch("apex_rag.ingestion.semantic_model_builder.SemanticModelBuilder.build_signposts", return_value={node1.node_id: "Summary 1"}):
                # Instantiate ApexIndex with mocked dependencies
                index = ApexIndex(
                    storage=mock_storage,
                    parser=MagicMock(), # Replaced by patch
                    embedder=MagicMock(),
                    summariser=MagicMock(),
                    graph_builder=MagicMock(),
                    orchestrator=MagicMock(),
                )
                # Overwrite internal mocked attributes with functional ones
                index._parser = MagicMock()
                index._parser.parse_file = AsyncMock(return_value=[node1])
                index._summariser = MagicMock()
                index._summariser.build_signposts = AsyncMock(return_value={node1.node_id: "Summary 1"})
                index._embedder = MagicMock()
                index._embedder.embed_nodes = AsyncMock()
                index._graph_builder = MagicMock()
                index._graph_builder.build_all = AsyncMock(return_value=[])

                # Mock the fact pipeline to avoid storage session setup
                mock_pipeline = AsyncMock()
                mock_pipeline.enqueue_document = AsyncMock()
                index._get_fact_pipeline = MagicMock(return_value=mock_pipeline)

                doc_id = await index.ingest_file(dummy_file)
                
                assert doc_id == "test-doc"
                index._parser.parse_file.assert_called_once()
                index._summariser.build_signposts.assert_called_once()
                index._embedder.embed_nodes.assert_called_once()
                index._graph_builder.build_all.assert_called_once()
                mock_storage.save_nodes.assert_called_once()

    finally:
        if dummy_file.exists():
            dummy_file.unlink()


@pytest.mark.asyncio
async def test_query_conformal_guarantee(mock_llm) -> None:
    """Test querying with coverage guarantee."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.run = AsyncMock(return_value=ApexAnswer(
        answer_text="Synthesized answer",
        evidence_packets=[],
        coverage_guarantee=0.95,
        prediction_set_size=3,
        query="Test query",
    ))
    mock_orchestrator.conformal_wrapper = MagicMock()

    index = ApexIndex(
        storage=MagicMock(),
        parser=MagicMock(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        graph_builder=MagicMock(),
        orchestrator=mock_orchestrator,
    )

    answer = await index.query("Test query", doc_id="doc1", coverage=0.95)

    assert isinstance(answer, ApexAnswer)
    assert answer.answer_text == "Synthesized answer"
    assert answer.coverage_guarantee == 0.95
    assert mock_orchestrator.conformal_wrapper.coverage_level == 0.95
    mock_orchestrator.run.assert_called_once()


@pytest.mark.asyncio
async def test_get_causal_graph_utility() -> None:
    """Test retrieval of the causal graph as a NetworkX object."""
    from apex_rag.models.unified_models import CausalEdge, EdgeType

    edge1 = CausalEdge(
        edge_id="00000000-0000-4000-8000-0000000000a1",
        source_node_id="00000000-0000-4000-8000-000000000001",
        target_node_id="00000000-0000-4000-8000-000000000002",
        edge_type=EdgeType.SUPPORTS,
    )

    mock_storage = MagicMock()
    mock_storage.get_all_edges = AsyncMock(return_value=[edge1])

    index = ApexIndex(
        storage=mock_storage,
        parser=MagicMock(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        graph_builder=MagicMock(),
        orchestrator=MagicMock(),
    )

    graph = await index.get_causal_graph()

    assert isinstance(graph, nx.DiGraph)
    assert len(graph.edges) == 1
    assert "00000000-0000-4000-8000-000000000001" in graph.nodes
    assert graph.edges["00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"]["type"] == EdgeType.SUPPORTS


@pytest.mark.asyncio
async def test_explain_node_utility() -> None:
    """Test the explain() method for node provenance."""
    node_id = "00000000-0000-4000-8000-000000000001"
    node = ASTNode(
        node_id=node_id,
        content="Passage content",
        node_type=NodeType.PARAGRAPH,
        doc_id="doc1",
    )
    temporal = TemporalMetadata(node_id=node_id, freshness_score=0.9)

    mock_storage = MagicMock()
    mock_storage.get_node = AsyncMock(return_value=node)
    mock_storage.get_temporal_metadata = AsyncMock(return_value=temporal)
    mock_storage.get_edges_for_node = AsyncMock(return_value=[])

    index = ApexIndex(
        storage=mock_storage,
        parser=MagicMock(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        graph_builder=MagicMock(),
        orchestrator=MagicMock(),
    )

    explanation = await index.explain(node_id)

    assert explanation["node"]["node_id"] == node_id
    assert explanation["temporal"]["freshness_score"] == 0.9
    assert isinstance(explanation["edges"], list)
