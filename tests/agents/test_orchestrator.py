"""
tests/agents/test_orchestrator.py — Tests for the enhanced Orchestrator.

Covers:
    - Basic Plan → Navigate → Critic success
    - Iterative refinement (critic fails → re-plan → retry)
    - Max iterations exceeded (best-effort fallback)
    - No evidence retrieved
    - Integrated pipeline (temporal scoring, contradiction, causal graph, synthesis)
    - Integrated pipeline with missing components (graceful fallback)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apex_rag.agents.orchestrator import Orchestrator
from apex_rag.models.unified_models import ASTNode as UnifiedASTNode
from apex_rag.models.unified_models import NodeType
from apex_rag.retrieval.agentic.navigator import ASTNavigationResult

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_planner():
    m = AsyncMock()
    m.plan.return_value = ["Q2 revenue?", "Q3 revenue?"]
    return m


@pytest.fixture
def mock_navigator():
    m = AsyncMock()
    m.find.side_effect = [
        ASTNavigationResult(
            node=UnifiedASTNode(
                content="Q2 is $40M",
                node_id="11111111-1111-4111-8111-111111111111",
                doc_id="doc1",
                node_type=NodeType.PARAGRAPH,
            ),
            path="/Q2",
            title="Q2 Revenue",
            trace=[],
            verified=True,
            confidence=0.95,
        ),
        ASTNavigationResult(
            node=UnifiedASTNode(
                content="Q3 is $50M",
                node_id="22222222-2222-4222-8222-222222222222",
                doc_id="doc1",
                node_type=NodeType.PARAGRAPH,
            ),
            path="/Q3",
            title="Q3 Revenue",
            trace=[],
            verified=True,
            confidence=0.96,
        ),
    ]
    return m


@pytest.fixture
def mock_critic():
    m = AsyncMock()
    m.evaluate.return_value = True
    return m


@pytest.fixture
def mock_synthesizer():
    m = AsyncMock()
    m.synthesize.return_value = "Q2 was $40M and Q3 was $50M."
    return m


@pytest.fixture
def mock_causal_builder():
    m = AsyncMock()
    m.build_all.return_value = []
    return m


@pytest.fixture
def mock_causal_retriever():
    m = AsyncMock()
    m.build_chain.return_value = []
    return m


@pytest.fixture
def mock_contradiction_detector():
    m = AsyncMock()
    m.detect_all.return_value = []
    return m


@pytest.fixture
def orchestrator(mock_planner, mock_navigator, mock_critic):
    return Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
        max_iterations=3,
    )


# ═══════════════════════════════════════════════════════════════════════
# Basic execution test
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_basic_success(orchestrator, mock_critic):
    """Plan → Navigate → Critic → returns 2 packets."""
    result = await orchestrator.execute_query("Compare Q2 and Q3", "doc1")

    assert result is not None
    assert len(result) == 2
    assert result[0].node.content == "Q2 is $40M"
    assert result[1].node.content == "Q3 is $50M"
    assert result[0].retrieval_score == 0.95
    assert result[1].retrieval_score == 0.96

    mock_critic.evaluate.assert_called_once()


@pytest.mark.asyncio
async def test_no_evidence(orchestrator):
    """Navigator returns nothing → None."""
    orchestrator.navigator.find = AsyncMock(return_value=None)

    result = await orchestrator.execute_query("Nonexistent", "doc1")
    assert result is None


# ═══════════════════════════════════════════════════════════════════════
# Iterative refinement tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_iterative_refinement_recovers(orchestrator):
    """Critic fails first time, then recovers on second iteration."""
    # First call → fail, second call → pass
    orchestrator.critic.evaluate = AsyncMock(side_effect=[False, True])

    # Planner should be called twice
    orchestrator.planner.plan = AsyncMock(return_value=["Q2 revenue?"])

    # Navigator returns something for both iterations
    nav_result = ASTNavigationResult(
        node=UnifiedASTNode(
            content="Q2 is $40M",
            node_id="11111111-1111-4111-8111-111111111111",
            doc_id="doc1",
            node_type=NodeType.PARAGRAPH,
        ),
        path="/Q2",
        title="Q2",
        trace=[],
        verified=True,
        confidence=0.9,
    )
    orchestrator.navigator.find = AsyncMock(return_value=nav_result)

    result = await orchestrator.execute_query("What is Q2 revenue?", "doc1")

    assert result is not None
    assert len(result) == 1
    assert result[0].node.content == "Q2 is $40M"
    assert orchestrator.planner.plan.call_count == 2
    assert orchestrator.critic.evaluate.call_count == 2


@pytest.mark.asyncio
async def test_iterative_refinement_exhausted(orchestrator):
    """Critic always fails — returns best-effort after max iterations."""
    orchestrator.critic.evaluate = AsyncMock(return_value=False)

    nav_result = ASTNavigationResult(
        node=UnifiedASTNode(
            content="Some content",
            node_id="11111111-1111-4111-8111-111111111111",
            doc_id="doc1",
            node_type=NodeType.PARAGRAPH,
        ),
        path="/",
        title="Content",
        trace=[],
        verified=True,
        confidence=0.5,
    )
    orchestrator.navigator.find = AsyncMock(return_value=nav_result)
    orchestrator.planner.plan = AsyncMock(return_value=["Q2 revenue?"])

    result = await orchestrator.execute_query("Query", "doc1")

    # Returns best-effort (not None)
    assert result is not None
    assert len(result) >= 1
    # Should have called critic 3 times (max_iterations=3)
    assert orchestrator.critic.evaluate.call_count >= 3


@pytest.mark.asyncio
async def test_iterative_refinement_max_iterations_custom(orchestrator):
    """Custom max_iterations=1 means only one try — no refinement."""
    orchestrator.critic.evaluate = AsyncMock(return_value=False)

    nav_result = ASTNavigationResult(
        node=UnifiedASTNode(
            content="Content",
            node_id="11111111-1111-4111-8111-111111111111",
            doc_id="doc1",
            node_type=NodeType.PARAGRAPH,
        ),
        path="/",
        title="Content",
        trace=[],
        verified=True,
        confidence=0.5,
    )
    orchestrator.navigator.find = AsyncMock(return_value=nav_result)
    orchestrator.planner.plan = AsyncMock(return_value=["sub?"])

    result = await orchestrator.execute_query("Query", "doc1", max_iterations=1)

    assert result is not None  # best-effort
    assert orchestrator.critic.evaluate.call_count >= 1


@pytest.mark.asyncio
async def test_missing_context_included_in_replan(orchestrator):
    """When critic fails, the next plan includes missing context."""
    call_count = 0

    async def plan_side_effect(query: str) -> list[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ["sub_a", "sub_b"]
        # Second call — should include missing context
        assert "missing" in query.lower() or "previous" in query.lower()
        return ["sub_a"]

    orchestrator.critic.evaluate = AsyncMock(side_effect=[False, True])
    orchestrator.planner.plan = AsyncMock(side_effect=plan_side_effect)

    nav_result = ASTNavigationResult(
        node=UnifiedASTNode(
            content="Answer",
            node_id="11111111-1111-4111-8111-111111111111",
            doc_id="doc1",
            node_type=NodeType.PARAGRAPH,
        ),
        path="/",
        title="Ans",
        trace=[],
        verified=True,
        confidence=0.8,
    )
    orchestrator.navigator.find = AsyncMock(return_value=nav_result)

    result = await orchestrator.execute_query("Original question", "doc1")
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# Integrated pipeline tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_execute_query_integrated_full(
    mock_planner,
    mock_navigator,
    mock_critic,
    mock_synthesizer,
    mock_causal_builder,
    mock_causal_retriever,
    mock_contradiction_detector,
):
    """Full integrated pipeline returns a complete ApexAnswer."""
    orchestrator = Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
        synthesizer=mock_synthesizer,
        causal_builder=mock_causal_builder,
        causal_retriever=mock_causal_retriever,
        contradiction_detector=mock_contradiction_detector,
    )

    result = await orchestrator.execute_query_integrated(
        "Compare Q2 and Q3", "doc1", domain="financial"
    )

    assert result is not None
    assert result.answer_text == "Q2 was $40M and Q3 was $50M."
    assert len(result.evidence_packets) == 2
    assert result.temporal_freshness >= 0.0
    assert result.prediction_set_size == 2
    assert result.query == "Compare Q2 and Q3"
    assert result.latency_ms > 0
    # Contradiction detector and causal builder were called
    mock_causal_builder.build_all.assert_called_once()
    mock_causal_retriever.build_chain.assert_called_once()
    mock_contradiction_detector.detect_all.assert_called_once()
    mock_synthesizer.synthesize.assert_called_once()


@pytest.mark.asyncio
async def test_execute_query_integrated_no_evidence(orchestrator):
    """No evidence retrieved → None."""
    orchestrator.navigator.find = AsyncMock(return_value=None)

    result = await orchestrator.execute_query_integrated("Query", "doc1")
    assert result is None


@pytest.mark.asyncio
async def test_execute_query_integrated_no_synthesizer(mock_planner, mock_navigator, mock_critic):
    """Without a synthesizer, falls back to formatted text."""
    orchestrator = Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
    )

    result = await orchestrator.execute_query_integrated("Compare Q2 and Q3", "doc1")

    assert result is not None
    assert "Q2 is $40M" in result.answer_text or "Source" in result.answer_text
    assert len(result.evidence_packets) == 2


@pytest.mark.asyncio
async def test_execute_query_integrated_no_causal_components(
    mock_planner, mock_navigator, mock_critic, mock_synthesizer
):
    """Without causal builder/retriever, still returns valid ApexAnswer."""
    orchestrator = Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
        synthesizer=mock_synthesizer,
    )

    result = await orchestrator.execute_query_integrated("Compare Q2 and Q3", "doc1")

    assert result is not None
    assert result.causal_chain == []
    assert result.contradictions == []


@pytest.mark.asyncio
async def test_execute_query_integrated_temporal_scoring(
    mock_planner, mock_navigator, mock_critic, mock_synthesizer
):
    """Temporal scoring assigns freshness scores to packets."""
    orchestrator = Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
        synthesizer=mock_synthesizer,
    )

    with patch("apex_rag.agents.orchestrator.FreshnessScorer") as MockScorer:
        mock_scorer_instance = MagicMock()
        mock_scorer_instance.compute.return_value = 0.85
        MockScorer.return_value = mock_scorer_instance

        result = await orchestrator.execute_query_integrated(
            "Compare Q2 and Q3", "doc1", domain="legal"
        )

        assert result is not None
        assert result.temporal_freshness == 0.85
        MockScorer.assert_called_once_with(domain="legal")


@pytest.mark.asyncio
async def test_execute_query_integrated_latency_measured(
    mock_planner, mock_navigator, mock_critic, mock_synthesizer
):
    """Latency is measured and reported."""
    orchestrator = Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
        synthesizer=mock_synthesizer,
    )

    result = await orchestrator.execute_query_integrated("Query", "doc1")
    assert result is not None
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_execute_query_integrated_contradictions_chain(
    mock_planner, mock_navigator, mock_critic, mock_synthesizer
):
    """Causal edges from both contradiction detector and retriever are merged."""
    from apex_rag.models.unified_models import CausalEdge, EdgeType

    node_a_id = "11111111-1111-4111-8111-111111111111"
    node_b_id = "22222222-2222-4222-8222-222222222222"

    fake_contradiction = CausalEdge(
        source_node_id=node_a_id,
        target_node_id=node_b_id,
        edge_type=EdgeType.CONTRADICTS,
        evidence="Test contradiction",
    )

    class FakeBuilder:
        async def build_all(self, nodes, **kwargs):
            from apex_rag.graph.edges.models import GraphEdge, RelationType

            return [
                GraphEdge(
                    source_id=node_a_id,
                    target_id=node_b_id,
                    relation_type=RelationType.SUPPORTS,
                    strength=0.7,
                )
            ]

    class FakeRetriever:
        async def build_chain(self, seed_nodes, **kwargs):
            return [fake_contradiction]

    class FakeDetector:
        async def detect_all(self, nodes):
            return [fake_contradiction]

    orchestrator = Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
        synthesizer=mock_synthesizer,
        causal_builder=FakeBuilder(),
        causal_retriever=FakeRetriever(),
        contradiction_detector=FakeDetector(),
    )

    result = await orchestrator.execute_query_integrated("Query", "doc1")
    assert result is not None
    assert len(result.contradictions) >= 1
    assert len(result.causal_chain) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_single_sub_query(orchestrator):
    """Single sub-query should work fine."""
    orchestrator.planner.plan = AsyncMock(return_value=["What is Q2?"])
    nav_result = ASTNavigationResult(
        node=UnifiedASTNode(
            content="Q2 is $40M",
            node_id="11111111-1111-4111-8111-111111111111",
            doc_id="doc1",
            node_type=NodeType.PARAGRAPH,
        ),
        path="/Q2",
        title="Q2",
        trace=[],
        verified=True,
        confidence=0.9,
    )
    orchestrator.navigator.find = AsyncMock(return_value=nav_result)

    result = await orchestrator.execute_query("Q2?", "doc1")
    assert result is not None
    assert len(result) == 1


@pytest.mark.asyncio
async def test_partial_retrieval(orchestrator):
    """Only some sub-queries resolve — should still return partial results."""
    orchestrator.planner.plan = AsyncMock(return_value=["Q2?", "Q3?", "Q4?"])
    nav_results = [
        ASTNavigationResult(
            node=UnifiedASTNode(
                content="Q2 is $40M",
                node_id="11111111-1111-4111-8111-111111111111",
                doc_id="doc1",
                node_type=NodeType.PARAGRAPH,
            ),
            path="/Q2",
            title="Q2",
            trace=[],
            verified=True,
            confidence=0.9,
        ),
        None,  # Q3 fails
        ASTNavigationResult(
            node=UnifiedASTNode(
                content="Q4 is $60M",
                node_id="33333333-3333-4333-8333-333333333333",
                doc_id="doc1",
                node_type=NodeType.PARAGRAPH,
            ),
            path="/Q4",
            title="Q4",
            trace=[],
            verified=True,
            confidence=0.85,
        ),
    ]
    # Repeat nav_results enough for multiple iterations
    orchestrator.navigator.find = AsyncMock(side_effect=nav_results * 5)
    orchestrator.critic.evaluate = AsyncMock(return_value=False)

    result = await orchestrator.execute_query("Multi", "doc1")
    # Returns whatever was found (best-effort after critic failure)
    assert result is not None
    # Should have at least 2 packets (Q2 and Q4)
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_unverified_navigation_results(orchestrator):
    """Unverified results are filtered out by _navigate_all."""
    orchestrator.planner.plan = AsyncMock(return_value=["Q2?"])

    unverified = ASTNavigationResult(
        node=UnifiedASTNode(
            content="Q2 is $40M",
            node_id="11111111-1111-4111-8111-111111111111",
            doc_id="doc1",
            node_type=NodeType.PARAGRAPH,
        ),
        path="/Q2",
        title="Q2",
        trace=[],
        verified=False,
        confidence=0.3,
    )
    orchestrator.navigator.find = AsyncMock(return_value=unverified)

    result = await orchestrator.execute_query("Q2?", "doc1")
    assert result is None  # No verified results
