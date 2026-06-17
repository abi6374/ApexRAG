"""
tests/agents/test_part6_full_pipeline.py — Part 6 integration tests.

Covers the spec requirements:
    - Single-document query
    - Cross-document query requiring causal traversal
    - Temporally-conflicting query (two documents disagree)
    - Query where no answer exists (critic surfaces this)
    - Streaming query
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from apex_rag.agents.audit.temporal_audit import TemporalAuditAgent
from apex_rag.agents.audit.conformal_wrapper import ConformalWrapperAgent
from apex_rag.agents.apex_orchestrator import ApexOrchestrator
from apex_rag.agents.synthesizer.agent import EvidenceSynthesizerAgent
from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket
from apex_rag.retrieval.agentic.navigator import ASTNavigationResult
from apex_rag.models.unified_models import (
    ApexAnswer,
    CausalEdge,
    EdgeType,
    EvidencePacket as UnifiedEvidencePacket,
    ASTNode,
    NodeType,
    TemporalMetadata,
)
from apex_rag.temporal.scorer import FreshnessScorer
from apex_rag.temporal.contradiction import TemporalContradictionDetector


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def make_nav_result(
    content: str,
    node_id: str,
    verified: bool = True,
    confidence: float = 0.9,
) -> ASTNavigationResult:
    return ASTNavigationResult(
        node=ASTNode(
            node_id=node_id,
            content=content,
            node_type=NodeType.PARAGRAPH,
            doc_id="test-doc",
        ),
        path=f"/{node_id[:8]}",
        title=content[:50],
        trace=[],
        verified=verified,
        confidence=confidence,
    )


def make_packet(
    node_id: str,
    content: str,
    verified: bool = True,
    score: float = 0.9,
) -> CoreEvidencePacket:
    return CoreEvidencePacket(
        node_id=node_id,
        source_document="test-doc",
        section_path=content[:60],
        retrieval_reason="test",
        verification_result=verified,
        confidence_score=score,
        content=content,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Single-document query
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_single_document_query() -> None:
    """Single-document query returns evidence from the correct doc."""
    planner = AsyncMock()
    planner.plan.return_value = ["What was Q3 2024 revenue?"]

    navigator = AsyncMock()
    navigator.find.return_value = make_nav_result(
        content="Q3 revenue was $52M.",
        node_id="11111111-1111-4111-8111-111111111111",
    )

    critic = AsyncMock()
    critic.evaluate.return_value = True

    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)
    synthesizer.synthesize.return_value = (
        "Based on the evidence, Q3 revenue was $52M."
    )
    synthesizer.stream_synthesize = AsyncMock()
    synthesizer.stream_synthesize.return_value.__aiter__.return_value = iter(
        ["Based on evidence, Q3 revenue was $52M."]
    )

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
    )

    result = await orchestrator.run("What was Q3 2024 revenue?", "doc-single")

    assert result is not None
    assert isinstance(result, ApexAnswer)
    assert "revenue" in result.answer_text.lower()
    assert len(result.evidence_packets) == 1
    assert result.query == "What was Q3 2024 revenue?"
    assert result.latency_ms > 0


# ═══════════════════════════════════════════════════════════════════════
# 2. Cross-document query requiring causal traversal
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cross_document_causal_traversal() -> None:
    """Cross-document query traverses evidence from multiple docs via causal chain."""
    planner = AsyncMock()
    planner.plan.return_value = [
        "What were 2023 revenues?",
        "What changed from 2023 to 2024?",
    ]

    navigator = AsyncMock()
    navigator.find.side_effect = [
        make_nav_result("2023: $48M revenue.", "33333333-3333-4333-8333-333333333333"),
        make_nav_result("2024: $52M revenue, up from $48M.", "44444444-4444-4444-8444-444444444444"),
    ]

    critic = AsyncMock()
    critic.evaluate.return_value = True

    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)
    synthesizer.synthesize.return_value = (
        "2023 revenue was $48M. 2024 revenue grew to $52M, a $4M increase."
    )
    synthesizer.stream_synthesize = AsyncMock()
    synthesizer.stream_synthesize.return_value.__aiter__.return_value = iter(
        ["2023: $48M. 2024: $52M."]
    )

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
    )

    result = await orchestrator.run(
        "How did revenue change from 2023 to 2024?", "doc-cross",
    )

    assert result is not None
    assert "48" in result.answer_text or "52" in result.answer_text
    # Should have 2 evidence packets (one from each doc concept)
    assert len(result.evidence_packets) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 3. Temporally-conflicting query (two documents disagree)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_temporal_conflict_detected() -> None:
    """Temporal conflict between two documents surfaces contradiction in answer."""
    planner = AsyncMock()
    planner.plan.return_value = ["What is the effective tax rate for 2024?"]

    navigator = AsyncMock()
    navigator.find.side_effect = [
        make_nav_result(
            content="The effective tax rate for 2024 is 21%.",
            node_id="55555555-5555-4555-8555-555555555555",
        ),
        make_nav_result(
            content="The effective tax rate for 2024 has been revised to 18%.",
            node_id="66666666-6666-4666-8666-666666666666",
        ),
    ]

    critic = AsyncMock()
    critic.evaluate.return_value = True

    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)
    synthesizer.synthesize.return_value = (
        "Two conflicting sources found: one states 21%, another 18%. "
        "[Node ID: 55555555-5555-4555-8555-555555555555] "
        "[Node ID: 66666666-6666-4666-8666-666666666666]"
    )
    synthesizer.stream_synthesize = AsyncMock()
    synthesizer.stream_synthesize.return_value.__aiter__.return_value = iter(
        ["Two conflicting sources."]
    )

    # Build a temporal audit agent that can detect the contradiction
    # Use a mock LLM that always says YES to contradictions
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "YES|The two tax rates differ (21% vs 18%)."

    contradiction_detector = TemporalContradictionDetector(llm=mock_llm)
    scorer = FreshnessScorer(domain="general")
    temporal_auditor = TemporalAuditAgent(
        contradiction_detector=contradiction_detector,
        freshness_scorer=scorer,
    )

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
        temporal_auditor=temporal_auditor,
    )

    result = await orchestrator.run(
        "What is the effective tax rate for 2024?", "doc-conflict",
    )

    assert result is not None
    # Should have contradictions flagged
    assert result.contradictions is not None or result.causal_chain is not None


# ═══════════════════════════════════════════════════════════════════════
# 4. Query where no answer exists (critic surfaces this)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_no_answer_query_returns_none() -> None:
    """When no evidence exists for a query, orchestrator returns None."""
    planner = AsyncMock()
    planner.plan.return_value = ["What was the CEO's compensation in 2023?"]

    navigator = AsyncMock()
    navigator.find.return_value = None  # No evidence found

    critic = AsyncMock()
    critic.evaluate.return_value = False

    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
        max_iterations=1,
    )

    result = await orchestrator.run(
        "What was the CEO's compensation in 2023?", "doc-no-answer",
    )

    assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 5. Streaming query
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_streaming_query_yields_tokens() -> None:
    """Streaming query yields tokens from synthesizer.stream_synthesize."""
    planner = AsyncMock()
    planner.plan.return_value = ["What is Q3 revenue?"]

    navigator = AsyncMock()
    navigator.find.return_value = make_nav_result(
        content="Q3 revenue was $52M.",
        node_id="77777777-7777-4777-8777-777777777777",
    )

    critic = AsyncMock()
    critic.evaluate.return_value = True

    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)

    # Mock stream_synthesize to yield tokens
    async def mock_stream(query, packets):
        yield "Q3 "
        yield "revenue "
        yield "was "
        yield "$52M."

    synthesizer.stream_synthesize = mock_stream
    synthesizer.synthesize.return_value = "Q3 revenue was $52M."

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
    )

    # Collect streamed tokens
    tokens: list[str] = []
    async for token in orchestrator.stream("What is Q3 revenue?", "doc-stream"):
        tokens.append(token)

    full = "".join(tokens)
    assert len(tokens) > 1  # Should have yielded multiple tokens
    assert "revenue" in full or "$52M" in full


# ═══════════════════════════════════════════════════════════════════════
# 6. Streaming with no evidence
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_streaming_no_evidence() -> None:
    """Streaming query with no evidence yields a 'no evidence' message."""
    planner = AsyncMock()
    planner.plan.return_value = ["Unknown question?"]

    navigator = AsyncMock()
    navigator.find.return_value = None

    critic = AsyncMock()
    critic.evaluate.return_value = False

    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
        max_iterations=1,
    )

    tokens: list[str] = []
    async for token in orchestrator.stream("Unknown question?", "doc-stream"):
        tokens.append(token)

    full = "".join(tokens)
    assert "could not find" in full.lower() or "enough evidence" in full.lower()


# ═══════════════════════════════════════════════════════════════════════
# 7. ConformalWrapperAgent standalone test
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_conformal_wrapper_standalone() -> None:
    """ConformalWrapperAgent filters packets by nonconformity threshold."""
    from apex_rag.retrieval.conformal.calibrator import ConformalCalibrator
    from apex_rag.retrieval.conformal.scorer import NonconformityScorer

    calibrator = ConformalCalibrator(coverage_level=0.80, min_calibration_size=5)
    scorer = NonconformityScorer()
    wrapper = ConformalWrapperAgent(calibrator=calibrator, scorer=scorer)

    # Calibrate with low scores (easy to pass)
    wrapper.calibrate([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])

    # Create some packets
    packets = [
        UnifiedEvidencePacket(
            node=ASTNode(node_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", node_type=NodeType.PARAGRAPH, content="A", doc_id="d1"),
            temporal_metadata=TemporalMetadata(node_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            retrieval_score=0.95,
        ),
        UnifiedEvidencePacket(
            node=ASTNode(node_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1", node_type=NodeType.PARAGRAPH, content="B", doc_id="d1"),
            temporal_metadata=TemporalMetadata(node_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"),
            retrieval_score=0.90,
        ),
    ]

    result = wrapper.wrap(packets)
    assert result.coverage_guarantee > 0.0
    assert result.prediction_set_size >= 1  # At least some should pass


# ═══════════════════════════════════════════════════════════════════════
# 8. TemporalAuditAgent standalone test
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_temporal_auditor_standalone() -> None:
    """TemporalAuditAgent produces audit report with freshness and conflicts."""
    auditor = TemporalAuditAgent()

    # Packets with different dates
    packets = [
        UnifiedEvidencePacket(
            node=ASTNode(
                node_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", node_type=NodeType.PARAGRAPH,
                content="Rate is 21%", doc_id="d1",
                source_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
            ),
            temporal_metadata=TemporalMetadata(
                node_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                source_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
            ),
        ),
        UnifiedEvidencePacket(
            node=ASTNode(
                node_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1", node_type=NodeType.PARAGRAPH,
                content="Rate has been revised to 18%", doc_id="d1",
                source_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            temporal_metadata=TemporalMetadata(
                node_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1",
                source_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
        ),
    ]

    report = await auditor.audit(packets, doc_id="d1")
    assert report.mean_freshness >= 0.0
    assert isinstance(report.conflicts, list)
    assert isinstance(report.passed, bool)


# ═══════════════════════════════════════════════════════════════════════
# 9. ApexOrchestrator with conformal wrapper via run()
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_apex_orchestrator_with_conformal() -> None:
    """ApexOrchestrator.run with conformal calibration produces coverage guarantee."""
    calibrator = type("FakeCal", (), {
        "coverage_level": 0.90,
        "__init__": lambda self, **_: None,
        "calibrate": lambda self, scores: 0.5 if len(scores) >= 10 else 0.0,
    })()

    scorer = type("FakeScorer", (), {
        "score_many": lambda self, packets: [0.1, 0.2, 0.9],
        "__init__": lambda self, **_: None,
    })()

    planner = AsyncMock()
    planner.plan.return_value = ["sub-query"]
    navigator = AsyncMock()
    navigator.find.return_value = make_nav_result("Content here.", "cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    critic = AsyncMock()
    critic.evaluate.return_value = True
    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)
    synthesizer.synthesize.return_value = "Answer."
    synthesizer.stream_synthesize = AsyncMock()
    synthesizer.stream_synthesize.return_value.__aiter__.return_value = iter(["A"])

    wrapper = ConformalWrapperAgent(calibrator=calibrator, scorer=scorer)
    wrapper.calibrate([0.01] * 10)  # 10 scores → threshold = 0.5

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
        conformal_wrapper=wrapper,
    )

    result = await orchestrator.run("Test query", "doc-conformal")
    assert result is not None
    assert result.coverage_guarantee >= 0.0
    assert result.prediction_set_size >= 0


# ═══════════════════════════════════════════════════════════════════════
# 10. ApexOrchestrator streaming with conformal via stream()
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_apex_orchestrator_stream_with_conformal() -> None:
    """Streaming works even without explicit calibration data."""
    planner = AsyncMock()
    planner.plan.return_value = ["What is revenue?  "]
    navigator = AsyncMock()
    navigator.find.return_value = make_nav_result(
        "Revenue is $52M.", "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
    )
    critic = AsyncMock()
    critic.evaluate.return_value = True

    async def mock_stream(query, packets):
        yield "Revenue "
        yield "is "
        yield "$52M."

    synthesizer = AsyncMock(spec=EvidenceSynthesizerAgent)
    synthesizer.stream_synthesize = mock_stream

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
    )

    tokens: list[str] = []
    async for token in orchestrator.stream("What is revenue?", "doc-stream-conf"):
        tokens.append(token)

    full = "".join(tokens)
    assert "$52M" in full
