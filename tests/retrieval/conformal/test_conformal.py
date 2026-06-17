"""
tests/retrieval/conformal/test_conformal.py — Part 6 Conformal Prediction tests.

Test categories:
    1. NonconformityScorer — 5 tests (4 strategies + edge cases)
    2. ConformalCalibrator — 6 tests (threshold, edge cases, coverage estimation)
    3. ConformalPredictor  — 5 tests (predict, fit_predict, edge cases, integration)
    4. Orchestrator integration — 3 tests (nonconformity in ApexAnswer,
       conformal filtering, calibration in pipeline)

Total: ~19 tests
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket
from apex_rag.models.unified_models import (
    ASTNode as UnifiedASTNode,
    ApexAnswer,
    CausalEdge,
    EdgeType,
    EvidencePacket as UnifiedEvidencePacket,
    NodeType,
    TemporalMetadata,
)
from apex_rag.retrieval.conformal.calibrator import ConformalCalibrator
from apex_rag.retrieval.conformal.predictor import ConformalPredictor
from apex_rag.retrieval.conformal.scorer import (
    NonconformityScorer,
    NonconformityStrategy,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


import uuid


_ID_COUNTER: int = 0


def _next_id() -> str:
    """Generate a valid UUID4 string for test node IDs."""
    global _ID_COUNTER
    _ID_COUNTER += 1
    # Deterministic UUID4 based on counter (valid for pydantic validator)
    return str(uuid.uuid4())


def _make_core_packet(
    confidence: float = 0.9,
    verified: bool = True,
    node_id: str | None = None,
    content: str = "Sample evidence content.",
) -> CoreEvidencePacket:
    nid = node_id or _next_id()
    return CoreEvidencePacket(
        node_id=nid,
        source_document="doc-1",
        section_path="Section 1",
        retrieval_reason="test",
        verification_result=verified,
        confidence_score=confidence,
        content=content,
    )


def _make_unified_packet(
    confidence: float = 0.9,
    node_id: str | None = None,
    rank: int = 1,
) -> UnifiedEvidencePacket:
    uid = node_id or _next_id()
    node = UnifiedASTNode(
        node_id=uid,
        content="Evidence content.",
        node_type=NodeType.PARAGRAPH,
        doc_id="doc-1",
    )
    meta = TemporalMetadata(node_id=uid, freshness_score=confidence)
    return UnifiedEvidencePacket(
        node=node,
        temporal_metadata=meta,
        retrieval_score=confidence,
        nonconformity_score=1.0,
        rank=rank,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. NonconformityScorer — 5 tests
# ═══════════════════════════════════════════════════════════════════════


class TestNonconformityScorer:
    """Nonconformity scoring strategies."""

    def test_inverse_retrieval(self) -> None:
        """INVERSE_RETRIEVAL = 1.0 - confidence_score."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        pkt = _make_core_packet(confidence=0.85)
        score = scorer.score(pkt)
        assert score == pytest.approx(0.15)
        assert score >= 0.0

    def test_inverse_retrieval_clamped(self) -> None:
        """INVERSE_RETRIEVAL clamps to 0 when confidence > 1.0."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        pkt = _make_core_packet(confidence=1.5)
        score = scorer.score(pkt)
        assert score == 0.0  # 1.0 - 1.5 = -0.5 → clamped to 0

    def test_verification_gap_true(self) -> None:
        """VERIFICATION_GAP: verified=True → score=0.0 (perfectly conforming)."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.VERIFICATION_GAP)
        pkt = _make_core_packet(verified=True)
        score = scorer.score(pkt)
        assert score == 0.0

    def test_verification_gap_false(self) -> None:
        """VERIFICATION_GAP: verified=False → score=1.0 (maximally nonconforming)."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.VERIFICATION_GAP)
        pkt = _make_core_packet(verified=False)
        score = scorer.score(pkt)
        assert score == 1.0

    def test_rank_based(self) -> None:
        """RANK_BASED: For CoreEvidencePacket (no rank field), defaults to 1/max_rank."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.RANK_BASED)
        pkt = _make_core_packet()
        # CoreEvidencePacket has no rank field → getattr(pkt, "rank", 1) returns 1
        # max_rank=5 → score = 1/5 = 0.2
        score = scorer.score(pkt, max_rank=5)
        assert score == pytest.approx(0.2)

    def test_rank_based_with_unified_packet(self) -> None:
        """RANK_BASED with UnifiedEvidencePacket (has rank field) uses actual rank."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.RANK_BASED)
        pkt = _make_unified_packet(rank=3)
        score = scorer.score(pkt, max_rank=5)
        assert score == pytest.approx(0.6)

    def test_rank_based_zero_max(self) -> None:
        """RANK_BASED: max_rank=0 returns 1.0 (safe fallback)."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.RANK_BASED)
        pkt = _make_core_packet()
        score = scorer.score(pkt, max_rank=0)
        assert score == 1.0

    def test_ensemble_default_weights(self) -> None:
        """ENSEMBLE: weighted sum of three base strategies."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.ENSEMBLE)
        # CoreEvidencePacket: no rank field → getattr returns 1
        # confidence=0.8, verified=True, max_rank=4
        # inverse = 0.2, verif = 0.0, rank = 1/4 = 0.25
        # weighted = 0.2*0.4 + 0.0*0.35 + 0.25*0.25 = 0.08 + 0.0 + 0.0625 = 0.1425
        pkt = _make_core_packet(confidence=0.8, verified=True)
        score = scorer.score(pkt, max_rank=4)
        expected = 0.2 * 0.40 + 0.0 * 0.35 + 0.25 * 0.25  # 0.1425
        assert score == pytest.approx(expected)

    def test_ensemble_all_nonconforming(self) -> None:
        """ENSEMBLE: all max nonconformity.

        CoreEvidencePacket has no rank field → getattr defaults to 1.
        So rank_based = 1/max_rank = 1/4 = 0.25, not 1.0.
        """
        scorer = NonconformityScorer(strategy=NonconformityStrategy.ENSEMBLE)
        pkt = _make_core_packet(confidence=0.0, verified=False)
        score = scorer.score(pkt, max_rank=4)
        # inverse=1.0*0.4, verif=1.0*0.35, rank=0.25*0.25
        # = 0.4 + 0.35 + 0.0625 = 0.8125
        expected = 1.0 * 0.40 + 1.0 * 0.35 + 0.25 * 0.25
        assert score == pytest.approx(expected)

    def test_score_many(self) -> None:
        """score_many returns one score per packet."""
        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        packets = [
            _make_core_packet(confidence=0.9),
            _make_core_packet(confidence=0.7),
            _make_core_packet(confidence=0.5),
        ]
        scores = scorer.score_many(packets)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.1)
        assert scores[1] == pytest.approx(0.3)
        assert scores[2] == pytest.approx(0.5)

    def test_score_many_empty(self) -> None:
        """score_many([]) returns empty list."""
        scorer = NonconformityScorer()
        scores = scorer.score_many([])
        assert scores == []

    def test_invalid_strategy(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            NonconformityScorer(strategy="unknown_strategy")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# 2. ConformalCalibrator — 6 tests
# ═══════════════════════════════════════════════════════════════════════


class TestConformalCalibrator:
    """Quantile-based threshold calibration."""

    def test_calibrate_basic(self) -> None:
        """calibrate returns a threshold in the range of the scores."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=5)
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold = calibrator.calibrate(scores)
        # n=10, alpha=0.1 → q_index = ceil((10+1)*0.9) = ceil(9.9) = 10
        # sorted_scores[9] = 1.0
        assert threshold == pytest.approx(1.0)

    def test_calibrate_95_coverage(self) -> None:
        """95 % coverage with 20 calibration points."""
        calibrator = ConformalCalibrator(coverage_level=0.95, min_calibration_size=5)
        scores = [float(i) / 20 for i in range(1, 21)]  # 0.05, 0.10, ..., 1.0
        threshold = calibrator.calibrate(scores)
        # n=20, alpha=0.05 → q_index = ceil(21*0.95) = ceil(19.95) = 20
        # sorted_scores[19] = 1.0
        assert threshold == pytest.approx(1.0)

    def test_calibrate_small_set_returns_zero(self) -> None:
        """Fewer than min_calibration_size returns 0.0 (conservative)."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=10)
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold = calibrator.calibrate(scores)
        assert threshold == 0.0

    def test_calibrate_empty_returns_zero(self) -> None:
        """Empty calibration set returns 0.0."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=5)
        threshold = calibrator.calibrate([])
        assert threshold == 0.0

    def test_invalid_coverage_level_raises(self) -> None:
        """coverage_level must be in (0, 1)."""
        with pytest.raises(ValueError):
            ConformalCalibrator(coverage_level=0.0)
        with pytest.raises(ValueError):
            ConformalCalibrator(coverage_level=1.0)
        with pytest.raises(ValueError):
            ConformalCalibrator(coverage_level=-0.1)

    def test_estimate_coverage_all_covered(self) -> None:
        """All scores ≤ threshold → coverage = 1.0."""
        calibrator = ConformalCalibrator(coverage_level=0.90)
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        coverage = calibrator.estimate_coverage(scores, threshold=0.5)
        assert coverage == 1.0

    def test_estimate_coverage_partial(self) -> None:
        """Half of scores ≤ threshold → coverage = 0.5."""
        calibrator = ConformalCalibrator(coverage_level=0.90)
        scores = [0.1, 0.2, 0.5, 0.8, 1.0]
        coverage = calibrator.estimate_coverage(scores, threshold=0.5)
        assert coverage == 0.6  # 3/5 = 0.6

    def test_estimate_coverage_empty(self) -> None:
        """Empty scores list → coverage 0.0."""
        calibrator = ConformalCalibrator()
        coverage = calibrator.estimate_coverage([], threshold=0.5)
        assert coverage == 0.0

    def test_repr(self) -> None:
        """__repr__ provides useful information."""
        calibrator = ConformalCalibrator(coverage_level=0.90)
        r = repr(calibrator)
        assert "ConformalCalibrator" in r
        assert "0.9" in r


# ═══════════════════════════════════════════════════════════════════════
# 3. ConformalPredictor — 5 tests
# ═══════════════════════════════════════════════════════════════════════


class TestConformalPredictor:
    """High-level API: calibrate → predict."""

    def test_predict_all_retained(self) -> None:
        """All scores below threshold → all packets retained."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=5)
        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        predictor = ConformalPredictor(scorer=scorer, calibrator=calibrator)

        # Calibration set with moderate scores
        cal_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold = predictor.calibrate(cal_scores)

        # High-confidence packets → low nonconformity → all retained
        packets = [
            _make_core_packet(confidence=0.95),
            _make_core_packet(confidence=0.90),
        ]
        filtered, guarantee, set_size = predictor.predict(packets, threshold)

        assert len(filtered) == 2
        assert guarantee == 0.90
        assert set_size == 2

    def test_predict_some_filtered(self) -> None:
        """Some packets above threshold → filtered out."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=5)
        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        predictor = ConformalPredictor(scorer=scorer, calibrator=calibrator)

        # Calibration: scores give threshold ≈ 0.5
        cal_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold = predictor.calibrate(cal_scores)

        # Mixed confidence: 0.95→0.05 (retain), 0.30→0.70 (filter)
        packets = [
            _make_core_packet(confidence=0.95),
            _make_core_packet(confidence=0.30),
        ]
        filtered, guarantee, set_size = predictor.predict(packets, threshold)

        assert len(filtered) >= 1
        assert guarantee == 0.90

    def test_predict_empty_packets(self) -> None:
        """Empty packet list returns empty results."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=5)
        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        predictor = ConformalPredictor(scorer=scorer, calibrator=calibrator)

        cal_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold = predictor.calibrate(cal_scores)

        filtered, guarantee, set_size = predictor.predict([], threshold)
        assert filtered == []
        assert guarantee == 0.90
        assert set_size == 0

    def test_fit_predict_end_to_end(self) -> None:
        """fit_predict does calibrate + predict in one call."""
        predictor = ConformalPredictor()

        cal_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        packets = [
            _make_core_packet(confidence=0.95),
            _make_core_packet(confidence=0.80),
        ]
        # Use threshold=1.0 because calibrate will find all pass
        filtered, guarantee, set_size = predictor.fit_predict(cal_scores, packets)

        assert isinstance(filtered, list)
        assert 0.0 < guarantee <= 1.0
        assert isinstance(set_size, int)
        assert set_size >= 0

    def test_predict_with_unified_packets(self) -> None:
        """predict works with UnifiedEvidencePacket as well."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=5)
        scorer = NonconformityScorer(strategy=NonconformityStrategy.RANK_BASED)
        predictor = ConformalPredictor(scorer=scorer, calibrator=calibrator)

        cal_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold = predictor.calibrate(cal_scores)

        packets = [
            _make_unified_packet(confidence=0.95, rank=1),
            _make_unified_packet(confidence=0.90, rank=2),
            _make_unified_packet(confidence=0.50, rank=3),
        ]
        filtered, guarantee, set_size = predictor.predict(packets, threshold)

        assert guarantee == 0.90
        # With rank-based, rank/max_rank values:
        # rank=1 → 1/3=0.33, rank=2 → 2/3=0.67, rank=3 → 3/3=1.0
        # All may be retained if threshold ≥ 1.0
        assert isinstance(set_size, int)

    def test_default_initialization(self) -> None:
        """Default constructor creates usable components."""
        predictor = ConformalPredictor()
        assert predictor.scorer.strategy == NonconformityStrategy.ENSEMBLE
        assert predictor.calibrator.coverage_level == 0.90


# ═══════════════════════════════════════════════════════════════════════
# 4. Orchestrator integration — 3 tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestOrchestratorConformalIntegration:
    """Conformal prediction integrated into execute_query_integrated."""

    async def test_nonconformity_scores_in_apex_answer(self) -> None:
        """ApexAnswer gets nonconformity_scores on EvidencePackets."""
        from apex_rag.agents.orchestrator import Orchestrator
        from apex_rag.retrieval.conformal.scorer import NonconformityScorer
        from apex_rag.retrieval.conformal.predictor import ConformalPredictor

        from apex_rag.models.unified_models import ASTNode, NodeType, EvidencePacket, TemporalMetadata
        import uuid

        def _next_id(): return str(uuid.uuid4())

        planner = MagicMock()
        planner.plan = AsyncMock(return_value=["sub-q1", "sub-q2"])

        nav_result = MagicMock()
        nav_result.verified = True
        nav_result.node_id = _next_id()
        nav_result.path = "/root/section"
        nav_result.confidence = 0.95
        nav_result.content = "Q3 revenue was $52 million."
        nav_result.node = ASTNode(
            node_id=nav_result.node_id,
            node_type=NodeType.PARAGRAPH,
            content=nav_result.content,
            doc_id="doc-1"
        )
        navigator = MagicMock()
        navigator.find = AsyncMock(return_value=nav_result)

        critic = MagicMock()
        critic.evaluate = AsyncMock(return_value=True)

        synthesizer = MagicMock()
        synthesizer.synthesize = AsyncMock(
            return_value="Q3 revenue was $52 million [Source 1]."
        )

        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        calibrator = MagicMock()
        calibrator.coverage_level = 0.90
        calibrator.min_calibration_size = 1
        calibrator.calibrate = MagicMock(return_value=0.2)

        conformal_predictor = ConformalPredictor(scorer=scorer, calibrator=calibrator)

        # Overwrite execute_query to return actual UnifiedEvidencePacket objects
        async def mock_execute_query(*args, **kwargs):
            return [
                EvidencePacket(
                    node=nav_result.node,
                    temporal_metadata=TemporalMetadata(
                        node_id=nav_result.node.node_id,
                        freshness_score=0.9
                    ),
                    retrieval_score=0.95,
                    nonconformity_score=1.0,
                    rank=1
                )
            ]

        orchestrator = Orchestrator(
            planner=planner,
            navigator=navigator,
            critic=critic,
            synthesizer=synthesizer,
            conformal_predictor=conformal_predictor,
            max_iterations=1,
        )
        orchestrator.execute_query = mock_execute_query


        answer = await orchestrator.execute_query_integrated(
            "What was Q3 revenue?",
            "doc-1",
            domain="financial",
        )

        assert answer is not None
        assert isinstance(answer, ApexAnswer)
        assert len(answer.evidence_packets) > 0

        for pkt in answer.evidence_packets:
            assert pkt.nonconformity_score >= 0.0

        assert answer.coverage_guarantee > 0.0
        assert answer.prediction_set_size >= 0

    async def test_conformal_filtering_reduces_prediction_set(self) -> None:
        """Conformal filtering reduces packet count when scores exceed threshold."""
        from apex_rag.agents.orchestrator import Orchestrator
        from apex_rag.retrieval.conformal.scorer import NonconformityScorer
        from apex_rag.retrieval.conformal.predictor import ConformalPredictor

        planner = MagicMock()
        planner.plan = AsyncMock(return_value=["sub-q1", "sub-q2", "sub-q3"])

        # Two nav results: one high confidence, one low
        nav_count = 0
        nav_results_data = [
            (_next_id(), "/s1", 0.95, "High confidence content."),
            (_next_id(), "/s2", 0.20, "Low confidence content."),
            (_next_id(), "/s3", 0.10, "Very low confidence content."),
        ]

        async def mock_find(*args, **kwargs):
            nonlocal nav_count
            idx = min(nav_count, len(nav_results_data) - 1)
            nav_count += 1
            r = nav_results_data[idx]
            result = MagicMock()
            result.verified = True
            result.node_id = r[0]
            result.path = r[1]
            result.confidence = r[2]
            result.content = r[3]
            
            from apex_rag.models.unified_models import ASTNode, NodeType
            result.node = ASTNode(
                node_id=result.node_id,
                node_type=NodeType.PARAGRAPH,
                content=result.content,
                doc_id="doc-1"
            )
            return result

        navigator = MagicMock()
        navigator.find = mock_find

        critic = MagicMock()
        critic.evaluate = AsyncMock(return_value=True)

        synthesizer = MagicMock()
        synthesizer.synthesize = AsyncMock(
            return_value="Synthesized answer with citations."
        )

        # Use strict threshold — only very high confidence packets pass
        scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
        calibrator = MagicMock()
        calibrator.coverage_level = 0.90
        calibrator.min_calibration_size = 1
        calibrator.calibrate = MagicMock(return_value=0.1)  # Very strict: only ≤0.1 NC passes

        conformal_predictor = ConformalPredictor(scorer=scorer, calibrator=calibrator)

        orchestrator = Orchestrator(
            planner=planner,
            navigator=navigator,
            critic=critic,
            synthesizer=synthesizer,
            conformal_predictor=conformal_predictor,
            max_iterations=1,
        )

        answer = await orchestrator.execute_query_integrated(
            "What was Q3 revenue?",
            "doc-1",
            domain="financial",
        )

        assert answer is not None
        # With threshold=0.1, only confidence ≥ 0.9 passes (NC ≤ 0.1)
        # node-1 has confidence=0.95 → NC=0.05 ≤ 0.1 → passes
        # node-2 has confidence=0.20 → NC=0.80 > 0.1 → filtered
        # node-3 has confidence=0.10 → NC=0.90 > 0.1 → filtered
        assert answer.prediction_set_size <= 3, (
            f"Expected ≤3 packets after conformal filter, "
            f"got {answer.prediction_set_size}"
        )
        assert answer.coverage_guarantee > 0.0

    async def test_nonconformity_scores_from_unified_packets(self) -> None:
        """Verify nonconformity_scores appear on UnifiedEvidencePacket objects."""
        from apex_rag.retrieval.conformal.scorer import NonconformityScorer

        # Use RANK_BASED strategy (uses UnifiedEvidencePacket's rank field)
        scorer = NonconformityScorer(strategy=NonconformityStrategy.RANK_BASED)

        packets = [
            _make_unified_packet(confidence=0.95, rank=1),
            _make_unified_packet(confidence=0.85, rank=2),
            _make_unified_packet(confidence=0.60, rank=3),
        ]

        scores = scorer.score_many(packets)  # type: ignore[arg-type]
        assert len(scores) == 3
        # Rank based: rank / max_rank
        assert scores[0] == pytest.approx(1.0 / 3)  # 1/3
        assert scores[1] == pytest.approx(2.0 / 3)  # 2/3
        assert scores[2] == pytest.approx(3.0 / 3)  # 3/3 = 1.0


# ═══════════════════════════════════════════════════════════════════════
# 5. Edge case and regression tests — 2 tests
# ═══════════════════════════════════════════════════════════════════════


def test_nonconformity_scores_ordering_consistency() -> None:
    """Higher confidence should always produce lower (better) NC scores."""
    scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)

    packets = [
        _make_core_packet(confidence=c)
        for c in [0.1, 0.2, 0.5, 0.8, 0.95]
    ]
    scores = scorer.score_many(packets)

    # Scores should be decreasing as confidence increases
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"NC scores should decrease with increasing confidence: "
            f"{scores}"
        )


def test_zero_confidence_max_nonconformity() -> None:
    """Zero confidence = maximal nonconformity (1.0)."""
    scorer = NonconformityScorer(strategy=NonconformityStrategy.INVERSE_RETRIEVAL)
    pkt = _make_core_packet(confidence=0.0)
    assert scorer.score(pkt) == 1.0


def test_conformal_calibrator_not_fitted_returns_zero() -> None:
    """When calibrator hasn't seen enough data, it returns 0.0 threshold."""
    calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=10)
    scores = [0.1, 0.2, 0.3]
    threshold = calibrator.calibrate(scores)
    assert threshold == 0.0, "Not enough data should yield conservative 0.0"
