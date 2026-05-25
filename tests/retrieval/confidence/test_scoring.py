import pytest

from apex_rag.retrieval.confidence.scoring import ConfidenceEngine


def test_calculate_score_perfect():
    engine = ConfidenceEngine()
    score = engine.calculate_score(retrieval_confidence=1.0, verifier_score=1.0, graph_depth=0)
    assert score == 1.0

def test_calculate_score_with_depth_penalty():
    engine = ConfidenceEngine()
    # base_score = 1.0 * 0.6 + 1.0 * 0.4 = 1.0
    # penalty = 2 * 0.05 = 0.1
    # score = 0.9
    score = engine.calculate_score(retrieval_confidence=1.0, verifier_score=1.0, graph_depth=2)
    assert pytest.approx(score) == 0.9

def test_calculate_score_clamped_bottom():
    engine = ConfidenceEngine()
    # base_score = 0.1 * 0.6 + 0.1 * 0.4 = 0.1
    # penalty = 5 * 0.05 = 0.25
    # score = -0.15 -> clamped to 0.0
    score = engine.calculate_score(retrieval_confidence=0.1, verifier_score=0.1, graph_depth=5)
    assert score == 0.0

def test_calculate_score_clamped_top():
    engine = ConfidenceEngine()
    # If inputs exceed 1.0
    score = engine.calculate_score(retrieval_confidence=2.0, verifier_score=2.0, graph_depth=0)
    assert score == 1.0

def test_calculate_score_weights():
    engine = ConfidenceEngine()
    # base_score = 0.5 * 0.6 + 0.8 * 0.4 = 0.3 + 0.32 = 0.62
    score = engine.calculate_score(retrieval_confidence=0.8, verifier_score=0.5, graph_depth=0)
    assert pytest.approx(score) == 0.62
