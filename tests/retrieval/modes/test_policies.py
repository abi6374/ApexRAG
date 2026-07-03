import pytest

from apex_rag.retrieval.modes.policies import RetrievalMode, RetrievalPolicy, get_policy_for_mode


def test_get_policy_for_mode_factual():
    policy = get_policy_for_mode(RetrievalMode.FACTUAL)
    assert isinstance(policy, RetrievalPolicy)
    assert policy.max_depth == 3
    assert policy.verifier_strictness == 0.9
    assert policy.allow_backtracking is False
    assert policy.use_hybrid_search is False


def test_get_policy_for_mode_analytical():
    policy = get_policy_for_mode(RetrievalMode.ANALYTICAL)
    assert policy.max_depth == 5
    assert policy.verifier_strictness == 0.7
    assert policy.allow_backtracking is True
    assert policy.use_hybrid_search is True


def test_get_policy_for_mode_legal():
    policy = get_policy_for_mode(RetrievalMode.LEGAL)
    assert policy.max_depth == 7
    assert policy.verifier_strictness == 0.95
    assert policy.allow_backtracking is True
    assert policy.use_hybrid_search is False


def test_get_policy_for_mode_financial():
    policy = get_policy_for_mode(RetrievalMode.FINANCIAL)
    assert policy.max_depth == 5
    assert policy.verifier_strictness == 0.95
    assert policy.allow_backtracking is True
    assert policy.use_hybrid_search is True


def test_get_policy_for_mode_code():
    policy = get_policy_for_mode(RetrievalMode.CODE)
    assert policy.max_depth == 10
    assert policy.verifier_strictness == 0.8
    assert policy.allow_backtracking is True
    assert policy.use_hybrid_search is True


def test_get_policy_for_invalid_mode():
    # If mode is not a RetrievalMode, it won't match any if branch
    with pytest.raises(ValueError):
        get_policy_for_mode("invalid")  # type: ignore
