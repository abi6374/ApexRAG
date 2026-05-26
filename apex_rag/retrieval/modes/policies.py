from enum import Enum

from pydantic import BaseModel


class RetrievalPolicy(BaseModel):
    max_depth: int
    verifier_strictness: float
    allow_backtracking: bool
    use_hybrid_search: bool


class RetrievalMode(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    CODE = "code"


def get_policy_for_mode(mode: RetrievalMode) -> RetrievalPolicy:
    if mode == RetrievalMode.FACTUAL:
        return RetrievalPolicy(
            max_depth=3, verifier_strictness=0.9, allow_backtracking=False, use_hybrid_search=False
        )
    elif mode == RetrievalMode.ANALYTICAL:
        return RetrievalPolicy(
            max_depth=5, verifier_strictness=0.7, allow_backtracking=True, use_hybrid_search=True
        )
    elif mode == RetrievalMode.LEGAL:
        return RetrievalPolicy(
            max_depth=7, verifier_strictness=0.95, allow_backtracking=True, use_hybrid_search=False
        )
    elif mode == RetrievalMode.FINANCIAL:
        return RetrievalPolicy(
            max_depth=5, verifier_strictness=0.95, allow_backtracking=True, use_hybrid_search=True
        )
    elif mode == RetrievalMode.CODE:
        return RetrievalPolicy(
            max_depth=10, verifier_strictness=0.8, allow_backtracking=True, use_hybrid_search=True
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
