import time
from typing import Any


class NavigationBudget:
    """
    Enforces strict structural navigation budget limits to prevent infinite recursion,
    minimize LLM overhead, and guarantee predictable latency.
    """

    def __init__(
        self,
        max_depth: int = 10,
        max_backtracks: int = 15,
        max_candidates: int = 5,
        max_runtime_ms: float = 3000.0,
        max_llm_calls: int = 20,
    ) -> None:
        self.max_depth = max_depth
        self.max_backtracks = max_backtracks
        self.max_candidates = max_candidates
        self.max_runtime_ms = max_runtime_ms
        self.max_llm_calls = max_llm_calls

        # Active tracking state
        self.current_depth: int = 0
        self.backtracks: int = 0
        self.llm_calls: int = 0
        self.start_time: float = time.perf_counter()

    def check_valid(self) -> bool:
        """
        Check if the navigation budget is still valid. Returns False if exceeded.
        """
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        if elapsed_ms > self.max_runtime_ms:
            return False
        if self.llm_calls > self.max_llm_calls:
            return False
        if self.backtracks > self.max_backtracks:
            return False
        return not self.current_depth > self.max_depth

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def record_backtrack(self) -> None:
        self.backtracks += 1

    def record_depth(self, depth: int) -> None:
        self.current_depth = max(self.current_depth, depth)

    def to_dict(self) -> dict[str, Any]:
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        return {
            "max_depth": self.max_depth,
            "max_backtracks": self.max_backtracks,
            "max_candidates": self.max_candidates,
            "max_runtime_ms": self.max_runtime_ms,
            "max_llm_calls": self.max_llm_calls,
            "current_depth": self.current_depth,
            "backtracks": self.backtracks,
            "llm_calls": self.llm_calls,
            "elapsed_ms": round(elapsed_ms, 2),
        }
