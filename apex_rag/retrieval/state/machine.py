import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalState(str, Enum):
    """Formal retrieval lifecycle states for ApexRAG V3."""

    QUERY_RECEIVED = "QUERY_RECEIVED"
    QUERY_CLASSIFIED = "QUERY_CLASSIFIED"
    PLAN_GENERATED = "PLAN_GENERATED"
    FILTERING_COMPLETE = "FILTERING_COMPLETE"
    NAVIGATION_RUNNING = "NAVIGATION_RUNNING"
    VERIFICATION_RUNNING = "VERIFICATION_RUNNING"
    GRAPH_REASONING = "GRAPH_REASONING"
    TEMPORAL_AUDIT = "TEMPORAL_AUDIT"
    CRITIC_REVIEW = "CRITIC_REVIEW"
    CONFORMAL_FILTERING = "CONFORMAL_FILTERING"
    SYNTHESIS = "SYNTHESIS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StateTransition(BaseModel):
    """Records a single state transition in the lifecycle."""

    from_state: RetrievalState | None
    to_state: RetrievalState
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    retries: int = 0
    is_rollback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateTimeoutError(Exception):
    """Raised when a state execution exceeds its defined timeout limit."""
    pass


class RetrievalStateMachine:
    """
    Manages the formal retrieval lifecycle of a query, handling metrics,
    tracing, timeouts, retries, and rollback capabilities.
    """

    def __init__(self, query_id: str):
        self.query_id = query_id
        self.current_state: RetrievalState = RetrievalState.QUERY_RECEIVED
        self.transitions: list[StateTransition] = [
            StateTransition(from_state=None, to_state=self.current_state)
        ]
        self.retry_counts: dict[RetrievalState, int] = {}
        self.state_start_times: dict[RetrievalState, float] = {
            self.current_state: time.perf_counter()
        }

    def transition_to(
        self,
        new_state: RetrievalState,
        metadata: dict[str, Any] | None = None,
        is_rollback: bool = False,
        timeout_sec: float | None = None,
    ) -> None:
        """
        Transition the state machine to a new state.
        Validates if the current state has timed out before completing the transition.
        """
        now_time = time.perf_counter()
        last_state = self.current_state
        start_time = self.state_start_times.get(last_state, now_time)
        duration = (now_time - start_time) * 1000.0

        if timeout_sec is not None and (duration / 1000.0) > timeout_sec:
            raise StateTimeoutError(
                f"State {last_state.value} timed out after {duration:.2f}ms (limit: {timeout_sec}s)"
            )

        now_utc = datetime.now(timezone.utc)
        retries = self.retry_counts.get(new_state, 0)

        transition = StateTransition(
            from_state=last_state,
            to_state=new_state,
            timestamp=now_utc,
            duration_ms=duration,
            retries=retries,
            is_rollback=is_rollback,
            metadata=metadata or {},
        )
        self.transitions.append(transition)
        self.current_state = new_state
        self.state_start_times[new_state] = now_time

    def record_retry(self, state: RetrievalState) -> None:
        """Increment and record the retry count for a state."""
        self.retry_counts[state] = self.retry_counts.get(state, 0) + 1

    def rollback_to(self, target_state: RetrievalState, reason: str) -> None:
        """Rollback to a previous state in the workflow."""
        self.transition_to(
            target_state,
            metadata={"rollback_reason": reason},
            is_rollback=True,
        )

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """Returns the full temporal audit trail of transitions."""
        return [t.model_dump(mode="json") for t in self.transitions]
