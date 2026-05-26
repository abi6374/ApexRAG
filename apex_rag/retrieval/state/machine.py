from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalState(str, Enum):
    """Formal retrieval lifecycle states."""

    QUERY_RECEIVED = "QUERY_RECEIVED"
    QUERY_CLASSIFIED = "QUERY_CLASSIFIED"
    RETRIEVAL_PLAN_CREATED = "RETRIEVAL_PLAN_CREATED"
    CANDIDATES_FILTERED = "CANDIDATES_FILTERED"
    NAVIGATION_STARTED = "NAVIGATION_STARTED"
    NODE_EVALUATION = "NODE_EVALUATION"
    LEAF_VERIFICATION = "LEAF_VERIFICATION"
    EVIDENCE_AGGREGATION = "EVIDENCE_AGGREGATION"
    CRITIC_VALIDATION = "CRITIC_VALIDATION"
    ANSWER_SYNTHESIS = "ANSWER_SYNTHESIS"
    CITATION_GROUNDING = "CITATION_GROUNDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StateTransition(BaseModel):
    """Records a single transition in the lifecycle."""

    from_state: RetrievalState | None
    to_state: RetrievalState
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalStateMachine:
    """
    Manages the formal lifecycle of a query, handling metrics, tracing, and recovery.
    """

    def __init__(self, query_id: str):
        self.query_id = query_id
        self.current_state: RetrievalState = RetrievalState.QUERY_RECEIVED
        self.transitions: list[StateTransition] = [
            StateTransition(from_state=None, to_state=self.current_state)
        ]

    def transition_to(
        self, new_state: RetrievalState, metadata: dict[str, Any] | None = None
    ) -> None:
        """Transitions the machine to a new state and records the duration."""
        last_transition = self.transitions[-1]
        now = datetime.now(timezone.utc)
        duration = (now - last_transition.timestamp).total_seconds() * 1000

        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=now,
            duration_ms=duration,
            metadata=metadata or {},
        )
        self.transitions.append(transition)
        self.current_state = new_state

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """Returns the full temporal audit trail of the retrieval."""
        return [t.model_dump(mode="json") for t in self.transitions]
