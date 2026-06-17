import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BaseTraceEvent(BaseModel):
    """Base definition for all observability traces in ApexRAG V3."""

    trace_id: str
    trace_type: str  # reasoning, navigation, verification, temporal, graph, conformal
    event_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_sec: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)


class TraceManager:
    """
    Manages and exposes V3 traces: ReasoningTrace, NavigationTrace,
    VerificationTrace, TemporalTrace, GraphTrace, and ConformalTrace.
    Supports real-time SSE streaming via active query queues.
    """

    def __init__(self) -> None:
        self.active_listeners: Dict[str, List[asyncio.Queue]] = {}
        self.trace_start_times: Dict[str, float] = {}

    def start_trace(self, trace_id: str) -> None:
        """Starts time tracking for a specific trace session."""
        self.trace_start_times[trace_id] = time.perf_counter()

    def register_listener(self, trace_id: str) -> asyncio.Queue:
        """Registers a queue listener to yield SSE events for a trace_id."""
        queue: asyncio.Queue = asyncio.Queue()
        self.active_listeners.setdefault(trace_id, []).append(queue)
        return queue

    def unregister_listener(self, trace_id: str, queue: asyncio.Queue) -> None:
        """Removes a listener queue once streaming completes."""
        if trace_id in self.active_listeners:
            self.active_listeners[trace_id].remove(queue)
            if not self.active_listeners[trace_id]:
                del self.active_listeners[trace_id]

    def publish(self, trace_id: str, trace_type: str, event_name: str, data: dict[str, Any]) -> None:
        """Publishes a trace event to all active queue listeners for a trace_id."""
        now = time.perf_counter()
        start = self.trace_start_times.get(trace_id, now)
        elapsed = now - start

        event = BaseTraceEvent(
            trace_id=trace_id,
            trace_type=trace_type,
            event_name=event_name,
            timestamp=datetime.now(timezone.utc),
            elapsed_sec=round(elapsed, 4),
            data=data,
        )

        queues = self.active_listeners.get(trace_id, [])
        for q in queues:
            q.put_nowait(event.model_dump(mode="json"))


# Global TraceManager Singleton
trace_manager = TraceManager()
