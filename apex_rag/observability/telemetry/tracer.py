from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer

_tracer = trace.get_tracer("apex_rag.enterprise")


def get_tracer(_name: str = "apex_rag.enterprise") -> Tracer:
    """
    Returns the OpenTelemetry tracer for the enterprise module.
    """
    return _tracer


class TelemetryTracker:
    """
    Utility for cleanly creating tracing spans around Agent actions.
    """

    @staticmethod
    def start_span(name: str, attributes: dict[str, Any] | None = None) -> trace.Span:
        """Starts an OpenTelemetry span."""
        span = _tracer.start_span(name)
        if attributes:
            span.set_attributes(attributes)
        return span
