from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer


def get_tracer(name: str = "apex_rag.enterprise") -> Tracer:
    """Return an OpenTelemetry tracer for the given instrumentation name.

    Args:
        name: The instrumentation scope name (default: ``apex_rag.enterprise``).

    Returns:
        An OpenTelemetry :class:`Tracer` instance.
    """
    return trace.get_tracer(name)


class TelemetryTracker:
    """Utility for cleanly creating tracing spans around Agent actions."""

    @staticmethod
    def start_span(name: str, attributes: dict[str, Any] | None = None) -> trace.Span:
        """Starts an OpenTelemetry span."""
        tracer = get_tracer("apex_rag.enterprise")
        span = tracer.start_span(name)
        if attributes:
            span.set_attributes(attributes)
        return span
