from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.trace import Tracer

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

if HAS_OPENTELEMETRY:
    _tracer = trace.get_tracer("apex_rag.enterprise")
else:
    # Define dummy tracer and span classes to avoid runtime crashes when extras are not installed
    class DummySpan:
        def __enter__(self) -> "DummySpan":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def set_attributes(self, attributes: dict[str, Any]) -> None:
            pass

        def end(self) -> None:
            pass

    class DummyTracer:
        def start_span(self, name: str, *args: Any, **kwargs: Any) -> DummySpan:
            return DummySpan()

    _tracer = DummyTracer()  # type: ignore


def get_tracer(_name: str = "apex_rag.enterprise") -> Any:
    """
    Returns the OpenTelemetry tracer for the enterprise module.
    """
    return _tracer


class TelemetryTracker:
    """
    Utility for cleanly creating tracing spans around Agent actions.
    """

    @staticmethod
    def start_span(name: str, attributes: dict[str, Any] | None = None) -> Any:
        """Starts an OpenTelemetry span."""
        span = _tracer.start_span(name)
        if attributes and hasattr(span, "set_attributes"):
            span.set_attributes(attributes)
        return span
