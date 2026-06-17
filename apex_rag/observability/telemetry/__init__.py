"""
apex_rag.observability.telemetry — OpenTelemetry tracing utilities.

Provides the :class:`TelemetryTracker` for instrumenting agent actions
and the :func:`get_tracer` function for creating named tracers.
"""

from apex_rag.observability.telemetry.tracer import TelemetryTracker, get_tracer

__all__ = [
    "TelemetryTracker",
    "get_tracer",
]
