"""
apex_rag.observability — Production observability infrastructure.

Provides OpenTelemetry-based distributed tracing for instrumenting
agent actions across the ApexRAG pipeline.
"""

from apex_rag.observability.telemetry import TelemetryTracker, get_tracer

__all__ = [
    "TelemetryTracker",
    "get_tracer",
]
