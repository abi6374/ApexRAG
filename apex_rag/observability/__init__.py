"""
Observability module for ApexRAG.

Provides centralized access to metrics, tracing, and logging.

Usage:
    from apex_rag.observability import metrics_service
    metrics_service.record_retrieval_latency(ms=145.2)
"""

from apex_rag.observability.metrics_service import MetricsService, metrics_service
from apex_rag.observability.trace_manager import trace_manager

__all__ = [
    "MetricsService",
    "metrics_service",
    "trace_manager",
]
