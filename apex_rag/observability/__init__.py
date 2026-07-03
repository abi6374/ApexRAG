"""
Observability module for ApexRAG.

Provides centralized access to metrics, tracing, logging, and accuracy tracking.

Usage:
    from apex_rag.observability import metrics_service, accuracy_tracker
    metrics_service.record_retrieval_latency(ms=145.2)
    accuracy_tracker.record_query(query="...", doc_id="...", precision=0.95, ...)
"""

from apex_rag.observability.accuracy_tracker import AccuracyTracker, accuracy_tracker
from apex_rag.observability.metrics_service import MetricsService, metrics_service
from apex_rag.observability.trace_manager import trace_manager

__all__ = [
    "AccuracyTracker",
    "accuracy_tracker",
    "MetricsService",
    "metrics_service",
    "trace_manager",
]
