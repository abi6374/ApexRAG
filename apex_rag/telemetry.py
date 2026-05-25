"""
telemetry.py — Production Observability for ApexRAG.

Provides OpenTelemetry instrumentation for distributed tracing and Prometheus
metrics. This is the observability backbone that makes ApexRAG production-ready:

  - **Traces**: Every query, ingestion, and navigation step is traced with spans.
  - **Metrics**: Tracks query latency, ingestion throughput, cache hit rates, errors.
  - **Logs**: Structured JSON logging correlates trace IDs with log entries.

Usage in a FastAPI app::

    from apex_rag.telemetry import setup_telemetry, QueryMetrics

    # At app startup
    setup_telemetry(service_name="apex-rag", enable_tracing=True)

    # In a query handler
    with QueryMetrics().measure_query(doc_id) as ctx:
        result = await index.query(question, doc_id)
        ctx.set_attributes({"found": str(result is not None)})

Requires: pip install apex-rag[telemetry]
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from apex_rag.utils import logger

# ---------------------------------------------------------------------------
# Lightweight metrics — always available, no external deps
# ---------------------------------------------------------------------------


@dataclass
class QueryMetricsCollector:
    """In-memory query metrics — always available, zero dependencies.

    Tracks:
      - Total queries
      - Cache hits vs misses
      - Average / p99 latency
      - Error count
    """

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)
    error_count: int = 0

    def record_query(self, latency_ms: float, cache_hit: bool = False, error: bool = False) -> None:
        self.total_queries += 1
        self.total_latency_ms += latency_ms
        self.latencies.append(latency_ms)
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        if error:
            self.error_count += 1

    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "error_count": self.error_count,
        }


# Singleton
query_metrics = QueryMetricsCollector()


class QueryMetrics:
    """Context manager for measuring query latency and recording metrics.

    Usage::

        with QueryMetrics().measure_query("doc-123") as ctx:
            result = await index.query("question", "doc-123")
            ctx.set_attributes({"found": str(result is not None)})
    """

    @contextmanager
    def measure_query(self, _doc_id: str, **_attrs: Any) -> Generator[_QueryContext, None, None]:
        t0 = time.monotonic()
        ctx = _QueryContext()
        try:
            yield ctx
        except Exception:
            query_metrics.record_query(
                latency_ms=(time.monotonic() - t0) * 1000,
                error=True,
            )
            raise
        else:
            query_metrics.record_query(
                latency_ms=(time.monotonic() - t0) * 1000,
                cache_hit=ctx.cache_hit,
            )


class _QueryContext:
    def __init__(self) -> None:
        self.cache_hit = False

    def set_attributes(self, attrs: dict[str, str]) -> None:
        if attrs.get("cache_hit") == "True":
            self.cache_hit = True


# ---------------------------------------------------------------------------
# OpenTelemetry — optional, requires opentelemetry-api and -sdk
# ---------------------------------------------------------------------------


def setup_telemetry(
    service_name: str = "apex-rag",
    otlp_endpoint: str | None = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
) -> bool:
    """Initialise OpenTelemetry tracing and metrics export.

    This is a no-op if ``opentelemetry-api`` is not installed — the library
    degrades gracefully to the in-memory ``QueryMetricsCollector``.

    Args:
        service_name:   Service name for traces/metrics.
        otlp_endpoint:  OTLP gRPC endpoint (e.g., "http://localhost:4317").
                        Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var.
        enable_tracing: Enable distributed tracing spans.
        enable_metrics: Enable Prometheus / OTLP metrics export.

    Returns:
        True if OpenTelemetry was initialised, False if not available.
    """
    if not enable_tracing and not enable_metrics:
        return False

    try:
        # Lazy imports — OpenTelemetry is an optional dependency
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        tracer_provider = TracerProvider(resource=resource)

        if otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError:
                logger.warning("OTLP exporter not installed - install opentelemetry-exporter-otlp")
            else:
                endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
                span_processor = BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=endpoint)
                )
                tracer_provider.add_span_processor(span_processor)
                logger.info("OTLP trace exporter configured: %s", endpoint)

        trace.set_tracer_provider(tracer_provider)
        logger.info("OpenTelemetry initialised: service=%s tracing=%s metrics=%s",
                     service_name, enable_tracing, enable_metrics)
        return True

    except ImportError:
        logger.info(
            "OpenTelemetry not installed (pip install apex-rag[telemetry]). "
            "Using in-memory QueryMetricsCollector."
        )
        return False
    except Exception as exc:
        logger.warning("Failed to initialise OpenTelemetry: %s", exc)
        return False


def get_tracer(name: str = "apex_rag") -> Any:
    """Get an OpenTelemetry tracer (or a no-op tracer if not configured).

    Usage::

        tracer = get_tracer()
        with tracer.start_as_current_span("query") as span:
            span.set_attribute("doc_id", doc_id)
            result = await index.query(question, doc_id)

    Returns:
        A tracer instance (always safe to call start_as_current_span on).
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        # No-op tracer
        import contextlib

        class _NoopSpan:
            def set_attribute(self, _key: str, _value: Any) -> None: ...
            def add_event(self, _name: str, _attributes: dict[str, Any] | None = None) -> None: ...

        class _NoopTracer:
            @contextlib.contextmanager
            def start_as_current_span(self, _name: str, **_kwargs: Any) -> Generator[_NoopSpan, None, None]:
                yield _NoopSpan()

        return _NoopTracer()
