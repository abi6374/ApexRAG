"""
observability/metrics_service.py — Production metrics and monitoring.

Provides:
  - Structured JSON logging with request_id, tenant_id, user_id, query_id
  - Prometheus-compatible metrics endpoint (retrieval latency, cache hits, etc.)
  - In-process metrics aggregation

Metrics tracked:
  - retrieval_latency  (histogram)
  - planner_latency    (histogram)
  - navigator_latency  (histogram)
  - verifier_latency   (histogram)
  - critic_latency     (histogram)
  - cache_hits         (counter)
  - cache_misses       (counter)
  - tenant_queries     (counter, labelled by tenant)
  - LLM_calls          (counter)

Usage:
    from apex_rag.observability.metrics_service import metrics_service

    # Record a metric
    metrics_service.record_retrieval_latency(ms=145.2)
    metrics_service.record_cache_hit()
    metrics_service.increment_llm_calls()

    # Get Prometheus output
    prom_output = metrics_service.get_prometheus_metrics()

    # Get structured log entry
    log_entry = metrics_service.make_log_entry(
        request_id="req-123", tenant_id="tenant-a",
        user_id="user-1", query_id="q-456",
    )
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from apex_rag.observability._stats import histogram_summary

logger = logging.getLogger("apex_rag.observability.metrics_service")


class MetricsService:
    """In-process metrics aggregation for ApexRAG.

    Thread-safe, async-compatible, and designed for Prometheus scraping.

    This initial implementation uses in-memory counters and histograms.
    In production, these would be replaced with actual Prometheus metrics
    using the ``prometheus_client`` library.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # ── Counters ─────────────────────────────────────────────────
        self._retrieval_latencies: list[float] = []
        self._planner_latencies: list[float] = []
        self._navigator_latencies: list[float] = []
        self._verifier_latencies: list[float] = []
        self._critic_latencies: list[float] = []
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._tenant_query_counts: dict[str, int] = defaultdict(int)
        self._llm_calls: int = 0
        self._total_queries: int = 0
        self._start_time: float = time.monotonic()

    # ── Query Lifecycle ──────────────────────────────────────────────

    def record_completed_query(self) -> None:
        """Record that one top-level query has completed.

        Use this instead of relying on ``record_retrieval_latency`` to
        increment the query count, because a single query may invoke
        multiple retrieval passes.
        """
        with self._lock:
            self._total_queries += 1

    # ── Latency Recording ────────────────────────────────────────────

    def record_retrieval_latency(self, ms: float) -> None:
        with self._lock:
            self._retrieval_latencies.append(ms)

    def record_planner_latency(self, ms: float) -> None:
        with self._lock:
            self._planner_latencies.append(ms)

    def record_navigator_latency(self, ms: float) -> None:
        with self._lock:
            self._navigator_latencies.append(ms)

    def record_verifier_latency(self, ms: float) -> None:
        with self._lock:
            self._verifier_latencies.append(ms)

    def record_critic_latency(self, ms: float) -> None:
        with self._lock:
            self._critic_latencies.append(ms)

    # ── Cache Recording ──────────────────────────────────────────────

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    # ── Tenant Recording ─────────────────────────────────────────────

    def record_tenant_query(self, tenant_id: str) -> None:
        with self._lock:
            self._tenant_query_counts[tenant_id] += 1

    # ── LLM Recording ────────────────────────────────────────────────

    def increment_llm_calls(self, count: int = 1) -> None:
        with self._lock:
            self._llm_calls += count

    # ── Statistics ───────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics for all metrics."""
        with self._lock:
            return {
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "total_queries": self._total_queries,
                "llm_calls": self._llm_calls,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": round(
                    self._cache_hits / (self._cache_hits + self._cache_misses)
                    if (self._cache_hits + self._cache_misses) > 0
                    else 0.0,
                    4,
                ),
                "retrieval_latency": histogram_summary(self._retrieval_latencies),
                "planner_latency": histogram_summary(self._planner_latencies),
                "navigator_latency": histogram_summary(self._navigator_latencies),
                "verifier_latency": histogram_summary(self._verifier_latencies),
                "critic_latency": histogram_summary(self._critic_latencies),
                "tenant_queries": dict(self._tenant_query_counts),
            }

    # ── Prometheus Output (text format) ──────────────────────────────

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output.

        Returns:
            A string in Prometheus text exposition format.
        """
        stats = self.get_stats()
        lines: list[str] = []

        # Metadata
        lines.append("# HELP apex_rag_uptime_seconds Uptime of the ApexRAG service")
        lines.append("# TYPE apex_rag_uptime_seconds gauge")
        lines.append(f"apex_rag_uptime_seconds {stats['uptime_seconds']}")

        lines.append("# HELP apex_rag_total_queries Total queries processed")
        lines.append("# TYPE apex_rag_total_queries counter")
        lines.append(f"apex_rag_total_queries {stats['total_queries']}")

        lines.append("# HELP apex_rag_llm_calls Total LLM calls made")
        lines.append("# TYPE apex_rag_llm_calls counter")
        lines.append(f"apex_rag_llm_calls {stats['llm_calls']}")

        lines.append("# HELP apex_rag_cache_hits Total cache hits")
        lines.append("# TYPE apex_rag_cache_hits counter")
        lines.append(f"apex_rag_cache_hits {stats['cache_hits']}")

        lines.append("# HELP apex_rag_cache_misses Total cache misses")
        lines.append("# TYPE apex_rag_cache_misses counter")
        lines.append(f"apex_rag_cache_misses {stats['cache_misses']}")

        lines.append("# HELP apex_rag_cache_hit_rate Cache hit rate (0-1)")
        lines.append("# TYPE apex_rag_cache_hit_rate gauge")
        lines.append(f"apex_rag_cache_hit_rate {stats['cache_hit_rate']}")

        # Latency histograms — standard Prometheus _bucket / _count / _sum format
        PROMETHEUS_BUCKETS = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, float("inf")]

        for name, label in [
            ("retrieval_latency", "retrieval"),
            ("planner_latency", "planner"),
            ("navigator_latency", "navigator"),
            ("verifier_latency", "verifier"),
            ("critic_latency", "critic"),
        ]:
            h = stats[name]
            metric_name = f"apex_rag_{name}_ms"
            lines.append(f"# HELP {metric_name} Latency for {label} stage (ms)")
            lines.append(f"# TYPE {metric_name} histogram")
            lines.append(f"{metric_name}_count {h['count']}")
            lines.append(f"{metric_name}_sum {h['sum']}")

            # Buckets — use the actual values list for accurate bucketing
            raw_values: list[float] = []
            with self._lock:
                if name == "retrieval_latency":
                    raw_values = list(self._retrieval_latencies)
                elif name == "planner_latency":
                    raw_values = list(self._planner_latencies)
                elif name == "navigator_latency":
                    raw_values = list(self._navigator_latencies)
                elif name == "verifier_latency":
                    raw_values = list(self._verifier_latencies)
                elif name == "critic_latency":
                    raw_values = list(self._critic_latencies)

            for bucket in PROMETHEUS_BUCKETS:
                if bucket == float("inf"):
                    lines.append(f"{metric_name}_bucket{{le=\"+Inf\"}} {h['count']}")
                else:
                    count_in_bucket = sum(1 for v in raw_values if v <= bucket)
                    lines.append(f"{metric_name}_bucket{{le=\"{bucket}\"}} {count_in_bucket}")

        # Tenant query counts
        lines.append("# HELP apex_rag_tenant_queries_total Total queries per tenant")
        lines.append("# TYPE apex_rag_tenant_queries_total counter")
        for tenant_id, count in stats["tenant_queries"].items():
            lines.append(f'apex_rag_tenant_queries_total{{tenant="{tenant_id}"}} {count}')

        return "\n".join(lines) + "\n"

    # ── Structured JSON Logging ──────────────────────────────────────

    @staticmethod
    def make_log_entry(
        *,
        request_id: str,
        tenant_id: str,
        user_id: str,
        query_id: str,
        event: str = "query",
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Create a structured JSON log entry.

        Args:
            request_id:  Unique request identifier.
            tenant_id:   Tenant ID for multi-tenant isolation.
            user_id:     User making the request.
            query_id:    Query identifier.
            event:       Event type (default: "query").
            extra:       Additional key-value pairs to include.

        Returns:
            A JSON string ready for log output.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "logger": "apex_rag.observability",
            "request_id": request_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "query_id": query_id,
            "event": event,
        }
        if extra:
            entry.update(extra)
        return json.dumps(entry, ensure_ascii=False, default=str)


# Global singleton
metrics_service = MetricsService()
