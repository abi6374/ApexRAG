"""
observability/accuracy_tracker.py — Per-query accuracy metrics for ApexRAG.

Tracks precision, recall, F1, and other retrieval quality metrics for every query.

Metrics computed per query:
  - **Precision**:   verified_packets / total_packets_retrieved
                     (How many retrieved evidence packets passed LLM verification)
  - **Recall**:      resolved_subqueries / total_subqueries
                     (How many planned sub-queries were successfully answered)
  - **F1 Score**:    Harmonic mean of precision and recall
  - **Hit Rate**:    1.0 if any evidence was found, 0.0 otherwise
  - **Critic Pass Rate**:    critic_passes / total_iterations
  - **Navigation Efficiency**: nodes_visited / packets_retrieved
  - **LLM Call Efficiency**:  sub_queries / llm_calls_per_query
  - **Coverage**:    conformal prediction coverage guarantee (already in ApexAnswer)

All metrics are stored per-query in-memory and can be exported as:
  - Aggregate stats (mean, p50, p95, p99 across queries)
  - Prometheus-compatible format
  - Structured JSON log entries

Usage:
    from apex_rag.observability.accuracy_tracker import accuracy_tracker

    # Record accuracy for a completed query
    accuracy_tracker.record_query(
        query="What is Q3 revenue?",
        doc_id="doc-123",
        precision=0.85,
        recall=1.0,
        f1_score=0.92,
        hit=True,
        coverage_guarantee=0.90,
        critic_pass_rate=1.0,
        llm_calls=8,
        latency_ms=1450.0,
        tenant_id="default",
    )

    # Get aggregate stats
    stats = accuracy_tracker.get_aggregate_stats()
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from apex_rag.observability._stats import percentile_summary, zero_percentile_summary
from apex_rag.utils import logger

# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════


class QueryAccuracyRecord(BaseModel):
    """Per-query accuracy metrics record.

    Captured for every query that goes through the orchestrator pipeline.
    """

    query: str
    doc_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Core accuracy metrics
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hit: bool = False  # Whether any evidence was found

    # Retrieval quality
    coverage_guarantee: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_set_size: int = 0
    total_packets_retrieved: int = 0
    verified_packets_count: int = 0
    total_subqueries: int = 0
    resolved_subqueries: int = 0

    # Process quality
    critic_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    total_iterations: int = 0
    critic_passes: int = 0
    nodes_visited: int = 0
    llm_calls: int = 0
    backtracks: int = 0

    # Performance
    latency_ms: float = 0.0
    tenant_id: str = "default"

    model_config = {"frozen": False}


# ═══════════════════════════════════════════════════════════════════════
# Accuracy Tracker
# ═══════════════════════════════════════════════════════════════════════


class AccuracyTracker:
    """Thread-safe, per-query accuracy metrics collector.

    Stores every query's accuracy record and provides aggregate statistics
    across all queries for monitoring and observability.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[QueryAccuracyRecord] = []
        self._tenant_counts: dict[str, int] = defaultdict(int)
        self._start_time: float = time.monotonic()

    # ── Record a single query ─────────────────────────────────────────

    def record_query(
        self,
        *,
        query: str,
        doc_id: str,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        hit: bool = False,
        coverage_guarantee: float = 0.0,
        prediction_set_size: int = 0,
        total_packets_retrieved: int = 0,
        verified_packets_count: int = 0,
        total_subqueries: int = 0,
        resolved_subqueries: int = 0,
        critic_pass_rate: float = 1.0,
        total_iterations: int = 0,
        critic_passes: int = 0,
        nodes_visited: int = 0,
        llm_calls: int = 0,
        backtracks: int = 0,
        latency_ms: float = 0.0,
        tenant_id: str = "default",
    ) -> QueryAccuracyRecord:
        """Record accuracy metrics for a single query.

        Args:
            query:                    The original user query.
            doc_id:                   Target document ID.
            precision:                verified_packets / total_packets_retrieved.
            recall:                   resolved_subqueries / total_subqueries.
            f1_score:                 Harmonic mean of precision and recall.
            hit:                      Whether any evidence was found.
            coverage_guarantee:       Conformal prediction coverage.
            prediction_set_size:      Size of the conformal prediction set.
            total_packets_retrieved:  All evidence packets retrieved before filtering.
            verified_packets_count:   Packets that passed LLM verification.
            total_subqueries:         Sub-queries the planner decomposed the query into.
            resolved_subqueries:      Sub-queries that were successfully answered.
            critic_pass_rate:         Fraction of iterations where critic approved.
            total_iterations:         Total planner→navigator→critic iterations.
            critic_passes:            Number of iterations that passed the critic.
            nodes_visited:            Total AST nodes visited during navigation.
            llm_calls:                Total LLM calls made during the query.
            backtracks:               Number of navigation backtracks.
            latency_ms:               End-to-end query latency.
            tenant_id:                Tenant ID for multi-tenant isolation.

        Returns:
            The recorded :class:`QueryAccuracyRecord`.
        """
        record = QueryAccuracyRecord(
            query=query[:200] if query else "",
            doc_id=doc_id,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1_score, 4),
            hit=hit,
            coverage_guarantee=round(coverage_guarantee, 4),
            prediction_set_size=prediction_set_size,
            total_packets_retrieved=total_packets_retrieved,
            verified_packets_count=verified_packets_count,
            total_subqueries=total_subqueries,
            resolved_subqueries=resolved_subqueries,
            critic_pass_rate=round(critic_pass_rate, 4),
            total_iterations=total_iterations,
            critic_passes=critic_passes,
            nodes_visited=nodes_visited,
            llm_calls=llm_calls,
            backtracks=backtracks,
            latency_ms=round(latency_ms, 1),
            tenant_id=tenant_id,
        )

        with self._lock:
            self._records.append(record)
            self._tenant_counts[tenant_id] += 1

        logger.debug(
            "[ACCURACY] query=%s precision=%.3f recall=%.3f f1=%.3f hit=%s",
            query[:50],
            record.precision,
            record.recall,
            record.f1_score,
            record.hit,
        )

        return record

    # ── Aggregate Statistics ──────────────────────────────────────────

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Return aggregate accuracy statistics across all recorded queries.

        Returns:
            A dict with aggregate precision, recall, F1, latency, etc.
        """
        with self._lock:
            if not self._records:
                return self._empty_stats()

            precisions = [r.precision for r in self._records]
            recalls = [r.recall for r in self._records]
            f1s = [r.f1_score for r in self._records]
            latencies = [r.latency_ms for r in self._records]
            hits = [1.0 if r.hit else 0.0 for r in self._records]
            coverages = [r.coverage_guarantee for r in self._records]
            pred_sizes = [r.prediction_set_size for r in self._records]
            llm_calls = [r.llm_calls for r in self._records]
            critic_rates = [r.critic_pass_rate for r in self._records]

            n = len(self._records)

            return {
                "total_queries": n,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "hit_rate": round(sum(hits) / n, 4) if n > 0 else 0.0,
                "precision": self._percentile_summary(precisions),
                "recall": self._percentile_summary(recalls),
                "f1_score": self._percentile_summary(f1s),
                "latency_ms": self._percentile_summary(latencies),
                "coverage_guarantee": self._percentile_summary(coverages),
                "prediction_set_size": {
                    "mean": round(sum(pred_sizes) / n, 2) if n > 0 else 0.0,
                    "min": min(pred_sizes) if pred_sizes else 0,
                    "max": max(pred_sizes) if pred_sizes else 0,
                },
                "llm_calls_per_query": self._percentile_summary(llm_calls),
                "critic_pass_rate": self._percentile_summary(critic_rates),
                "tenant_queries": dict(self._tenant_counts),
            }

    def get_recent_queries(
        self, limit: int = 10, min_precision: float | None = None
    ) -> list[dict[str, Any]]:
        """Return the most recent query accuracy records, optionally filtered.

        Args:
            limit:           Max number of records to return.
            min_precision:   Optional minimum precision filter.

        Returns:
            List of query accuracy record dicts.
        """
        with self._lock:
            records = list(self._records)

        # Reverse for most recent first
        records.reverse()

        filtered = [
            r.model_dump(mode="json")
            for r in records
            if min_precision is None or r.precision >= min_precision
        ]

        return filtered[:limit]

    def get_structured_log_entry(
        self, record: QueryAccuracyRecord, extra: dict[str, Any] | None = None
    ) -> str:
        """Create a structured JSON log entry for a query accuracy record.

        Args:
            record: The query accuracy record to log.
            extra:  Optional additional fields to merge into the log entry.

        Returns:
            A JSON string ready for log output.
        """
        entry: dict[str, Any] = {
            "timestamp": record.timestamp.isoformat(),
            "level": "INFO",
            "logger": "apex_rag.observability.accuracy",
            "event": "query_accuracy",
            "query": record.query[:100],
            "doc_id": record.doc_id,
            "tenant_id": record.tenant_id,
            "precision": record.precision,
            "recall": record.recall,
            "f1_score": record.f1_score,
            "hit": record.hit,
            "coverage_guarantee": record.coverage_guarantee,
            "prediction_set_size": record.prediction_set_size,
            "total_packets": record.total_packets_retrieved,
            "verified_packets": record.verified_packets_count,
            "total_subqueries": record.total_subqueries,
            "resolved_subqueries": record.resolved_subqueries,
            "critic_pass_rate": record.critic_pass_rate,
            "total_iterations": record.total_iterations,
            "nodes_visited": record.nodes_visited,
            "llm_calls": record.llm_calls,
            "backtracks": record.backtracks,
            "latency_ms": record.latency_ms,
        }
        if extra:
            entry.update(extra)
        return json.dumps(entry, ensure_ascii=False, default=str)

    # ── Internal helpers ──────────────────────────────────────────────

    def _empty_stats(self) -> dict[str, Any]:
        return {
            "total_queries": 0,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "hit_rate": 0.0,
            "precision": self._zero_summary(),
            "recall": self._zero_summary(),
            "f1_score": self._zero_summary(),
            "latency_ms": self._zero_summary(),
            "coverage_guarantee": self._zero_summary(),
            "prediction_set_size": {"mean": 0.0, "min": 0, "max": 0},
            "llm_calls_per_query": self._zero_summary(),
            "critic_pass_rate": self._zero_summary(),
            "tenant_queries": {},
        }

    @staticmethod
    def _zero_summary() -> dict[str, float]:
        return zero_percentile_summary()

    @staticmethod
    def _percentile_summary(values: list[float]) -> dict[str, float]:
        """Compute percentile summary from a list of values."""
        return percentile_summary(values)

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible accuracy metrics output.

        Returns:
            A string in Prometheus text exposition format.
        """
        stats = self.get_aggregate_stats()
        lines: list[str] = []

        # Hit rate
        lines.append("# HELP apex_rag_accuracy_hit_rate Query hit rate (0-1)")
        lines.append("# TYPE apex_rag_accuracy_hit_rate gauge")
        lines.append(f"apex_rag_accuracy_hit_rate {stats['hit_rate']}")

        # Precision
        prec = stats["precision"]
        lines.append("# HELP apex_rag_accuracy_precision Query precision (0-1)")
        lines.append("# TYPE apex_rag_accuracy_precision gauge")
        lines.append(f"apex_rag_accuracy_precision_mean {prec['mean']}")
        lines.append(f"apex_rag_accuracy_precision_p50 {prec['p50']}")
        lines.append(f"apex_rag_accuracy_precision_p95 {prec['p95']}")
        lines.append(f"apex_rag_accuracy_precision_p99 {prec['p99']}")

        # Recall
        rec = stats["recall"]
        lines.append("# HELP apex_rag_accuracy_recall Query recall (0-1)")
        lines.append("# TYPE apex_rag_accuracy_recall gauge")
        lines.append(f"apex_rag_accuracy_recall_mean {rec['mean']}")
        lines.append(f"apex_rag_accuracy_recall_p50 {rec['p50']}")
        lines.append(f"apex_rag_accuracy_recall_p95 {rec['p95']}")
        lines.append(f"apex_rag_accuracy_recall_p99 {rec['p99']}")

        # F1 Score
        f1 = stats["f1_score"]
        lines.append("# HELP apex_rag_accuracy_f1 F1 score (0-1)")
        lines.append("# TYPE apex_rag_accuracy_f1 gauge")
        lines.append(f"apex_rag_accuracy_f1_mean {f1['mean']}")
        lines.append(f"apex_rag_accuracy_f1_p50 {f1['p50']}")
        lines.append(f"apex_rag_accuracy_f1_p95 {f1['p95']}")
        lines.append(f"apex_rag_accuracy_f1_p99 {f1['p99']}")

        # Latency
        lat = stats["latency_ms"]
        lines.append("# HELP apex_rag_accuracy_latency_ms Query latency (ms)")
        lines.append("# TYPE apex_rag_accuracy_latency_ms gauge")
        lines.append(f"apex_rag_accuracy_latency_ms_mean {lat['mean']}")
        lines.append(f"apex_rag_accuracy_latency_ms_p95 {lat['p95']}")
        lines.append(f"apex_rag_accuracy_latency_ms_p99 {lat['p99']}")

        # LLM calls
        llm = stats["llm_calls_per_query"]
        lines.append("# HELP apex_rag_accuracy_llm_calls LLM calls per query")
        lines.append("# TYPE apex_rag_accuracy_llm_calls gauge")
        lines.append(f"apex_rag_accuracy_llm_calls_mean {llm['mean']}")
        lines.append(f"apex_rag_accuracy_llm_calls_p95 {llm['p95']}")

        # Total queries
        lines.append("# HELP apex_rag_accuracy_total_queries Total queries tracked")
        lines.append("# TYPE apex_rag_accuracy_total_queries counter")
        lines.append(f"apex_rag_accuracy_total_queries {stats['total_queries']}")

        # Tenant query counts
        lines.append("# HELP apex_rag_accuracy_tenant_queries Queries per tenant")
        lines.append("# TYPE apex_rag_accuracy_tenant_queries counter")
        for tenant_id, count in stats["tenant_queries"].items():
            lines.append(f'apex_rag_accuracy_tenant_queries{{tenant="{tenant_id}"}} {count}')

        return "\n".join(lines) + "\n"


# Global singleton
accuracy_tracker = AccuracyTracker()
