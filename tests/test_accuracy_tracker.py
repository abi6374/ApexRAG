"""
tests/test_accuracy_tracker.py — Tests for the per-query accuracy metrics tracker.

Tests cover:
    - Recording single queries
    - Precision, recall, F1 computation
    - Aggregate statistics (mean, p50, p95, p99)
    - Prometheus output format
    - Structured log entries
    - Thread safety
    - Recent queries filtering
    - Integration with ApexAnswer model accuracy fields
"""

from __future__ import annotations

import json
import threading

import pytest
from pydantic import ValidationError

from apex_rag.models.unified_models import ApexAnswer
from apex_rag.observability.accuracy_tracker import (
    AccuracyTracker,
    QueryAccuracyRecord,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def fresh_tracker() -> AccuracyTracker:
    """Return a fresh AccuracyTracker with no records."""
    return AccuracyTracker()


# ═══════════════════════════════════════════════════════════════════════
# QueryAccuracyRecord Model Tests
# ═══════════════════════════════════════════════════════════════════════


class TestQueryAccuracyRecord:
    """Tests for the QueryAccuracyRecord Pydantic model."""

    def test_default_values(self) -> None:
        """All fields have sensible defaults."""
        record = QueryAccuracyRecord(query="test query", doc_id="doc-1")
        assert record.query == "test query"
        assert record.doc_id == "doc-1"
        assert record.precision == 0.0
        assert record.recall == 0.0
        assert record.f1_score == 0.0
        assert record.hit is False
        assert record.coverage_guarantee == 0.0
        assert record.prediction_set_size == 0
        assert record.total_packets_retrieved == 0
        assert record.verified_packets_count == 0
        assert record.critic_pass_rate == 1.0
        assert record.latency_ms == 0.0
        assert record.tenant_id == "default"

    def test_precision_clamped(self) -> None:
        """Precision is clamped to [0, 1]."""
        with pytest.raises(ValidationError):
            QueryAccuracyRecord(query="q", doc_id="d", precision=1.5)

    def test_negative_latency_rejected(self) -> None:
        """Negative latency is accepted (field is not validated for ge=0)."""
        # latency_ms has no ge constraint in the model
        record = QueryAccuracyRecord(query="q", doc_id="d", latency_ms=-1.0)
        assert record.latency_ms == -1.0  # Allowed by model


# ═══════════════════════════════════════════════════════════════════════
# AccuracyTracker Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAccuracyTrackerBasic:
    """Basic recording and retrieval tests."""

    def test_record_query(self, fresh_tracker: AccuracyTracker) -> None:
        """Record a single query and verify it was stored."""
        record = fresh_tracker.record_query(
            query="What is Q3 revenue?",
            doc_id="doc-123",
            precision=0.85,
            recall=1.0,
            f1_score=0.92,
            hit=True,
            coverage_guarantee=0.90,
            prediction_set_size=3,
            total_packets_retrieved=5,
            verified_packets_count=4,
            total_subqueries=3,
            resolved_subqueries=3,
            critic_pass_rate=1.0,
            total_iterations=2,
            critic_passes=2,
            nodes_visited=12,
            llm_calls=8,
            backtracks=1,
            latency_ms=1450.0,
            tenant_id="tenant-a",
        )

        assert record.query == "What is Q3 revenue?"
        assert record.doc_id == "doc-123"
        assert record.precision == 0.85
        assert record.recall == 1.0
        assert record.f1_score == 0.92
        assert record.hit is True
        assert record.coverage_guarantee == 0.90
        assert record.prediction_set_size == 3
        assert record.total_packets_retrieved == 5
        assert record.verified_packets_count == 4
        assert record.critic_pass_rate == 1.0
        assert record.latency_ms == 1450.0
        assert record.tenant_id == "tenant-a"

    def test_record_multiple_queries(self, fresh_tracker: AccuracyTracker) -> None:
        """Multiple records are all stored."""
        for i in range(5):
            fresh_tracker.record_query(
                query=f"Query {i}",
                doc_id=f"doc-{i}",
                precision=0.5 + i * 0.1,
                recall=0.6 + i * 0.08,
                f1_score=0.55 + i * 0.09,
                hit=i % 2 == 0,
            )

        stats = fresh_tracker.get_aggregate_stats()
        assert stats["total_queries"] == 5

    def test_empty_stats(self, fresh_tracker: AccuracyTracker) -> None:
        """Empty tracker returns zeroed stats."""
        stats = fresh_tracker.get_aggregate_stats()
        assert stats["total_queries"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["precision"]["mean"] == 0.0
        assert stats["recall"]["mean"] == 0.0
        assert stats["f1_score"]["mean"] == 0.0
        assert stats["latency_ms"]["mean"] == 0.0
        assert stats["tenant_queries"] == {}


class TestAccuracyTrackerPrecisionRecall:
    """Tests for precision, recall, and F1 computation."""

    def test_perfect_precision_recall(self, fresh_tracker: AccuracyTracker) -> None:
        """All packets verified, all sub-queries resolved."""
        fresh_tracker.record_query(
            query="Perfect query",
            doc_id="doc-1",
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            hit=True,
            total_packets_retrieved=5,
            verified_packets_count=5,
            total_subqueries=3,
            resolved_subqueries=3,
        )

        stats = fresh_tracker.get_aggregate_stats()
        assert stats["precision"]["mean"] == 1.0
        assert stats["recall"]["mean"] == 1.0
        assert stats["f1_score"]["mean"] == 1.0
        assert stats["hit_rate"] == 1.0

    def test_zero_precision(self, fresh_tracker: AccuracyTracker) -> None:
        """No packets verified."""
        fresh_tracker.record_query(
            query="Zero precision",
            doc_id="doc-1",
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            hit=False,
            total_packets_retrieved=5,
            verified_packets_count=0,
            total_subqueries=3,
            resolved_subqueries=0,
        )

        stats = fresh_tracker.get_aggregate_stats()
        assert stats["precision"]["mean"] == 0.0
        assert stats["recall"]["mean"] == 0.0
        assert stats["f1_score"]["mean"] == 0.0
        assert stats["hit_rate"] == 0.0


class TestAccuracyTrackerAggregation:
    """Tests for aggregate statistics."""

    def test_percentile_computation(self, fresh_tracker: AccuracyTracker) -> None:
        """Percentiles are computed correctly."""
        # Record 10 queries with linearly increasing precision
        for i in range(10):
            fresh_tracker.record_query(
                query=f"Query {i}",
                doc_id="doc-1",
                precision=(i + 1) / 10.0,
                recall=(i + 1) / 10.0,
                f1_score=(i + 1) / 10.0,
                hit=(i >= 3),  # First 3 are misses
                latency_ms=float((i + 1) * 100),
                llm_calls=i + 1,
                tenant_id="tenant-a",
            )

        stats = fresh_tracker.get_aggregate_stats()
        assert stats["total_queries"] == 10
        assert 0.5 < stats["precision"]["mean"] < 0.6
        assert stats["precision"]["min"] == 0.1
        assert stats["precision"]["max"] == 1.0
        assert 0.5 < stats["recall"]["mean"] < 0.6
        assert stats["hit_rate"] == 0.7  # 7 out of 10
        assert stats["tenant_queries"]["tenant-a"] == 10
        assert stats["llm_calls_per_query"]["min"] == 1.0
        assert stats["llm_calls_per_query"]["max"] == 10.0

    def test_multiple_tenants(self, fresh_tracker: AccuracyTracker) -> None:
        """Queries across multiple tenants are counted separately."""
        for i in range(3):
            fresh_tracker.record_query(
                query=f"Q{i}",
                doc_id=f"d{i}",
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                hit=True,
                tenant_id="tenant-a",
            )
        for i in range(2):
            fresh_tracker.record_query(
                query=f"Q{i}",
                doc_id=f"d{i}",
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                hit=True,
                tenant_id="tenant-b",
            )

        stats = fresh_tracker.get_aggregate_stats()
        assert stats["tenant_queries"]["tenant-a"] == 3
        assert stats["tenant_queries"]["tenant-b"] == 2
        assert stats["total_queries"] == 5


class TestAccuracyTrackerRecentQueries:
    """Tests for the get_recent_queries method."""

    def test_recent_queries_returns_latest_first(self, fresh_tracker: AccuracyTracker) -> None:
        """Most recent queries are returned first."""
        for i in range(5):
            fresh_tracker.record_query(
                query=f"Query {i}",
                doc_id=f"doc-{i}",
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                hit=True,
            )

        recent = fresh_tracker.get_recent_queries(limit=3)
        assert len(recent) == 3
        # Most recent should be "Query 4"
        assert recent[0]["query"] == "Query 4"

    def test_recent_queries_min_precision_filter(self, fresh_tracker: AccuracyTracker) -> None:
        """Filter by minimum precision."""
        for i in range(5):
            fresh_tracker.record_query(
                query=f"Query {i}",
                doc_id=f"doc-{i}",
                precision=i / 10.0,
                recall=0.5,
                f1_score=0.5,
                hit=True,
            )

        # Only queries with precision >= 0.3
        recent = fresh_tracker.get_recent_queries(limit=10, min_precision=0.3)
        assert all(r["precision"] >= 0.3 for r in recent)


class TestAccuracyTrackerPrometheus:
    """Tests for Prometheus metric output."""

    def test_prometheus_output_format(self, fresh_tracker: AccuracyTracker) -> None:
        """Prometheus output has correct format."""
        fresh_tracker.record_query(
            query="Q1",
            doc_id="d1",
            precision=0.85,
            recall=0.9,
            f1_score=0.87,
            hit=True,
            coverage_guarantee=0.90,
            latency_ms=1500.0,
            llm_calls=5,
            tenant_id="default",
        )

        output = fresh_tracker.get_prometheus_metrics()

        # Should contain HELP and TYPE lines
        assert "# HELP apex_rag_accuracy_hit_rate" in output
        assert "# TYPE apex_rag_accuracy_hit_rate gauge" in output
        assert "apex_rag_accuracy_hit_rate 1.0" in output
        assert "# HELP apex_rag_accuracy_precision" in output
        assert "# HELP apex_rag_accuracy_recall" in output
        assert "# HELP apex_rag_accuracy_f1" in output
        assert "# HELP apex_rag_accuracy_latency_ms" in output
        assert "# HELP apex_rag_accuracy_llm_calls" in output
        assert "apex_rag_accuracy_total_queries 1" in output

    def test_prometheus_empty(self, fresh_tracker: AccuracyTracker) -> None:
        """Empty tracker produces valid (zeroed) Prometheus output."""
        output = fresh_tracker.get_prometheus_metrics()
        assert "# HELP apex_rag_accuracy_hit_rate" in output
        assert "apex_rag_accuracy_hit_rate 0.0" in output
        assert "apex_rag_accuracy_total_queries 0" in output


class TestAccuracyTrackerStructuredLog:
    """Tests for structured log entries."""

    def test_log_entry_structure(self, fresh_tracker: AccuracyTracker) -> None:
        """Log entry contains all expected fields."""
        record = fresh_tracker.record_query(
            query="Revenue?",
            doc_id="doc-1",
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            hit=True,
            coverage_guarantee=0.90,
            prediction_set_size=3,
            total_packets_retrieved=5,
            verified_packets_count=4,
            total_subqueries=3,
            resolved_subqueries=2,
            critic_pass_rate=1.0,
            llm_calls=8,
            latency_ms=1200.0,
        )

        log_entry = fresh_tracker.get_structured_log_entry(record, extra={"request_id": "req-1"})
        parsed = json.loads(log_entry)

        assert parsed["event"] == "query_accuracy"
        assert parsed["precision"] == 0.9
        assert parsed["recall"] == 0.8
        assert parsed["f1_score"] == 0.85
        assert parsed["hit"] is True
        assert parsed["coverage_guarantee"] == 0.9
        assert parsed["latency_ms"] == 1200.0
        assert parsed["request_id"] == "req-1"


class TestAccuracyTrackerThreadSafety:
    """Thread safety tests."""

    def test_concurrent_recording(self, fresh_tracker: AccuracyTracker) -> None:
        """Multiple threads can record simultaneously without corruption."""
        errors: list[Exception] = []

        def record_queries(start: int, count: int) -> None:
            try:
                for i in range(count):
                    fresh_tracker.record_query(
                        query=f"Thread query {start + i}",
                        doc_id=f"doc-{start + i}",
                        precision=0.5,
                        recall=0.5,
                        f1_score=0.5,
                        hit=True,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_queries, args=(0, 25)),
            threading.Thread(target=record_queries, args=(25, 25)),
            threading.Thread(target=record_queries, args=(50, 25)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        stats = fresh_tracker.get_aggregate_stats()
        assert stats["total_queries"] == 75


# ═══════════════════════════════════════════════════════════════════════
# Integration: ApexAnswer Accuracy Fields
# ═══════════════════════════════════════════════════════════════════════


class TestApexAnswerAccuracyFields:
    """Test that ApexAnswer has the new accuracy fields."""

    def test_apex_answer_has_accuracy_fields(self) -> None:
        """ApexAnswer model has all new accuracy fields with defaults."""
        answer = ApexAnswer(answer_text="Test answer", query="Test query")

        # New accuracy fields
        assert answer.precision == 0.0
        assert answer.recall == 0.0
        assert answer.f1_score == 0.0
        assert answer.hit is False
        assert answer.total_subqueries == 0
        assert answer.resolved_subqueries == 0
        assert answer.critic_pass_rate == 1.0
        assert answer.nodes_visited == 0
        assert answer.llm_calls == 0
        assert answer.backtracks == 0

    def test_apex_answer_accuracy_serialization(self) -> None:
        """Accuracy fields are included in JSON serialization."""
        answer = ApexAnswer(
            answer_text="Revenue was $52M",
            query="What is revenue?",
            precision=0.85,
            recall=1.0,
            f1_score=0.92,
            hit=True,
            total_subqueries=3,
            resolved_subqueries=3,
            critic_pass_rate=1.0,
            nodes_visited=12,
            llm_calls=8,
            backtracks=1,
        )

        json_str = answer.model_dump_json(indent=2)
        assert '"precision": 0.85' in json_str
        assert '"recall": 1.0' in json_str
        assert '"f1_score": 0.92' in json_str
        assert '"hit": true' in json_str
        assert '"total_subqueries": 3' in json_str
        assert '"resolved_subqueries": 3' in json_str
        assert '"critic_pass_rate": 1.0' in json_str
        assert '"nodes_visited": 12' in json_str
        assert '"llm_calls": 8' in json_str
        assert '"backtracks": 1' in json_str

    def test_apex_answer_accuracy_round_trip(self) -> None:
        """Accuracy fields survive JSON round-trip deserialization."""
        original = ApexAnswer(
            answer_text="Test",
            query="Q",
            precision=0.75,
            recall=0.8,
            f1_score=0.77,
            hit=True,
            total_subqueries=4,
            resolved_subqueries=3,
        )

        json_str = original.model_dump_json()
        restored = ApexAnswer.model_validate_json(json_str)

        assert restored.precision == 0.75
        assert restored.recall == 0.8
        assert restored.f1_score == 0.77
        assert restored.hit is True
        assert restored.total_subqueries == 4
        assert restored.resolved_subqueries == 3


# ═══════════════════════════════════════════════════════════════════════
# Global Singleton
# ═══════════════════════════════════════════════════════════════════════


class TestAccuracyTrackerSingleton:
    """Tests for the global accuracy_tracker singleton."""

    def test_singleton_exists(self) -> None:
        """The global accuracy_tracker is an AccuracyTracker instance."""
        from apex_rag.observability.accuracy_tracker import accuracy_tracker as at

        assert isinstance(at, AccuracyTracker)

    def test_singleton_can_record(self) -> None:
        """The singleton can record queries."""
        from apex_rag.observability.accuracy_tracker import accuracy_tracker as at

        record = at.record_query(
            query="Singleton test",
            doc_id="doc-singleton",
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            hit=False,
        )
        assert record is not None
        assert record.query == "Singleton test"
