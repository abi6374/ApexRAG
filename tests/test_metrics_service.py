"""
tests/test_metrics_service.py — Tests for the MetricsService (Phase 11).
"""

from __future__ import annotations

import pytest

from apex_rag.observability.metrics_service import MetricsService


class TestMetricsService:
    """Tests for the production metrics service."""

    @pytest.fixture
    def metrics(self) -> MetricsService:
        return MetricsService()

    def test_record_retrieval_latency(self, metrics: MetricsService) -> None:
        metrics.record_retrieval_latency(150.5)
        stats = metrics.get_stats()
        assert stats["total_queries"] == 1
        assert stats["retrieval_latency"]["count"] == 1
        assert stats["retrieval_latency"]["sum"] == 150.5

    def test_multiple_latencies(self, metrics: MetricsService) -> None:
        metrics.record_retrieval_latency(100.0)
        metrics.record_retrieval_latency(200.0)
        metrics.record_retrieval_latency(300.0)
        stats = metrics.get_stats()
        assert stats["retrieval_latency"]["count"] == 3
        assert stats["retrieval_latency"]["avg"] == 200.0
        assert stats["retrieval_latency"]["p50"] == 200.0

    def test_cache_hits_and_misses(self, metrics: MetricsService) -> None:
        metrics.record_cache_hit()
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        stats = metrics.get_stats()
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == round(2.0 / 3.0, 4)

    def test_tenant_queries(self, metrics: MetricsService) -> None:
        metrics.record_tenant_query("tenant-a")
        metrics.record_tenant_query("tenant-a")
        metrics.record_tenant_query("tenant-b")
        stats = metrics.get_stats()
        assert stats["tenant_queries"]["tenant-a"] == 2
        assert stats["tenant_queries"]["tenant-b"] == 1

    def test_llm_calls(self, metrics: MetricsService) -> None:
        metrics.increment_llm_calls(5)
        stats = metrics.get_stats()
        assert stats["llm_calls"] == 5

    def test_prometheus_output(self, metrics: MetricsService) -> None:
        metrics.record_retrieval_latency(100.0)
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        metrics.record_tenant_query("tenant-a")

        output = metrics.get_prometheus_metrics()
        assert "apex_rag_total_queries" in output
        assert "apex_rag_cache_hits" in output
        assert "apex_rag_cache_misses" in output
        assert "apex_rag_retrieval_latency_ms_count" in output
        assert 'apex_rag_tenant_queries_total{tenant="tenant-a"}' in output

    def test_structured_log_entry(self) -> None:
        log_entry = MetricsService.make_log_entry(
            request_id="req-123",
            tenant_id="tenant-a",
            user_id="user-1",
            query_id="q-456",
            event="query_completed",
        )
        assert "req-123" in log_entry
        assert "tenant-a" in log_entry
        assert "user-1" in log_entry
        assert "q-456" in log_entry

    def test_all_latency_types(self, metrics: MetricsService) -> None:
        metrics.record_planner_latency(50.0)
        metrics.record_navigator_latency(100.0)
        metrics.record_verifier_latency(20.0)
        metrics.record_critic_latency(10.0)

        stats = metrics.get_stats()
        assert stats["planner_latency"]["count"] == 1
        assert stats["navigator_latency"]["count"] == 1
        assert stats["verifier_latency"]["count"] == 1
        assert stats["critic_latency"]["count"] == 1

    def test_empty_histogram(self, metrics: MetricsService) -> None:
        stats = metrics.get_stats()
        assert stats["retrieval_latency"]["count"] == 0
        assert stats["retrieval_latency"]["sum"] == 0.0
