"""
tests/test_metrics_service.py — Tests for the MetricsService (Phase 11).
"""

from __future__ import annotations

import re

import pytest

from apex_rag.observability.metrics_service import MetricsService


class TestMetricsService:
    """Tests for the production metrics service."""

    @pytest.fixture
    def metrics(self) -> MetricsService:
        return MetricsService()

    def test_completed_query_and_latency(self, metrics: MetricsService) -> None:
        metrics.record_completed_query()
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

    # ═══════════════════════════════════════════════════════════════════
    # Prometheus Format Integration Tests
    # ═══════════════════════════════════════════════════════════════════

    def test_prometheus_format_structure(self, metrics: MetricsService) -> None:
        """Verify the overall Prometheus text exposition format structure.

        Every non-header line must match: metric_name[{labels}] value.
        HELP and TYPE lines must precede the metrics they describe.
        """
        metrics.record_retrieval_latency(100.0)
        metrics.record_cache_hit()
        output = metrics.get_prometheus_metrics()

        # Must end with a trailing newline
        assert output.endswith("\n")

        lines = output.strip().split("\n")
        assert len(lines) > 0

        for line in lines:
            line = line.strip()
            assert line, "No blank lines allowed in Prometheus output"

            if line.startswith("#"):
                # Must be either HELP or TYPE
                assert line.startswith("# HELP ") or line.startswith("# TYPE "), (
                    f"Invalid header line: {line}"
                )
            else:
                # Must be a metric line: metric_name[{labels}] value
                assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\{.*\})?\s[\d\.eE+\-InfNaN]+$", line), (
                    f"Invalid metric line: {line}"
                )

    def test_prometheus_histogram_has_required_fields(self, metrics: MetricsService) -> None:
        """Verify each histogram has TYPE, _count, _sum, and _bucket lines."""
        metrics.record_retrieval_latency(50.0)
        metrics.record_retrieval_latency(200.0)
        output = metrics.get_prometheus_metrics()

        # Check the retrieval latency histogram
        metric = "apex_rag_retrieval_latency_ms"
        assert f"# TYPE {metric} histogram" in output
        assert f"{metric}_count 2" in output
        assert f"{metric}_sum" in output

        # Check +Inf bucket matches _count
        assert f'{metric}_bucket{{le="+Inf"}} 2' in output

        # Check we have bucket lines
        bucket_lines = [line for line in output.split("\n") if f"{metric}_bucket{{le=" in line]
        assert len(bucket_lines) >= 3, "Expected at least 3 bucket lines"

    def test_prometheus_histogram_bucket_counts_correct(self, metrics: MetricsService) -> None:
        """Verify bucket counts reflect actual data distribution."""
        values = [3.0, 8.0, 20.0, 60.0, 150.0, 600.0, 3000.0]
        for v in values:
            metrics.record_retrieval_latency(v)

        output = metrics.get_prometheus_metrics()
        metric = "apex_rag_retrieval_latency_ms"

        # Expected bucket counts for standard buckets: [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, +Inf]
        # Values: 3.0, 8.0, 20.0, 60.0, 150.0, 600.0, 3000.0
        # 3.0 <= 5.0 ✓ | 8.0 <= 10.0 ✓ | 20.0 <= 25.0 ✓
        # 60.0 > 50.0 — not counted in le="50.0"
        # 60.0 <= 100.0 ✓ | 150.0 <= 250.0 ✓
        # 600.0 > 500.0 — not counted in le="500.0"
        # 600.0 <= 1000.0 ✓ | 3000.0 <= 5000.0 ✓

        lines = output.splitlines()
        assert f'{metric}_bucket{{le="5.0"}} 1' in lines
        assert f'{metric}_bucket{{le="10.0"}} 2' in lines
        assert f'{metric}_bucket{{le="25.0"}} 3' in lines
        assert f'{metric}_bucket{{le="50.0"}} 3' in lines  # 60 > 50
        assert f'{metric}_bucket{{le="100.0"}} 4' in lines
        assert f'{metric}_bucket{{le="250.0"}} 5' in lines
        assert f'{metric}_bucket{{le="500.0"}} 5' in lines  # 600 > 500
        assert f'{metric}_bucket{{le="1000.0"}} 6' in lines
        assert f'{metric}_bucket{{le="5000.0"}} 7' in lines
        assert f'{metric}_bucket{{le="+Inf"}} 7' in lines

    def test_prometheus_all_five_latency_histograms_present(self, metrics: MetricsService) -> None:
        """Verify all 5 latency stages produce histogram metrics."""
        metrics.record_retrieval_latency(10.0)
        metrics.record_planner_latency(20.0)
        metrics.record_navigator_latency(30.0)
        metrics.record_verifier_latency(15.0)
        metrics.record_critic_latency(5.0)

        output = metrics.get_prometheus_metrics()

        stages = ["retrieval", "planner", "navigator", "verifier", "critic"]
        for stage in stages:
            metric = f"apex_rag_{stage}_latency_ms"
            assert f"# TYPE {metric} histogram" in output, f"Missing TYPE for {metric}"
            assert f"{metric}_count" in output, f"Missing _count for {metric}"
            assert f"{metric}_sum" in output, f"Missing _sum for {metric}"
            assert f'{metric}_bucket{{le="+Inf"}}' in output, f"Missing +Inf bucket for {metric}"

    def test_prometheus_counter_types_declared(self, metrics: MetricsService) -> None:
        """Verify all counter metrics have the correct TYPE declaration."""
        metrics.record_completed_query()
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        metrics.record_tenant_query("t1")
        metrics.increment_llm_calls(3)

        output = metrics.get_prometheus_metrics()

        counters = {
            "apex_rag_total_queries": 1,
            "apex_rag_cache_hits": 1,
            "apex_rag_cache_misses": 1,
            "apex_rag_llm_calls": 3,
            "apex_rag_tenant_queries_total": None,  # checked separately
        }

        for name, expected_value in counters.items():
            if expected_value is not None:
                assert f"# TYPE {name} counter" in output, f"Missing TYPE counter for {name}"
                assert f"{name} {expected_value}" in output, (
                    f"Expected {name} = {expected_value}"
                )

        # Tenant query lines have labels
        assert "# TYPE apex_rag_tenant_queries_total counter" in output
        assert 'apex_rag_tenant_queries_total{tenant="t1"} 1' in output

    def test_prometheus_gauge_types_declared(self, metrics: MetricsService) -> None:
        """Verify all gauge metrics have the correct TYPE declaration."""
        output = metrics.get_prometheus_metrics()

        gauges = [
            "apex_rag_uptime_seconds",
            "apex_rag_cache_hit_rate",
        ]

        for name in gauges:
            assert f"# TYPE {name} gauge" in output, f"Missing TYPE gauge for {name}"

    def test_prometheus_empty_output_all_zeros(self, metrics: MetricsService) -> None:
        """Verify output with no data recorded shows all zero counts."""
        output = metrics.get_prometheus_metrics()

        lines = output.strip().split("\n")
        # Find metric value lines (not HELP/TYPE)
        metric_lines = [line for line in lines if not line.startswith("#")]

        for line in metric_lines:
            # Extract the value part (after optional labels)
            parts = line.split()
            if len(parts) >= 2:
                value_str = parts[-1]
                # Only assert 0 for counter-type metrics, skip uptime
                if "_total" in line or "_count" in line or "_hit" in line or "_miss" in line or "_llm" in line:
                    assert value_str in ("0", "0.0"), (
                        f"Expected 0 for empty metric, got {value_str} in: {line}"
                    )

    def test_prometheus_tenant_labels_format(self, metrics: MetricsService) -> None:
        """Verify tenant query count uses correct Prometheus label syntax."""
        metrics.record_tenant_query("tenant-alpha")
        metrics.record_tenant_query("tenant-beta")
        metrics.record_tenant_query("tenant-beta")

        output = metrics.get_prometheus_metrics()

        # Check label syntax: metric_name{label="value"} value
        assert 'apex_rag_tenant_queries_total{tenant="tenant-alpha"} 1' in output
        assert 'apex_rag_tenant_queries_total{tenant="tenant-beta"} 2' in output

    def test_prometheus_no_duplicate_metric_names(self, metrics: MetricsService) -> None:
        """Verify no metric name appears in multiple TYPE declarations."""
        metrics.record_retrieval_latency(10.0)

        output = metrics.get_prometheus_metrics()

        # Count TYPE declarations per metric name
        type_lines = [line for line in output.split("\n") if line.startswith("# TYPE ")]
        metric_names = [line.split()[2] for line in type_lines]

        assert len(metric_names) == len(set(metric_names)), (
            f"Duplicate TYPE declarations: {metric_names}"
        )

    def test_prometheus_histogram_types_for_all_stages(self, metrics: MetricsService) -> None:
        """Verify all 5 latency histogram stages have TYPE histogram declaration."""
        output = metrics.get_prometheus_metrics()

        stages = [
            "apex_rag_retrieval_latency_ms",
            "apex_rag_planner_latency_ms",
            "apex_rag_navigator_latency_ms",
            "apex_rag_verifier_latency_ms",
            "apex_rag_critic_latency_ms",
        ]

        for stage in stages:
            assert f"# TYPE {stage} histogram" in output, f"Missing TYPE histogram for {stage}"
