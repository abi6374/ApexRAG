# Monitoring & Observability

ApexRAG provides multiple layers of observability — from simple console traces
to full OpenTelemetry distributed tracing.

## Console Trace

The **ReasoningTrace** prints colour-coded navigation decisions in real-time:

```
━━━ ApexRAG Navigation Start ━━━
Query : What was the Q3 revenue growth?
Root  : node_id=1

  ↳ ENTER node=1 path=1
    [Annual Report 2024 — Financial overview...]

  ↳ EXPLORE node=2 → evaluating 3 child summaries
    ✔ AGENT → node=3  reason: "Financial data most relevant"

  ↳ ENTER node=3 path=2
    [Financials — Revenue, expenses, P&L highlights...]

    ↳ EXPLORE node=3 → evaluating 3 child summaries
      ✔ AGENT → node=4  reason: "Q3 data contains revenue"

    ↳ ENTER node=4 path=2.3
      ★ LEAF REACHED node=4
        preview: Q3 revenue reached $165M...

      ✔ VERIFY node=4 → answers_query=true (confidence=0.95)

━━━ Navigation Complete ━━━  result=SUCCESS  elapsed=4.23s
```

Disable with `trace_enabled=False` in production environments.

## Query Metrics

ApexRAG tracks in-memory query metrics that are always available:

```python
from apex_rag.telemetry import query_metrics

# Get current metrics
metrics = query_metrics.to_dict()
print(metrics)
# {
#   "total_queries": 150,
#   "cache_hits": 42,
#   "cache_misses": 108,
#   "cache_hit_rate": 0.28,
#   "avg_latency_ms": 3250.5,
#   "p99_latency_ms": 8921.3,
#   "error_count": 2
# }
```

Accessible via the REST API by importing `query_metrics` in your own handlers.

## OpenTelemetry

For production deployments, ApexRAG supports full OpenTelemetry instrumentation:

```bash
pip install apex-rag[telemetry]
```

```python
from apex_rag.telemetry import setup_telemetry, get_tracer

# Initialize at app startup
setup_telemetry(
    service_name="apex-rag",
    otlp_endpoint="http://localhost:4317",
    enable_tracing=True,
    enable_metrics=True,
)

# Use in your code
tracer = get_tracer()
with tracer.start_as_current_span("query") as span:
    span.set_attribute("doc_id", doc_id)
    span.set_attribute("question_length", len(question))
    result = await index.query(question, doc_id)
```

## Structured JSON Logging

For log aggregators (DataDog, ELK, Grafana Loki):

```bash
export APEX_LOG_FORMAT=json
export APEX_LOG_LEVEL=INFO
```

This produces JSON log entries:

```json
{"timestamp": "2026-04-10T15:30:00+00:00", "level": "INFO", "logger": "apex_rag",
 "module": "client", "function": "create", "line": 120,
 "message": "ApexIndex ready | db=sqlite+aiosqlite:///apex.db | verify=True"}
```

## Prometheus + Grafana

For metrics-based monitoring, use the included Docker Compose profile:

```bash
docker compose --profile monitoring up -d
```

This starts:
- **Prometheus** on port 9090 — scrapes metrics from the API server
- **Grafana** on port 3000 — pre-configured dashboards for query latency,
  cache hit rates, and error counts

## Metrics Dashboard

Track these key metrics:

| Metric | What it tells you |
|--------|-------------------|
| Query latency (avg/p99) | Performance degradation over time |
| Cache hit rate | How effective the semantic cache is |
| Error rate | LLM/DB connectivity issues |
| Ingestion throughput | Documents processed per minute |
| Navigation depth | Complexity of queries (deeper = harder) |
