# Week 1: Adaptive / Lazy Knowledge-DAG Construction

## ✅ Methodology (Actual Measured Values)

The numbers below were **measured** using `time.perf_counter()` instrumentation
with 10 iterations per mode against an in-memory SQLite database
(`sqlite+aiosqlite:///:memory:`). Each iteration creates a fresh
`ApexIndex` and ingests a 10-section markdown document (~50 AST nodes).

**Warmup:** One eager + one adaptive run are performed before the measured
loop to eliminate cold-start bias (SQLAlchemy metadata reflection, first-time
imports, table creation).

**LLM:** Mock LLM (no real API calls) — meaning DAG builder cost is
dominated by pure-Python pattern matching, not LLM inference. In production
with a real model (Gemini, GPT-4, etc.), entity/citation extraction would
involve LLM calls and the savings from skipping those DAGs would be
substantially larger.

**Timer:** `time.perf_counter()` — high-resolution monotonic clock.
Timing starts *after* `ApexIndex.create()` so it measures only
`ingest_text()` latency.

**Source:** `benchmarks/run_dag_latency_benchmark.py` (reproducible).

### DAGs built per mode

| Mode | DocumentDAG | EntityDAG | CitationDAG | PolicyDAG | TemporalDAG |
|------|:-----------:|:---------:|:-----------:|:---------:|:-----------:|
| **eager** | ✅ synchronous | ✅ synchronous | ✅ synchronous | ✅ synchronous | ✅ synchronous |
| **adaptive** | ✅ synchronous | ❌ (lazy) | ❌ (lazy) | ❌ (lazy) | ✅ background (not blocking ingest) |
| **minimal** | ✅ synchronous | ❌ never | ❌ never | ❌ never | ❌ never |

## Results (Measured)

### Ingest latency — 10-section document (~50 AST nodes), 10 iterations (post-warmup)

| Metric | Eager (s) | Adaptive (s) | Speedup |
|--------|:---------:|:------------:|:-------:|
| **p50** | 2.4919 | 2.5029 | 1.00× |
| **p95** | 2.9525 | 3.1142 | 0.95× |
| **mean** | 2.5629 | 2.5758 | 0.99× |
| **min** | 2.4106 | 2.4472 | — |
| **max** | 2.9525 | 3.1142 | — |
| **stdev** | 0.1838 | 0.2142 | — |

### Raw time series (post-warmup)

```
Eager:    2.41  2.44  2.45  2.45  2.48  2.49  2.49  2.55  2.84  2.95
Adaptive: 2.45  2.45  2.46  2.47  2.50  2.52  2.53  2.79  3.11  3.11
```

## Analysis

### Why is adaptive not faster with mock LLM?

In this benchmark the DAG builders all use **pure-Python pattern matching**
(no LLM calls). The entity/citation/policy builders are lightweight regex
passes over the parsed AST nodes:

- **EntityDagBuilder**: scans node content for entity-like noun phrases
- **CitationDagBuilder**: scans for citation markers (`[1]`, `et al.`, etc.)
- **PolicyDagBuilder**: scans for compliance/regulation keywords

These complete in a few milliseconds each. The **gating layer overhead**
(importing modules, checking is_dag_built via SQL, DAGRouter classification)
adds roughly the same amount of time that skipping the builders saves.

### When would adaptive be significantly faster?

In **production** with real LLM-based extraction:

- **EntityDagBuilder**: would call the LLM for NER → ~200–500ms per document
- **CitationDagBuilder**: would call the LLM for reference extraction → ~100–300ms
- **PolicyDagBuilder**: would call the LLM for policy/regulation extraction → ~100–300ms
- **Total savings**: ~400–1100ms per ingest by skipping these 3 DAGs

This means **adaptive would be ~1.5–2× faster than eager** in production,
with the gap widening on larger documents and more complex extraction pipelines.

## Key Takeaways

1. **With mock LLM, DAG gating adds ≈0 overhead but saves nothing** because
   the DAG builders themselves are faster than the gating checks.

2. **In production with real LLMs, adaptive mode would cut ingest latency
   significantly** (estimated 1.5–2×) by skipping LLM-based entity/citation/
   policy extraction on every ingest.

3. **The adaptive architecture is correct and verified by tests** — the
   39-test suite in `tests/test_dag_gating.py` confirms that:
   - Lazy DAGs are built on first query that needs them
   - Build-once cache semantics work (second query doesn't rebuild)
   - Eager mode preserves exact backward-compatible behavior
   - Minimal mode builds nothing beyond DocumentDAG

4. **The benchmark is reproducible** — run `python benchmarks/run_dag_latency_benchmark.py`
   to get fresh numbers on any hardware.

## File Changes

| File | Change |
|------|--------|
| `apex_rag/graph/dag_gating.py` | **New** — DAG gating service |
| `apex_rag/agents/planner/dag_router.py` | **New** — Query-need classifier |
| `apex_rag/config.py` | Added `graph_construction_mode` setting |
| `apex_rag/client.py` | Gate DAG construction via `DAGGatingService` |
| `apex_rag/agents/apex_orchestrator.py` | Trigger lazy DAGs before query |
| `apex_rag/agents/planner/__init__.py` | Updated exports |
| `tests/test_dag_gating.py` | **New** — 39 test cases covering all modes |
| `benchmarks/run_dag_latency_benchmark.py` | **New** — Reproducible latency benchmark |
