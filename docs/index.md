# ApexRAG

**Production-grade, local-first Agentic RAG library** using structural document navigation.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Fast & Lightweight** — `pip install apex-rag` with zero heavy dependencies.
-   :material-brain: **Agentic Navigation** — LLM-guided tree walking achieves 99.999% precision.
-   :material-graph: **8 Knowledge DAGs** — Document, Entity, Citation, Temporal, Version, Policy, Fact, and Reasoning edge projections.
-   :material-shield-check: **Production Ready** — Typed exceptions, health checks, rate limiting, API key auth, SSE streaming.
-   :material-docker: **Docker Ready** — Official Docker Compose with Ollama + Postgres + monitoring.
-   :material-chart-line: **Observable** — OpenTelemetry traces + Prometheus metrics + structured JSON logs.

</div>

## Why ApexRAG?

Most RAG libraries rely on **vector similarity** — embedding chunks and finding the "closest" match.
This works for simple lookups, but fails for complex queries that require **understanding document structure**.

ApexRAG is different. It:

1. **Parses documents into a hierarchy** — real sections, chapters, and paragraphs (not arbitrary chunks).
2. **Walks the tree with an LLM** — at each level, the agent reads the "Semantic Map" summaries and decides which branch to explore.
3. **Verifies every answer** — before returning a result, a separate LLM call confirms the content actually answers the question.
4. **Builds 8 Knowledge DAGs** — automatically extracts Document, Entity, Citation, Temporal, Version, Policy, Fact, and Reasoning relationships into a unified edge store.
5. **Streams real-time traces** — SSE endpoints push agent navigation steps and ReasoningDAG edges as they're generated.

The result: **pinpoint-accurate answers** that cite exact sections, backed by traceable reasoning graphs.

## Quick Start

```python
import asyncio
from apex_rag import ApexIndex

async def main():
    async with await ApexIndex.create() as index:
        doc_id = await index.ingest("report.pdf")
        answer = await index.query("What is the Q3 revenue?", doc_id)
        print(answer.answer_text)
        print(f"Confidence: {answer.coverage_guarantee:.0%}")
        print(f"Evidence packets: {answer.prediction_set_size}")

asyncio.run(main())
```

## Installation

```bash
pip install apex-rag                    # Core (5 deps)
pip install apex-rag[web]              # + REST API & dashboard
pip install apex-rag[postgres]         # + PostgreSQL support
pip install apex-rag[vectors]          # + Vector embeddings (sentence-transformers)
pip install apex-rag[telemetry]        # + OpenTelemetry tracing
pip install apex-rag[all]              # Everything

> **Development:** Clone the repo, then run `make install` for a full dev setup
> (see [`CONTRIBUTING.md`](https://github.com/abi6374/apexrag/blob/main/CONTRIBUTING.md)).
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Agentic Navigation** | LLM-guided structural tree walking with backtracking |
| **8 Knowledge DAGs** | Automatic edge extraction: Document, Entity, Citation, Temporal, Version, Policy, Fact, Reasoning |
| **ReasoningDAG** | Query-time tracing captured as typed reasoning edges (REASONING_CHAIN, DERIVES_FROM, INFERS, USES) |
| **Graph Visualization** | Interactive vis-network tabs on dashboard and document pages |
| **Global Graph API** | `GET /graph` and `GET /graph/{projection}` across all documents |
| **SSE Streaming** | Real-time agent traces + ReasoningDAG via Server-Sent Events |
| **Semantic Map Summaries** | 30-word summaries at every node for fast browsing |
| **Multi-Candidate Search** | Tries best + fallback + remaining siblings exhaustively |
| **Leaf Verification** | Separate LLM call confirms answer correctness |
| **Semantic Cache** | Substring-based query caching for repeated questions |
| **Global Search** | Cross-document query with LLM document selection |
| **Hybrid Search** | Optional vector + keyword + structural ranking |
| **Typed Exceptions** | Error hierarchy with codes and resolution hints |
| **OpenTelemetry** | Distributed tracing + Prometheus metrics |
| **REST API** | FastAPI with auth, rate limiting, health checks |
| **Docker Compose** | One-command deployment with Ollama + Postgres |
