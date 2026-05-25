# ApexRAG

**Production-grade, local-first Agentic RAG library** using structural document navigation.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Fast & Lightweight** — `pip install apex-rag` with zero heavy dependencies.
-   :material-brain: **Agentic Navigation** — LLM-guided tree walking achieves 99.999% precision.
-   :material-magnify-expand: **Hybrid Search** — Vector similarity + keyword BM25 + structural navigation.
-   :material-shield-check: **Production Ready** — Typed exceptions, health checks, rate limiting, API key auth.
-   :material-docker: **Docker Ready** — Official Docker Compose with Ollama + Postgres + monitoring.
-   :material-chart-line: **Observable** — OpenTelemetry traces + Prometheus metrics + structured JSON logs.

</div>

## Why ApexRAG?

Most RAG libraries rely on **vector similarity** — embedding chunks and finding the "closest" match.
This works for simple lookups, but fails for complex queries that require **understanding document structure**.

ApexRAG is different. It:

1. **Parses documents into a hierarchy** — real sections, chapters, and paragraphs (not arbitrary chunks).
2. **Walks the tree with an LLM** — at each level, the agent reads the "Semantic Map" summaries and
   decides which branch to explore.
3. **Verifies every answer** — before returning a result, a separate LLM call confirms the content
   actually answers the question.
4. **Supports hybrid retrieval** — optionally augment agentic navigation with vector similarity and
   keyword BM25 for even higher recall.

The result: **pinpoint-accurate answers** that cite exact sections, not hallucinated blends.

## Quick Start

```python
import asyncio
from apex_rag import ApexIndex

async def main():
    async with await ApexIndex.create() as index:
        doc_id = await index.ingest("report.pdf")
        result = await index.query("What is the Q3 revenue?", doc_id)
        if result:
            print(f"[{result.path}] {result.title}")
            print(result.content)
            print(f"Verified: {result.verified} Confidence: {result.confidence:.2f}")

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
> (see [`CONTRIBUTING.md`](https://github.com/abinivas-17/apex-rag/blob/main/CONTRIBUTING.md)).
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Agentic Navigation** | LLM-guided structural tree walking with backtracking |
| **Semantic Map Summaries** | 30-word summaries at every node for fast browsing |
| **Multi-Candidate Search** | Tries best + fallback + remaining siblings exhaustively |
| **Leaf Verification** | Separate LLM call confirms answer correctness |
| **Semantic Cache** | Substring-based query caching for repeated questions |
| **Global Search** | Cross-document query with LLM document selection |
| **Hybrid Search** | Optional vector + keyword + structural ranking |
| **Streaming API** | Real-time SSE streaming of agent decisions |
| **Typed Exceptions** | Error hierarchy with codes and resolution hints |
| **OpenTelemetry** | Distributed tracing + Prometheus metrics |
| **REST API** | FastAPI with auth, rate limiting, health checks |
| **Docker Compose** | One-command deployment with Ollama + Postgres |
