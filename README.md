<p align="center">
  <img src="https://img.shields.io/badge/ApexRAG-v1.0.4-6366f1?style=for-the-badge&logo=python&logoColor=white" alt="ApexRAG v1.0.4">
  <img src="https://img.shields.io/pypi/v/apex-rag?style=for-the-badge&color=6366f1" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/pyversions/apex-rag?style=for-the-badge" alt="Python Versions">
  <img src="https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge" alt="License">
  <a href="https://pepy.tech/projects/apex-rag"><img src="https://static.pepy.tech/personalized-badge/apex-rag?period=total&units=INTERNATIONAL_SYSTEM&left_color=black&right_color=6366f1&left_text=downloads" alt="PyPI Downloads"></a>
</p>

<h1 align="center">⚡ ApexRAG</h1>

<p align="center">
  <strong>High-Accuracy Structural Retrieval Infrastructure for Production AI.</strong><br>
  <em>Stop guessing with vectors. Start navigating with agents.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/apex-rag/">PyPI</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-cli-interface">CLI</a> •
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

## 🔍 What is ApexRAG?

**ApexRAG** is a **Multi-Agent, Structural Reasoning Engine** designed for precise enterprise document retrieval and production RAG deployments.

Traditional RAG pipelines rely on flat vector proximity — slicing documents into arbitrary chunks, destroying their logical hierarchy (headings, sections, tables, cross-references). This leads to **lost context** and **hallucinations**.

ApexRAG solves this by:

1. **Parsing documents into a Universal AST** — a strict hierarchical tree that preserves every structural relationship.
2. **Running a coordinated LLM Agent loop** — Planner → Navigator → Critic — that explicitly traverses the AST to find verifiable answers.
3. **Guaranteeing confidence** — every answer comes with a statistically grounded coverage guarantee via Conformal Prediction.

```
Document (PDF/MD/Code/Image)
        │
        ▼ ApexParser
  Universal AST Nodes ──► Semantic Signposts ──► Causal Knowledge Graph
        │
        ▼ ApexStorage (SQLite / PostgreSQL)

User Query
        │
        ▼ QueryPlannerAgent  →  ASTNavigationAgent  →  EvaluationCriticAgent
                                                              │
                                                              ▼
                                                  ApexAnswer + Confidence Score
```

---

## 🏗️ Architecture

### Phase 1 — Structural Foundation

| Component | Description |
|---|---|
| **Universal Document AST** | Documents are parsed into typed `ASTNode` trees, preserving exact paragraph-to-heading and table-to-caption structures. |
| **Deterministic Pre-Retrieval** | Keyword density scoring, FTS5 full-text search, and structural heading overlap narrow candidates before any LLM call — keeping costs low. |
| **StrictLeafVerifier** | An empirical verification engine that checks whether a retrieved node actually answers the query, acting as a firewall against hallucinated evidence. |

### Phase 2 — Structural Reasoning Engine

| Agent | Role |
|---|---|
| **Planner Agent** | Deconstructs complex, multi-hop queries into discrete sub-queries. |
| **Navigator Agent** | Traverses AST nodes and Semantic Map signposts to retrieve grounded context for each sub-query. |
| **Critic Agent** | Audits and scores the retrieved context, enforcing completeness before the final answer is synthesized. |

The **Structural Retrieval Graph (SRG)** connects nodes via typed semantic edges (`REFERENCES_TABLE`, `SUPERSEDES`, `CAUSED_BY`), enabling non-linear multi-hop reasoning.

### Phase 3 — Enterprise Ecosystem

- **Multi-Tenant RBAC** — SQLAlchemy models enforce strict data boundaries via `tenant_id`. All queries are automatically scoped.
- **Temporal Querying** — Query any document *as it was* at a specific point in time. Compare states across versions.
- **Distributed Ingestion** — A `DistributedIndexer` scales document parsing across workers via Redis or Celery queues.
- **Code Intelligence** — `PythonCodeParser` extracts ASTs from `.py` source files for precise code reasoning.
- **OpenTelemetry Tracing** — Every agent action (`[PLANNING]`, `[NAVIGATING]`, `[EVALUATING]`) is traced and exportable to any OTLP backend.

---

## 📦 Installation

```bash
pip install apex-rag
```

Install with optional feature extras:

```bash
# All features
pip install "apex-rag[all]"

# Extra LLM providers
pip install "apex-rag[anthropic]"    # Anthropic Claude
pip install "apex-rag[groq]"         # Groq (ultra-fast inference)
pip install "apex-rag[ollama]"       # Ollama (local models)
pip install "apex-rag[gemini]"       # Google Gemini

# Infrastructure
pip install "apex-rag[web]"          # FastAPI REST server + Gradio UI
pip install "apex-rag[postgres]"     # PostgreSQL backend (asyncpg)
pip install "apex-rag[vectors]"      # Dense vector embeddings (sentence-transformers)
pip install "apex-rag[telemetry]"    # OpenTelemetry OTLP exporter
pip install "apex-rag[docling]"      # Advanced document parsing (Docling)
```

**Requirements:** Python 3.10, 3.11, 3.12, or 3.13

---

## ⚡ Quick Start

```python
import asyncio
from apex_rag import ApexIndex

async def main():
    # Initialize with any supported LLM provider
    async with await ApexIndex.create(provider="openai", model="gpt-4o") as index:

        # Ingest a document — converts to AST, builds graph, indexes
        doc_id = await index.ingest("annual_report.pdf")
        print(f"Ingested: {doc_id}")

        # Query — runs Planner → Navigator → Critic agent loop
        answer = await index.query("What was the Q3 revenue change?", doc_id)

        print(answer.answer_text)
        print(f"Confidence: {answer.coverage_guarantee * 100:.1f}%")
        print(f"Supporting evidence packets: {answer.prediction_set_size}")

asyncio.run(main())
```

### Supported LLM Providers

```python
# OpenAI (default)
await ApexIndex.create(provider="openai", model="gpt-4o")

# Anthropic Claude
await ApexIndex.create(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Groq (fast inference)
await ApexIndex.create(provider="groq", model="llama-3.1-70b-versatile")

# Ollama (local, no API key)
await ApexIndex.create(provider="ollama", model="llama3.1")

# Google Gemini
await ApexIndex.create(provider="gemini", model="gemini-1.5-pro")
```

---

## 📖 API Reference

### Ingestion

```python
# Ingest a file (PDF, DOCX, MD, TXT, Python source, images)
doc_id = await index.ingest("financial_report.pdf")

# Ingest raw markdown/text directly
doc_id = await index.ingest_text(
    text="# Q3 Report\nRevenue grew by 15%.\n## Details\n...",
    doc_id="report_q3_2025"
)

# Concurrent batch ingestion
doc_ids = await index.ingest_many([
    ("finance_q3", "q3_report.pdf"),
    ("release_v2", "## Release Notes\nNo downtime recorded."),
])
```

### Querying

```python
# Standard agentic query
answer = await index.query("What is the net profit margin?", doc_id)

# Domain-tuned hybrid search (enables FTS5 + LLM with domain-specific freshness decay)
answer = await index.query("Current pricing", doc_id, domain="financial")
# Available domains: "general" (default), "financial", "legal", "analytical"

# Global query across all indexed documents
results = await index.query_global("Summarize all revenue figures")

# Streaming — token-by-token response
async for token in index.stream_query("Compare Q2 and Q3 revenue", doc_id):
    print(token, end="", flush=True)
```

### Document Inspection

```python
# Get the full AST tree for a document
tree = await index.get_tree(doc_id)

# List all indexed documents
docs = await index.list_documents()

# Get document metadata
info = await index.get_document_info(doc_id)

# Delete a document and all its data
await index.delete(doc_id)
```

### Causal Graph

```python
import networkx as nx

# Retrieve the causal knowledge graph built during ingestion
graph: nx.DiGraph = await index.get_causal_graph()

for source, target, data in graph.edges(data=True):
    print(f"[{source}] --({data['type']})--> [{target}]  strength={data['strength']}")
```

---

## 🏢 Enterprise Features

Enterprise features are accessed via the `index.enterprise` property.

### Temporal Querying (Time Travel)

```python
from datetime import datetime, timezone

enterprise = index.enterprise

# Query the document as it was on a specific date
result = await enterprise.temporal_query(
    question="What was the active product pricing?",
    doc_id=doc_id,
    as_of=datetime(2025, 6, 1, tzinfo=timezone.utc)
)
print(result["result"])      # Resolved answer
print(result["provenance"])  # Version history metadata

# Compare two points in time
comparison = await enterprise.temporal_compare(
    question="How did pricing change?",
    doc_id=doc_id,
    date_a=datetime(2025, 1, 1, tzinfo=timezone.utc),
    date_b=datetime(2025, 6, 1, tzinfo=timezone.utc)
)
```

### Role-Based Access Control (RBAC)

```python
from apex_rag import TenantContext

tenant_ctx = TenantContext(
    tenant_id="enterprise-co",
    user_id="user_948",
    roles=["FinanceManager"]
)

# Query is automatically scoped to the user's accessible nodes
answer = await enterprise.role_aware_query(
    question="Summarize executive compensation",
    doc_id=doc_id,
    tenant_context=tenant_ctx
)
print(answer.answer_text)
```

### Version History

```python
# Get version history for a specific node
history = await enterprise.get_version_history(node_id)

# Get full version lineage
lineage = await enterprise.get_version_lineage(node_id)
```

---

## 🛠️ CLI Interface

```bash
# Start the FastAPI REST API server (requires apex-rag[web])
python -m apex_rag serve --port 8000

# Ingest a file
python -m apex_rag ingest financial_report.pdf --doc-id finance-q3

# Query an ingested document
python -m apex_rag query finance-q3 "Compare Q2 and Q3 revenue"

# Stream a query response
python -m apex_rag stream finance-q3 "What is our effective tax rate?"

# List all indexed documents
python -m apex_rag list

# Get document info
python -m apex_rag info finance-q3

# Open interactive REPL session
python -m apex_rag repl

# Run system diagnostic checks
python -m apex_rag doctor
```

---

## 🔗 LangChain Integration

```python
from apex_rag.integrations.langchain import ApexRAGRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = ApexRAGRetriever(index=index, doc_id=doc_id)

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=retriever
)

result = chain.invoke({"query": "What are the key financial risks?"})
print(result["result"])
```

---

## ⚙️ Configuration

ApexRAG is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `APEX_DB_URL` | `sqlite+aiosqlite:///./apex_rag.db` | Database connection URL |
| `APEX_DATA_DIR` | `.` | Data directory for file storage |
| `APEX_API_KEY` | `None` | API key for endpoint authentication |
| `APEX_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `APEX_RATE_LIMIT` | `60/minute` | Request rate limit |
| `APEX_MAX_UPLOAD_MB` | `50` | Max upload file size in MB |
| `APEX_LOG_FORMAT` | `rich` | Log format: `rich` or `json` |
| `APEX_LOG_LEVEL` | `INFO` | Log level |
| `APEX_TRACE_ENABLED` | `true` | Enable agent navigation trace output |
| `APEX_DB_POOL_SIZE` | `10` | Database connection pool size |
| `APEX_DB_MAX_OVERFLOW` | `20` | Max overflow connections |
| `APEX_OLLAMA_TIMEOUT` | `120` | Ollama request timeout (seconds) |

---

## 📄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

### v1.0.4 — Latest
- Stable release aligned with git tag `v1.0.4`.

### v1.0.3
- **`EnterpriseClient`** introduced — temporal queries, RBAC, and version history extracted from `ApexIndex` into `index.enterprise`.
- **API stabilization** — dead parameters removed, exports cleaned to 11 public symbols.
- **Circular import fix** — lazy import on `ApexIndex.enterprise`.

### v1.0.0
- Production-stable release.
- Conformal Prediction confidence guarantees.
- Structural Retrieval Graph (SRG) with typed semantic edges.

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone and set up dev environment
git clone https://github.com/abi6374/apexrag.git
cd apexrag
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

---

## 📄 License

MIT License — Copyright © 2026 G S Abinivas. See [LICENSE](LICENSE) for full text.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/abi6374">G S Abinivas</a>
</p>
