# 📘 ApexRAG User Manual

Welcome to the definitive guide for ApexRAG. This manual covers everything from core concepts to production-grade deployment strategies.

## 🧠 Core Concepts

### Structural Navigation vs. Vector Search
Traditional RAG relies on "Top-K" retrieval from a vector database. This often fails for complex documents where semantics are spread across sections. 
ApexRAG treats a document as a **decision tree**. By navigating this tree, the agent maintains the hierarchical context (e.g., knowing that "Revenues" belongs to "Q3" and not "Q1").

### The Semantic Map
During ingestion, ApexRAG generates a ≤30-word summary for every node (heading). These summaries form a **Semantic Map** that acts as a signpost for the navigation agent.

---

## 📥 Ingestion Pipeline

ApexRAG supports multiple backends for converting documents to the structural tree:

1.  **MarkItDown:** Fast, reliable conversion for standard PDFs, Word docs, and Markdown.
2.  **Docling (Recommended for Enterprise):** Uses IBM's Docling for advanced OCR, table extraction, and layout-aware parsing.

### Configuration
```python
# Configure ingestion parameters on the settings singleton:
from apex_rag.config import settings

settings.parser_backend = "docling"
settings.max_concurrent_summaries = 8
settings.summariser_model = "phi3"

# Initialize client facade with applied settings:
index = await ApexIndex.create()
```

---

## 🔍 Navigation Agent

The agent follows a **Backtrack-and-Verify** loop:
1.  **Explore:** Evaluate child node summaries.
2.  **Recurse:** Move into the most promising node.
3.  **Verify:** At the leaf, perform a strict "Does this answer the question?" check.
4.  **Backtrack:** If verification fails or a path is exhausted, return to the parent and try the next candidate.

### Hybrid Search
For large document sets, ApexRAG uses **Hybrid Root Search**. It first performs a BM25 keyword search over all document roots and top-level headings to prune the search space before engaging the LLM agent.

---

## 🚀 Production Deployment

### Database Support
For production, use **PostgreSQL** with the `asyncpg` driver for high concurrency.
```bash
export APEX_DB_URL="postgresql+asyncpg://user:pass@localhost/apex_rag"
```

### Environment Variables

All settings are configured via environment variables. Copy the `.env.example` file from the repository root and adjust as needed.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `APEX_DB_URL` | SQLAlchemy async database URL (SQLite or PostgreSQL). | `sqlite+aiosqlite:///apex.db` |
| `APEX_DB_ECHO` | Log all SQL queries to stdout (dev only). | `false` |
| `APEX_DB_POOL_SIZE` | Connection pool size (PostgreSQL only). | `10` |
| `APEX_DB_MAX_OVERFLOW` | Max overflow connections (PostgreSQL only). | `20` |
| `APEX_OLLAMA_HOST` | URL for your Ollama instance. | `http://localhost:11434` |
| `APEX_OLLAMA_TIMEOUT` | Ollama request timeout in seconds. | `120` |
| `APEX_MODEL` | LLM for navigation decisions. | `llama3.1` |
| `APEX_SUMMARISER_MODEL` | Smaller/faster model for summary generation (falls back to `APEX_MODEL`). | — |
| `APEX_VERIFIER_MODEL` | Model for leaf verification (falls back to `APEX_MODEL`). | — |
| `APEX_AGGREGATOR_MODEL` | Model for cross-document synthesis (falls back to `APEX_MODEL`). | — |
| `APEX_PARSER_BACKEND` | Document parser: `markitdown`, `docling`, or `plaintext`. | `markitdown` |
| `APEX_MAX_CONCURRENT_SUMMARIES` | Max parallel LLM calls during ingestion. | `10` |
| `APEX_VERIFY` | Enable/disable leaf verification. | `true` |
| `APEX_CORS_ORIGINS` | Comma-separated allowed CORS origins (`*` for all). | `*` |
| `APEX_API_KEY` | Bearer token for API authentication (empty = disabled). | — |
| `APEX_RATE_LIMIT` | API request rate limit. | `60/minute` |
| `APEX_MAX_UPLOAD_MB` | Max upload file size in MB. | `50` |
| `APEX_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `APEX_LOG_FORMAT` | Log output format (`rich` or `json`). | `rich` |
| `APEX_TRACE_ENABLED` | Enable colour-coded navigation trace output. | `true` |
| `APEX_DATA_DIR` | Data directory for file storage. | `.` |

### Docker Setup

Use the provided `docker-compose.yml` to spin up a complete stack:

- **Ollama:** LLM inference server
- **Postgres:** Robust metadata and tree storage
- **ApexRAG API:** REST endpoints and Dashboard

**Quick start with Makefile:**
```bash
make docker-up      # Start all services in background
make docker-logs    # Tail logs
make docker-down    # Stop all services
```

**Or directly with Docker Compose:**
```bash
docker compose up -d
docker compose logs -f
docker compose down
```

---

## 🖥️ CLI Reference

ApexRAG ships with a built-in command-line interface for common operations.

```bash
# Start the API server (requires pip install apex-rag[web])
python -m apex_rag serve [--host 0.0.0.0] [--port 8000] [--reload]

# Ingest a document
python -m apex_rag ingest <file> [--doc-id <id>] [--no-summaries]

# Query an indexed document
python -m apex_rag query <doc_id> "<question>"

# Query across all documents
python -m apex_rag global-query "<question>"

# List all indexed documents
python -m apex_rag list

# Show system info
python -m apex_rag info
```

**Makefile shortcuts:**
- `make serve` → `python -m apex_rag serve`
- `make serve-reload` → `python -m apex_rag serve --reload`
- `make info` → `python -m apex_rag info`
- `make list` → `python -m apex_rag list`

---

## 🛠️ API Reference

### `POST /query`
The primary endpoint for natural language queries.
**Request Body:**
```json
{
  "doc_id": "report_2024",
  "question": "What is the net profit?"
}
```

### `GET /documents/{doc_id}/tree`
Returns the full structural tree of a document in JSON format.

---

## 📈 Performance Tuning

1.  **GPU Acceleration:** Ensure Ollama is running on a GPU. Navigation speed is directly tied to LLM token throughput.
2.  **Context Window:** For extremely long leaf nodes, ApexRAG automatically performs **Robust Chunking** to ensure the content fits within the LLM's context window.
3.  **Model Selection:** 
    - `llama3.1` (8B) is the sweet spot for accuracy/speed.
    - `phi3` or `mistral` are excellent for the `summariser_model`.
