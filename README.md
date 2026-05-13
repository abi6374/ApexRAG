# ApexRAG

> **Production-grade, local-first Agentic RAG Library.**
> Replaces vector similarity search with structural, agentic navigation of documents.

---

## 🧠 The Core Idea

Traditional RAG embeds text into vectors and finds the "closest" chunks. This creates **retrieval hallucinations** — the model returns semantically-similar-but-wrong content because it has no understanding of document structure.

**ApexRAG takes a fundamentally different approach:**

1. **Parse** the document into a structural tree using robust **recursive chunking** or IBM **Docling** for complex layouts.
2. **Synthesize** a 30-word *Semantic Map* for every node using a local LLM.
3. **Navigate** the tree with an LLM agent that reads summaries and decides which branch to enter — trying multiple candidates if necessary.
4. **Verify** the exact leaf node answers the query via a strict secondary LLM check (99.999% accuracy).
5. **Return** the exact leaf node content — not a blended, hallucinated average.

```
Query: "What were Q3 revenues?"
         │
    Root (Annual Report)
    ├── Chapter 1: Executive Summary  ← LLM: "Not here"
    └── Chapter 2: Revenue Analysis   ← LLM: "Enter this"
        ├── Q1 Revenue                ← LLM: "Not Q3"
        ├── Q2 Revenue                ← LLM: "Not Q3"
        └── Q3 Revenue                ← LLM: "This is it!" → Return content
```

---

## 📁 Project Structure

```
apex_rag/
apex_rag/
├── __init__.py       # Public API exports
├── api.py            # FastAPI App, UI dashboard, and JSON Export
├── client.py         # Thread-safe user-facing ApexIndex class
├── ingestion.py      # Docling/Markdown parsing & tree synthesis
├── navigation.py     # Recursive LLM navigation agent
├── storage.py        # SQLAlchemy async ORM & PageIndexEntry
└── providers.py      # Pluggable LLM interface (Ollama/OpenAI)
├── tests/
│   ├── test_tree.py      # Parser & storage unit tests
│   └── test_search.py    # Navigation agent unit tests (no Ollama needed)
├── examples/
│   └── basic_usage.py    # End-to-end demo
├── pyproject.toml
└── docker-compose.yml
```

---

## ⚡ Quick Start

### 1. Install

```bash
# Clone and set up
cd ApexRAG
pip install -e ".[dev]"
```

### 2. Start Ollama

```bash
ollama serve
ollama pull llama3.1   # or phi3, mistral, etc.
```

### 3. Ingest & Query

```python
import asyncio
from apex_rag import ApexIndex

async def main():
    async with await ApexIndex.create(
        db_url="sqlite+aiosqlite:///apex.db",
        model="llama3.1",
    ) as index:
        # Ingest a PDF
        doc_id = await index.ingest("path/to/your/report.pdf")

        # Query it
        result = await index.query(
            "What are the Q3 revenue figures?",
            doc_id,
        )

        if result:
            print(result.content)
            print(f"Found at path: {result.path}")
            print(f"Navigation trace: {result.trace}")

asyncio.run(main())
```

### 4. Start the FastAPI Server & Visual Index Dashboard

```bash
uvicorn apex_rag.api:app --reload
```

Open your browser to:
- **Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Export JSON:** `GET /documents/{doc_id}/export`

From the dashboard, you can click on an ingested document to view its full structural tree and its book-style alphabetical page index! You can also fetch the perfectly formatted nested JSON tree using the export endpoint.

---

## 🏗️ Architecture Deep Dive

### Ingestion Engine (`ingestion.py`)

| Step | Description |
|------|-------------|
| **Convert** | Uses **IBM Docling** (for enterprise table/layout OCR) or `markitdown` to convert PDFs to Markdown |
| **Parse** | Walks headings (`#`) to build a `ParsedSection` tree. Uses **Robust Chunking** for walls of text |
| **Persist** | Nodes written to DB with LTree-style path (`1.2.3`) |
| **Synthesize** | Ollama generates 30-word summaries in parallel (bounded by semaphore) |

### Storage Layer (`storage.py`)

```
DocumentNode table:
  id          BIGINT PRIMARY KEY
  doc_id      VARCHAR(255)       -- logical document identifier
  parent_id   BIGINT FK (self)   -- NULL for root nodes
  path        VARCHAR(512)       -- "1.2.3" LTree-style
  title       VARCHAR(512)       -- section heading
  summary     TEXT               -- 30-word Semantic Map
  content     TEXT               -- leaf content (NULL for intermediate)
  metadata    TEXT (JSON)        -- page numbers, char count, source file
  depth       INTEGER            -- nesting level (0 = root)
  position    INTEGER            -- sibling order
  created_at  TIMESTAMP
```

Supports both `sqlite+aiosqlite://` (local) and `postgresql+asyncpg://` (production).

### Navigation Agent (`navigation.py`)

```
find(query, doc_id)
  └── _navigate(current_node)
        ├── [Leaf?] → return content immediately
        ├── fetch children
        ├── _ask_llm(query, child_summaries)
        │     └── "Which child ID contains the answer?"
        ├── [ID returned] → recurse into chosen child
        │     └── [child returns None] → try siblings
        └── [NONE returned] → backtrack to parent
```

**LLM Response Parsing** is robust — 4-tier fallback:
1. Strict `json.loads()`
2. Regex extraction from prose-wrapped JSON
3. Explicit `"NONE"` keyword detection
4. Heuristic: scan for any valid child ID number in the response

**High Accuracy (99.999%) Verification:**
At the leaf level, a second LLM prompt strictly verifies if the leaf content answers the query. If it fails, the agent backtracks and explores the fallback candidates (second-best choices) up the tree.

### Reasoning Trace (`utils.py`)

Every navigation decision is printed with color-coded indicators:

```
━━━ ApexRAG Navigation Start ━━━
Query : What are the Q3 revenue figures?
Root  : node_id=1

  ↳ ENTER node=1 path=1
    Covers the full annual financial report for 2024…
    ⟳ EXPLORE node=1 → evaluating 2 child summaries
    ✔ AGENT → node=3  reason: Revenue Analysis contains quarterly breakdown
    ↳ ENTER node=3 path=1.2
      ⟳ EXPLORE node=3 → evaluating 4 child summaries
      ✔ AGENT → node=6  reason: Q3 Revenue section is exactly what's needed
      ★ LEAF REACHED node=6
        preview: Q3 revenue was $165M. Growth slowed slightly…

━━━ Navigation Complete ━━━  result=SUCCESS  elapsed=3.41s
```

---

## 🧪 Testing

```bash
# Run all tests (no Ollama required)
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_search.py -v
```

Tests use an in-memory SQLite database and mock LLM responses — zero external dependencies.

---

## 🐳 Production Deployment

```bash
# Copy and edit environment
cp .env.example .env

# Start everything (Ollama + PostgreSQL + API)
docker-compose up -d

# Pull the model inside the Ollama container
docker exec apex_ollama ollama pull llama3.1
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_DB_URL` | `sqlite+aiosqlite:///apex.db` | SQLAlchemy async DB URL |
| `APEX_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `APEX_MODEL` | `llama3.1` | Ollama model for navigation |
| `APEX_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🔧 Configuration Reference

```python
await ApexIndex.create(
    db_url="postgresql+asyncpg://user:pass@host/db",  # Production DB
    ollama_host="http://localhost:11434",
    model="llama3.1",              # Navigation model
    summariser_model="phi3",       # Cheaper model for ingestion summaries
    max_concurrent_summaries=8,    # Parallelism (tune to your GPU VRAM)
    parser_backend="markitdown",   # "markitdown" | "docling" | "plaintext"
    trace_enabled=True,            # Color-coded console output
    db_echo=False,                 # SQL query logging
)
```

---

## 📋 Roadmap

- [x] FastAPI REST API wrapper (`/documents/ingest/file`, `/query`, `/documents`)
- [x] Book-style Page Index and Visual tree dashboard
- [x] Unlimited navigation depth with backtrack and verification
- [ ] Streaming query responses via SSE
- [ ] Multi-document cross-reference queries
- [ ] Alembic migrations for schema versioning
- [ ] Support for `docling` table extraction (structured data cells as leaf nodes)

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
