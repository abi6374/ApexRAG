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
await ApexIndex.create(
    parser_backend="docling",  # or "markitdown"
    max_concurrent_summaries=8, # Concurrency for ingestion
    summariser_model="phi3",    # Use a smaller, faster model for summaries
)
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
| Variable | Description | Default |
| :--- | :--- | :--- |
| `APEX_MODEL` | The LLM used for navigation decisions. | `llama3.1` |
| `APEX_OLLAMA_HOST` | URL for your Ollama instance. | `http://localhost:11434` |
| `APEX_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, ERROR). | `INFO` |

### Docker Setup
Use the provided `docker-compose.yml` to spin up a complete stack:
- **Ollama:** LLM Inference.
- **Postgres:** Robust metadata and tree storage.
- **ApexRAG API:** REST endpoints and Dashboard.

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
