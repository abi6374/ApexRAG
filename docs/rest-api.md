# REST API Reference

ApexRAG ships with a production-grade **FastAPI** REST API featuring:

- API key authentication via `X-API-Key` header
- Configurable CORS origins
- Rate limiting (sliding window, in-memory)
- Health checks (liveness + readiness)
- SSE streaming for real-time agent traces
- Prometheus metrics

**Start the server:**

```bash
# From the project root
uvicorn apex_rag.api:app --host 0.0.0.0 --port 8000

# Or via the CLI
apex-rag api --host 0.0.0.0 --port 8000
```

---

## System

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/health` | Liveness probe — returns OK if the app is running | No |
| `GET` | `/health/ready` | Readiness probe — checks DB connectivity and Ollama status | No |
| `GET` | `/metrics` | Prometheus-compatible metrics (histograms, counters) | No |
| `GET` | `/accuracy` | Aggregate per-query accuracy metrics (precision, recall, F1) | No |
| `GET` | `/accuracy/recent` | Recent query accuracy records, optional `min_precision` filter | No |
| `GET` | `/accuracy/prometheus` | Accuracy metrics in Prometheus exposition format | No |

### `/health`

```json
{"status": "healthy", "app": "apex-rag", "started": true}
```

### `/health/ready`

```json
{
  "status": "healthy",
  "db": true,
  "ollama": true,
  "issues": []
}
```

### `/metrics`

Returns Prometheus text format with the following metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `apex_rag_uptime_seconds` | gauge | Server uptime |
| `apex_rag_total_queries` | counter | Total queries processed |
| `apex_rag_llm_calls` | counter | Total LLM calls |
| `apex_rag_cache_hits` | counter | Semantic cache hits |
| `apex_rag_cache_misses` | counter | Semantic cache misses |
| `apex_rag_cache_hit_rate` | gauge | Hit rate ratio |
| `apex_rag_retrieval_latency_ms` | histogram | Retrieval latency |
| `apex_rag_planner_latency_ms` | histogram | Planner latency |
| `apex_rag_navigator_latency_ms` | histogram | Navigator latency |
| `apex_rag_verifier_latency_ms` | histogram | Verifier latency |
| `apex_rag_critic_latency_ms` | histogram | Critic latency |
| `apex_rag_tenant_queries_total` | counter (labelled) | Tenant-scoped queries |

---

## Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/ingest/file` | Upload and ingest a document file |
| `POST` | `/documents/ingest/text` | Ingest raw Markdown/plain text |
| `GET` | `/documents` | List all indexed document IDs |
| `GET` | `/documents/{doc_id}/stats` | Document statistics |
| `GET` | `/documents/{doc_id}/tree` | Full node tree as flat JSON list |
| `GET` | `/documents/{doc_id}/export` | Nested tree export (children as sub-arrays) |
| `GET` | `/documents/{doc_id}/index` | Alphabetical page index (JSON) |
| `GET` | `/documents/{doc_id}/index/page` | Visual document index page (HTML) |
| `POST` | `/documents/{doc_id}/search` | Full-text search over page index |
| `DELETE` | `/documents/{doc_id}` | Delete a document and all its data |

### `POST /documents/ingest/file`

Upload a file (PDF, DOCX, MD, TXT, HTML, PPTX, XLSX). Max upload size is configurable via `APEX_MAX_UPLOAD_MB`.

**Request:** `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `UploadFile` | *required* | Document file |
| `doc_id` | `str` | auto-generated | Optional document ID override |
| `synthesize_summaries` | `bool` | `true` | Generate navigation summaries |

**Response:**
```json
{
  "ok": true,
  "doc_id": "doc-abc123",
  "stats": {"total_nodes": 42, "max_depth": 5, "leaf_count": 18}
}
```

### `POST /documents/ingest/text`

**Request:**
```json
{
  "doc_id": "my-doc",
  "text": "# Meeting Notes\n\nQ3 revenue grew 15%...",
  "synthesize_summaries": true
}
```

**Response:** Same format as file ingest.

### `GET /documents`

```json
{
  "documents": ["doc-abc123", "doc-def456"],
  "count": 2
}
```

### `GET /documents/{doc_id}/stats`

```json
{
  "doc_id": "doc-abc123",
  "total_nodes": 42,
  "max_depth": 5,
  "leaf_count": 18
}
```

### `GET /documents/{doc_id}/tree`

Returns all AST nodes as a flat list ordered depth-first.

```json
{
  "doc_id": "doc-abc123",
  "node_count": 42,
  "nodes": [
    {"node_id": "...", "content": "...", "node_type": "HEADING", "depth": 0, ...}
  ]
}
```

### `GET /documents/{doc_id}/export`

Returns a nested JSON structure with `children_nodes` arrays.

### `DELETE /documents/{doc_id}`

```json
{"ok": true, "doc_id": "doc-abc123", "nodes_deleted": 42}
```

---

## Graph

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/documents/{doc_id}/graph` | All knowledge edges for a document as JSON graph |
| `GET` | `/documents/{doc_id}/graph/{projection}` | Edges filtered by DAG projection for a document |
| `GET` | `/graph` | Combined graph across **all** documents |
| `GET` | `/graph/{projection}` | Global graph filtered by DAG projection |

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | `int` | `500` (doc), `1000` (global) | Maximum edges to return |

**DAG Projections:**

| Projection | Description |
|------------|-------------|
| `document` | Structural tree edges (REFINES, SUPPORTS) |
| `entity` | Named entity extraction and linking |
| `citation` | Citation and cross-reference relationships |
| `temporal` | Chronological ordering (SUCCESSOR, PREDECESSOR, VALID_DURING) |
| `version` | Version lineage (VERSION_OF, SUPERSEDES, REPLACED_BY) |
| `policy` | Policy/regulation governance edges |
| `fact` | Fact relationship edges (SUPPORTS, CONTRADICTS, SAME_TOPIC) |
| `reasoning` | Orchestrator reasoning traces (REASONING_CHAIN, DERIVES_FROM, INFERS, USES) |

**Response format (compatible with vis-network, cytoscape.js, d3.js):**

```json
{
  "nodes": [
    {
      "id": "uuid4...",
      "label": "Quarterly Revenue Growth Q3 2025",
      "group": "node",
      "node_type": "HEADING",
      "page_number": 42
    }
  ],
  "edges": [
    {
      "id": "edge-uuid",
      "source": "node-uuid-a",
      "target": "node-uuid-b",
      "type": "SUPPORTS",
      "strength": 0.92,
      "evidence": "Both nodes discuss revenue growth figures...",
      "projections": ["document", "reasoning"]
    }
  ],
  "edge_count": 24,
  "node_count": 18,
  "doc_id": "doc-abc123",
  "projection": "reasoning"
}
```

Node labels are resolved from the AST nodes table — showing the actual heading/content text. Nodes fall back to truncated UUIDs if the lookup fails.

---

## Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Query a single document (non-streaming) |
| `POST` | `/query/stream` | SSE-streamed query with real-time agent traces |
| `POST` | `/query/global` | Query across all documents (non-streaming) |
| `POST` | `/query/global/stream` | SSE-streamed global query |
| `POST` | `/query/orchestrate/stream` | SSE-streamed multi-agent orchestrator |
| `POST` | `/query/stream/reasoning-graph` | SSE-streamed query + **ReasoningDAG edges** |

### `POST /query`

**Request:**
```json
{
  "doc_id": "doc-abc123",
  "question": "What was Q3 revenue growth?",
  "verify_leaves": true
}
```

**Response:**
```json
{
  "found": true,
  "content": "Q3 revenue grew 15% year-over-year to $2.3B.",
  "node_id": 42,
  "path": "2.1.3",
  "title": "Revenue Overview",
  "verified": true,
  "confidence": 0.97,
  "trace": [[1, "Executive Summary"], [2, "Financials"], [5, "Revenue"]]
}
```

### `POST /query/stream`

Returns SSE events (`text/event-stream`) with real-time navigation trace:

| Event | Description |
|-------|-------------|
| `enter` | Agent entering a node |
| `explore` | Evaluating child sections |
| `choice` | Agent chose a branch |
| `leaf` | Leaf node reached with content preview |
| `verify` | LLM verification result |
| `backtrack` | Backtracking to parent |
| `result` | Final answer |

### `POST /query/stream/reasoning-graph`

Returns SSE events combining real-time agent traces with a final `reasoning_graph` event containing the full ReasoningDAG as `{nodes, edges}` JSON:

| Event | Description |
|-------|-------------|
| `start` | Query started |
| `trace` | Real-time orchestrator trace event |
| `reasoning_graph` | Full ReasoningDAG with enriched nodes |
| `result` | Final answer with coverage metrics |
| `error` | Error occurred |

---

## LLM

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/llm/stream` | Direct LLM streaming (bypasses retrieval) |
| `POST` | `/llm/generate` | Non-streaming LLM generation |

### `POST /llm/stream`

**Request:**
```json
{
  "prompt": "Summarize: ...",
  "temperature": 0.0,
  "max_tokens": 150,
  "images": ["base64..."]
}
```

Returns SSE events: `token` events streamed token-by-token, then `done`.

### `POST /llm/generate`

**Request:** Same as stream but with `temperature`, `max_tokens`, `images`.

**Response:**
```json
{"content": "Generated text response..."}
```

---

## UI

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard — document list, global search, drag-drop ingest, DAG visualization |
| `GET` | `/documents/{doc_id}/index/page` | Visual document index with expandable tree and alphabetical index |
| `GET` | `/docs` | Swagger UI (auto-generated by FastAPI) |
| `GET` | `/redoc` | ReDoc (auto-generated by FastAPI) |

---

## Error Format

All errors follow a consistent JSON structure:

```json
{
  "code": "APEX_404",
  "message": "Document 'doc-xyz' not found.",
  "hint": "Use GET /documents to list available documents."
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `APEX_400` | 401 | Missing/Invalid API key |
| `APEX_401` | 429 | Rate limit exceeded |
| `APEX_403` | 403 | Tenant isolation violation |
| `APEX_404` | 404 | Resource not found |
| `APEX_409` | 409 | Cycle detected in knowledge graph |
| `APEX_413` | 413 | File too large |
| `APEX_415` | 415 | Unsupported file type |
| `APEX_500` | 500 | Internal server error |
| `APEX_503` | 503 | Service unavailable (ApexIndex not initialized) |

---

## Authentication

When `APEX_API_KEY` environment variable is set, all endpoints except `/health`, `/health/ready`, `/docs`, `/redoc`, `/openapi.json`, and `/metrics` require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/documents
```

---

## Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_API_KEY` | *none* | Enables API key authentication |
| `APEX_CORS_ORIGINS` | `["*"]` | Comma-separated CORS origins |
| `APEX_RATE_LIMIT` | `60/minute` | Rate limit string |
| `APEX_MAX_UPLOAD_MB` | `50` | Max upload file size in MB |

---

## SSE Client Example (JavaScript)

```javascript
const response = await fetch('/query/stream/reasoning-graph', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ doc_id: "doc-123", question: "What is Q3 revenue?" })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split('\n\n');
  buffer = lines.pop();

  for (const line of lines) {
    if (!line.startsWith('data: ')) continue;
    const data = JSON.parse(line.substring(6));

    switch (data.event) {
      case 'trace':
        console.log('[Trace]', data.trace.event_name, data.trace.data);
        break;
      case 'reasoning_graph':
        console.log('Graph:', data.node_count, 'nodes,', data.edge_count, 'edges');
        break;
      case 'result':
        console.log('Answer:', data.content);
        break;
    }
  }
}
```
