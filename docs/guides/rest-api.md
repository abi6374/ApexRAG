# REST API

ApexRAG ships with a production-grade FastAPI server for remote access.

## Starting the Server

```bash
pip install apex-rag[web]
python -m apex_rag serve --host 0.0.0.0 --port 8000
```

Or using the Makefile:

```bash
make serve        # Development server
make serve-reload # With hot-reload
```

## Interactive Docs

Once running, visit:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Authentication

If `APEX_API_KEY` is set, include it in every request:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/health
```

## Endpoints

### Health Checks

```bash
# Liveness probe
GET /health
# Response: {"status": "healthy", "started": true}

# Readiness probe
GET /health/ready
# Response: {"status": "healthy", "db": true, "ollama": true, "issues": []}
```

### Document Ingestion

```bash
# Upload a file
curl -X POST http://localhost:8000/documents/ingest/file \
  -F "file=@report.pdf" \
  -F "doc_id=annual_report"

# Ingest raw text
curl -X POST http://localhost:8000/documents/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "notes", "text": "# Meeting Notes\nContent...", "synthesize_summaries": true}'
```

### Querying

```bash
# Query a single document
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "annual_report", "question": "What was Q3 revenue?"}'

# Stream query with real-time trace
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "annual_report", "question": "What was Q3 revenue?"}'

# Query across all documents
curl -X POST http://localhost:8000/query/global \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our total revenue?"}'
```

### Document Management

```bash
# List all documents
GET /documents

# Get document stats
GET /documents/{doc_id}/stats

# Get tree structure
GET /documents/{doc_id}/tree

# Export nested tree
GET /documents/{doc_id}/export

# Get page index
GET /documents/{doc_id}/index

# Search page index
POST /documents/{doc_id}/search?q=revenue

# Delete document
DELETE /documents/{doc_id}
```

### Dashboard

```bash
# Open in browser
GET /
GET /documents/{doc_id}/index/page
```

## Rate Limiting

The API has built-in rate limiting (configurable via `APEX_RATE_LIMIT`).
When exceeded, the server returns HTTP 429 with a helpful error message.

## Error Format

All errors follow a consistent format:

```json
{
  "code": "APEX_100",
  "message": "Document not found.",
  "hint": "Use index.list_documents() to see available documents."
}
```

## Deployment

For production deployment, see the [Docker Guide](../deployment/docker.md)
and [Production Checklist](../deployment/production-checklist.md).
