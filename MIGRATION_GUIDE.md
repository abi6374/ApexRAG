# 📋 ApexRAG Migration Guide

Guide for migrating between versions of ApexRAG.

---

## Migrating to 0.1.8

### Summary of Changes

- **Optional dependency extras**: Web server, Postgres, and Docling deps are now optional
- **New CLI**: `python -m apex_rag` with serve/ingest/query/list/info commands
- **API key authentication**: Optional bearer token for API endpoints
- **Health check endpoints**: `/health` and `/health/ready`
- **Rate limiting**: Configurable in-memory rate limiter
- **File upload validation**: Size and MIME type enforcement
- **Templated UI**: HTML extracted to templates directory
- **JSON logging**: New `APEX_LOG_FORMAT=json` option for production
- **Configurable settings**: All via environment variables
- **Cleaner public API**: `__all__` now only exposes user-facing classes

### Breaking Changes

#### 1. Dependency Extras

**Before**: `pip install apex-rag` installed everything (including FastAPI, uvicorn, etc.)
**After**: `pip install apex-rag` installs only core dependencies (markitdown, sqlalchemy, ollama, rich)

**Action Required**: If using the web server, install with extras:
```bash
# Old way (still works, but installs more than needed)
pip install apex-rag

# New recommended way
pip install apex-rag[web]    # For API server
pip install apex-rag[all]    # For everything
```

#### 2. CORS Configuration

**Before**: CORS was hardcoded to `allow_origins=["*"]`
**After**: CORS is configurable via `APEX_CORS_ORIGINS` env var (defaults to `["*"]`)

**Action Required**: For production, set:
```bash
export APEX_CORS_ORIGINS="https://myapp.com"
```

#### 3. Query Cache Changes

**Before**: Cache only matched exact queries
**After**: Cache supports substring matching (e.g., "Q3 revenue" matches "What is Q3 revenue?")
Cache also now has a 100-entry LIMIT for performance

**Action Required**: No action needed — fully backward-compatible.

#### 4. `__init__.py` Exports

**Before**: `from apex_rag import *` exported all internal classes
**After**: `from apex_rag import *` only exports user-facing API

**Action Required**: If your code uses `from apex_rag import StorageEngine` or similar, it still works — internal classes remain importable. Only `import *` behavior changed.

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_API_KEY` | `None` | API key for endpoint authentication |
| `APEX_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `APEX_RATE_LIMIT` | `60/minute` | Request rate limit |
| `APEX_MAX_UPLOAD_MB` | `50` | Max upload file size in MB |
| `APEX_LOG_FORMAT` | `rich` | Log format: `rich` or `json` |
| `APEX_LOG_LEVEL` | `INFO` | Log level |
| `APEX_OLLAMA_TIMEOUT` | `120` | Ollama request timeout in seconds |
| `APEX_DB_POOL_SIZE` | `10` | Database connection pool size |
| `APEX_DB_MAX_OVERFLOW` | `20` | Max overflow connections |
| `APEX_TRACE_ENABLED` | `true` | Enable navigation trace output |
| `APEX_DATA_DIR` | `.` | Data directory path |

### New CLI Commands

```bash
# Start the API server
python -m apex_rag serve --host 0.0.0.0 --port 8000

# Ingest a document
python -m apex_rag ingest report.pdf

# Query a document
python -m apex_rag query <doc_id> "What is the revenue?"

# Query across all documents
python -m apex_rag global-query "What are the key findings?"

# List indexed documents
python -m apex_rag list

# Show system info
python -m apex_rag info
```

### Deprecations

None in this release.

---

## Future Plans

### Upcoming (0.2.0)
- OpenTelemetry instrumentation
- Prometheus metrics
- Circuit breaker for Ollama calls
- Async generator-based streaming ingestion

---

## Rollback Instructions

If you need to rollback to a previous version:

```bash
pip install apex-rag==0.1.7
```

Note: The 0.1.7 database schema is compatible with 0.1.8. No migration needed.
