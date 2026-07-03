# 📋 ApexRAG Migration Guide

Guide for migrating between versions of ApexRAG.

---

## Migrating to v1.0 (API Stabilization)

### Summary of Changes

- **Enterprise features extracted**: Temporal queries, RBAC, and version services moved from `ApexIndex` into a dedicated `EnterpriseClient` accessed via `index.enterprise`
- **Dead parameters removed**: `source_date`, `root_node_id`, and `synthesize` removed from public API
- **Exports cleaned**: `apex_rag.__init__` exports reduced from 21 to 11 symbols; removed symbols raise helpful `ImportError` with migration guidance
- **`get_nodes()` removed**: Exact duplicate of `get_tree()`, now removed
- **Hybrid search API unified**: `hybrid=True` replaced by `domain="financial"` parameter

### Breaking Changes

#### 1. Enterprise Features Moved to `EnterpriseClient`

Temporal queries, version history, RBAC, and access control are no longer methods on `ApexIndex`. They now live on a dedicated `EnterpriseClient` accessed through the `index.enterprise` property.

**Before**:
```python
# Old way — methods on ApexIndex
result = await index.temporal_query("What is revenue?", doc_id, as_of=datetime(...))
versions = await index.get_version_history(node_id)
answer = await index.role_aware_query("Net profit?", doc_id, ctx)

# Old properties
resolver = index.version_resolver
service = index.temporal_reasoning
ac = index.access_control
```

**After**:
```python
# New way — via index.enterprise
enterprise = index.enterprise
result = await enterprise.temporal_query("What is revenue?", doc_id, as_of=datetime(...))
versions = await enterprise.get_version_history(node_id)
lineage = await enterprise.get_version_lineage(node_id)
answer = await enterprise.role_aware_query("Net profit?", doc_id, ctx)

# Properties also on EnterpriseClient
resolver = enterprise.version_resolver
service = enterprise.temporal_reasoning
ac = enterprise.access_control

# temporal_compare kept as deprecated wrapper on ApexIndex
result = await index.temporal_compare(question, doc_id, date_a, date_b)
# ^ Still works but emits deprecation — use index.enterprise.temporal_compare() instead
```

**Action Required**: Replace:
| Old `ApexIndex` method | New `index.enterprise` path |
|---|---|
| `index.temporal_query(...)` | `index.enterprise.temporal_query(...)` |
| `index.get_version_history(node_id)` | `index.enterprise.get_version_history(node_id)` |
| `index.get_version_lineage(node_id)` | `index.enterprise.get_version_lineage(node_id)` |
| `index.role_aware_query(...)` | `index.enterprise.role_aware_query(...)` |
| `index.temporal_compare(...)` | `index.enterprise.temporal_compare(...)` (backward-compat kept) |
| `index.version_resolver` | `index.enterprise.version_resolver` |
| `index.temporal_reasoning` | `index.enterprise.temporal_reasoning` |
| `index.access_control` | `index.enterprise.access_control` |

#### 2. Dead Parameters Removed

Three public methods had dead parameters that were marked `# noqa: ARG002` — they accepted values but **never used them**. These have been removed.

| Method | Removed Parameter | Migration |
|---|---|---|
| `ingest_file()` / `ingest()` | `source_date` | Remove the argument — `source_date` is extracted from file metadata by the parser automatically. For text ingestion, `ingest_text()` still accepts `source_date`. |
| `query()` | `root_node_id` | Include the target section name in the question instead (e.g., `"In the Financials section, what is net profit?"`). |
| `query_global()` | `synthesize` | Remove the argument — synthesis is always applied. |

**Before**:
```python
await index.ingest("report.pdf", source_date=datetime(2024, 1, 1))
result = await index.query("Net profit?", doc_id, root_node_id="abc-123")
result = await index.query_global("Revenue?", synthesize=True)
```

**After**:
```python
await index.ingest("report.pdf")  # source_date removed
result = await index.query("In the Financials section, what is net profit?", doc_id)  # root_node_id removed
result = await index.query_global("Revenue?")  # synthesize removed
```

#### 3. `get_nodes()` Removed

The `get_nodes()` method was an exact duplicate of `get_tree()`. Use `get_tree()` instead:

```python
# Before
nodes = await index.get_nodes(doc_id)

# After (same result)
nodes = await index.get_tree(doc_id)
```

#### 4. Hybrid Search API Changed

The `hybrid=True` parameter on `query()` has been replaced by the `domain` parameter, which automatically enables the appropriate hybrid search strategy based on the content domain.

**Before**:
```python
result = await index.query("Revenue?", doc_id, hybrid=True)
```

**After**:
```python
result = await index.query("Revenue?", doc_id, domain="financial")
```

Available domains: `"general"` (default, agentic-only), `"financial"`, `"legal"`, `"analytical"` (all enable hybrid search with domain-tuned freshness decay).

#### 5. `__init__.py` Exports Cleaned

The top-level `apex_rag` package now exports only **11 symbols** (down from 21). Removed symbols raise a helpful `ImportError` with migration guidance when you try to import them.

**Removed from `apex_rag` top level**:

| Symbol | Correct Import Path |
|---|---|
| `OpenAIProvider` | `from apex_rag.providers import OpenAIProvider` |
| `AnthropicProvider` | `from apex_rag.providers import AnthropicProvider` |
| `GroqProvider` | `from apex_rag.providers import GroqProvider` |
| `OllamaProvider` | `from apex_rag.providers import OllamaProvider` |
| `ASTNode` | `from apex_rag.core.ast.models import ASTNode` |
| `ASTNodeMetadata` | `from apex_rag.core.ast.models import ASTNodeMetadata` |
| `ASTNavigationResult` | `from apex_rag.retrieval.agentic.navigator import ASTNavigationResult` |
| `ApexRAGRetriever` | `from apex_rag.integrations.langchain import ApexRAGRetriever` |
| `VisionAdapter` | `from apex_rag.retrieval.vision import VisionAdapter` |
| `ImageParser` | `from apex_rag.retrieval.vision import ImageParser` |

**Remaining exports**: `ApexIndex`, `LLMProvider`, `TenantContext`, `ApexAnswer`, `EvidencePacket`, `ApexRAGError`, `AuthenticationError`, `ConfigurationError`, `DocumentNotFoundError`, `FileValidationError`, `StorageError`, `__version__`.

---

## Earlier Migration: 0.1.8

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

---

## Future Plans

### Upcoming
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

Note: Database schema is compatible across versions. No migration needed for database.
