# Configuration

ApexRAG is configured entirely through environment variables. There are no config files to manage.

## Quick Reference

```bash
# Copy the template and customize
cp .env.example .env

# Or set variables directly
export APEX_DB_URL="postgresql+asyncpg://user:pass@localhost:5432/apex_rag"
export APEX_MODEL="llama3.1"
export APEX_API_KEY="your-secret-key"
```

## All Environment Variables

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_DB_URL` | `sqlite+aiosqlite:///apex.db` | SQLAlchemy async database URL. Use `postgresql+asyncpg://...` for production. |
| `APEX_DB_ECHO` | `false` | Log all SQL queries to stdout (development only). |
| `APEX_DB_POOL_SIZE` | `10` | Connection pool size (PostgreSQL only; ignored for SQLite). |
| `APEX_DB_MAX_OVERFLOW` | `20` | Max overflow connections (PostgreSQL only; ignored for SQLite). |

### LLM / Ollama

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_OLLAMA_HOST` | `http://localhost:11434` | URL for your Ollama instance. |
| `APEX_OLLAMA_TIMEOUT` | `120` | Request timeout in seconds. |
| `APEX_MODEL` | `llama3.1` | LLM for navigation decisions (e.g., `llama3.1`, `mistral`, `phi3`). |
| `APEX_SUMMARISER_MODEL` | *(falls back to `APEX_MODEL`)* | Smaller/faster model for summary generation during ingestion. |
| `APEX_VERIFIER_MODEL` | *(falls back to `APEX_MODEL`)* | Model for leaf verification. Use a cheaper model here. |
| `APEX_AGGREGATOR_MODEL` | *(falls back to `APEX_MODEL`)* | Model for cross-document answer synthesis. |

### Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_PARSER_BACKEND` | `markitdown` | Document parser: `markitdown`, `docling`, or `plaintext`. |
| `APEX_MAX_CONCURRENT_SUMMARIES` | `10` | Max parallel LLM calls during summary generation. |
| `APEX_VERIFY` | `true` | Enable/disable leaf verification during queries. |

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins. |
| `APEX_API_KEY` | *(empty)* | Bearer token for API authentication. Leave empty to disable. |
| `APEX_RATE_LIMIT` | `60/minute` | API request rate limit. |
| `APEX_MAX_UPLOAD_MB` | `50` | Max upload file size in MB. |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `APEX_LOG_FORMAT` | `rich` | Log output format: `rich` (human-readable) or `json` (for log aggregators). |
| `APEX_TRACE_ENABLED` | `true` | Enable colour-coded navigation trace output to stderr. |

### File Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_DATA_DIR` | `.` | Data directory for file storage. |

## .env File

ApexRAG supports loading configuration from a `.env` file in the project root:

```bash
# .env example
APEX_DB_URL=postgresql+asyncpg://user:pass@localhost:5432/apex_rag
APEX_MODEL=llama3.1
APEX_API_KEY=sk-my-secret-key
APEX_LOG_LEVEL=DEBUG
APEX_TRACE_ENABLED=false
```

> **Note:** The `.env` file is loaded automatically by the CLI and API server.
> When using ApexRAG as a library in your own application, you'll need to handle
> environment variable loading yourself (e.g., via `python-dotenv`).

## Configuration Priority

1. Environment variables (highest priority)
2. `.env` file (if present)
3. Built-in defaults (lowest priority)
