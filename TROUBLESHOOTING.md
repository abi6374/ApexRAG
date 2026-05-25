# 🔧 ApexRAG Troubleshooting Guide

Common issues and their solutions when using ApexRAG.

---

## Installation Issues

### `pip install apex-rag` fails

```bash
# Ensure you have Python 3.10+
python --version

# Upgrade pip and try again
pip install --upgrade pip
pip install apex-rag
```

### ImportError: markitdown not found

`markitdown` is a core dependency and should be installed automatically. If missing:

```bash
pip install markitdown
```

### Optional extras not installing

```bash
# For web dashboard
pip install apex-rag[web]

# For PostgreSQL support
pip install apex-rag[postgres]

# For advanced PDF parsing
pip install apex-rag[docling]

# For all extras
pip install apex-rag[all]
```

---

## Database Issues

### SQLite: "database is locked"

ApexRAG uses WAL mode for SQLite, which handles concurrent reads well. If you see lock errors:

1. Ensure you're not sharing the DB file across processes
2. Check that all connections are properly closed
3. The lock is typically temporary — retry the operation

### PostgreSQL: connection refused

```bash
# Verify PostgreSQL is running
pg_isready

# Check the connection URL format
# Correct: postgresql+asyncpg://user:password@host:5432/dbname
```

### Alembic migrations not working

```bash
# Install migrations extra
pip install apex-rag[migrations]

# Initialize Alembic in your project
alembic init alembic
# Configure alembic.ini with your DB URL
```

---

## Ollama Issues

### "Connection refused" when starting

```bash
# Ensure Ollama is running
ollama serve

# Check the default host (should be http://localhost:11434)
# Customize via APEX_OLLAMA_HOST env var
```

### Model not found

```bash
# Pull the model first
ollama pull llama3.1

# Or use a different model
export APEX_MODEL=phi3
```

### Timeout errors during ingestion

Large documents with many sections can take time to summarize:

```bash
# Increase timeout (default 120s)
export APEX_OLLAMA_TIMEOUT=300

# Reduce concurrency to avoid GPU overload
export APEX_MAX_CONCURRENT_SUMMARIES=4
```

### Out of memory errors

```bash
# Reduce summary concurrency
export APEX_MAX_CONCURRENT_SUMMARIES=2

# Use a smaller model for summarisation
export APEX_SUMMARISER_MODEL=phi3
```

---

## API Server Issues

### Server won't start

```bash
# Check for port conflicts
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac

# Use a different port
python -m apex_rag serve --port 8001
```

### CORS errors in browser

By default, ApexRAG allows all origins (`*`). For production:

```bash
# Restrict to specific origins
export APEX_CORS_ORIGINS="https://myapp.com,https://admin.myapp.com"
```

### API Key authentication

```bash
# Set an API key
export APEX_API_KEY="your-secret-key"

# Use it in requests
curl -H "X-API-Key: your-secret-key" http://localhost:8000/health
```

### Rate limiting

Default rate limit is 60 requests/minute. Adjust:

```bash
export APEX_RATE_LIMIT="120/minute"
```

---

## Query Issues

### Agent returns "No results found"

1. Check that the document was ingested correctly:
   ```bash
   python -m apex_rag list
   ```

2. Verify the document structure:
   ```bash
   curl http://localhost:8000/documents/{doc_id}/stats
   ```

3. Try a different query phrasing — the agent works best with specific questions
4. Disable verification for broader results:
   ```python
   result = await index.query("question", doc_id, verify_leaves=False)
   ```

### Slow queries

1. Reduce verification overhead:
   ```bash
   export APEX_VERIFY=false
   ```

2. Use a faster verifier model:
   ```bash
   export APEX_VERIFIER_MODEL=phi3
   ```

3. The semantic cache will speed up repeated queries automatically

### Global query returns no results

Global queries search across all documents. If you have many documents, the LLM may miss the right one:

1. The agent first uses FTS5 keyword matching to narrow candidates
2. Then uses LLM-based selection
3. Try being more specific in your query

---

## Logging & Debugging

### Enable verbose logging

```bash
export APEX_LOG_LEVEL=DEBUG
```

### Use JSON log format (for production)

```bash
export APEX_LOG_FORMAT=json
```

### Disable navigation trace

```bash
export APEX_TRACE_ENABLED=false
```

---

## Performance Issues

### Ingestion is slow

```bash
# Increase summary parallelism (if GPU/CPU can handle it)
export APEX_MAX_CONCURRENT_SUMMARIES=20

# Skip summaries for faster ingestion (less accurate queries)
# Pass synthesize_summaries=False to ingest()
```

### Memory usage is high

```bash
# Reduce connection pool size
export APEX_DB_POOL_SIZE=5
export APEX_DB_MAX_OVERFLOW=10

# Reduce summary concurrency
export APEX_MAX_CONCURRENT_SUMMARIES=4
```

---

## File Upload Issues

### "File too large"

```bash
# Increase the upload limit (default 50 MB)
export APEX_MAX_UPLOAD_MB=100
```

### "Unsupported file type"

Supported formats: PDF, DOCX, MD, TXT, HTML, PPTX, XLSX

For unsupported formats:
- Convert to PDF or Markdown first
- Or pass the file path directly via the client library (bypasses upload validation)

---

## Getting Help

If you're still stuck:

- 📖 Read the full [README](README.md) and [User Manual](USER_MANUAL.md)
- 🐛 Open an issue on [GitHub](https://github.com/abinivas-17/apex-rag/issues)
- 📧 Check the [CHANGELOG](CHANGELOG.md) for recent changes
