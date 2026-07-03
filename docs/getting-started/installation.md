# Installation

## Core Library

```bash
pip install apex-rag
```

This installs only core dependencies: markitdown, sqlalchemy, aiosqlite, openai, pydantic, rich, typing-extensions, networkx, opentelemetry-api, opentelemetry-sdk, and langchain-core.

## Optional Extras

| Extra | What You Get | Install Command |
|-------|-------------|-----------------|
| `[web]` | FastAPI REST API + dashboard UI | `pip install apex-rag[web]` |
| `[postgres]` | PostgreSQL support | `pip install apex-rag[postgres]` |
| `[vectors]` | Vector embeddings (sentence-transformers) | `pip install apex-rag[vectors]` |
| `[telemetry]` | OpenTelemetry tracing + OTLP export | `pip install apex-rag[telemetry]` |
| `[dev]` | Development tools (pytest, mypy, ruff, tox) | `pip install apex-rag[dev]` |
| `[all]` | Everything above | `pip install apex-rag[all]` |

## Development Install

```bash
git clone https://github.com/abi6374/apexrag.git
cd apex-rag

# Option A: use the Makefile (recommended)
make install            # Installs with dev + web extras
make install-all        # Installs with ALL extras (postgres, docling, etc.)

# Option B: pip directly
pip install -e ".[dev,web]"
```

A full reference of Makefile commands is available in the [Contributing Guide](../contributing.md) and on GitHub as [`CONTRIBUTING.md`](https://github.com/abi6374/apexrag/blob/main/CONTRIBUTING.md).

## Docker (Production)

```bash
# Using the Makefile
docker compose up -d    # or: make docker-up
```

This starts the API server on port 8000 with Ollama for LLM inference.
