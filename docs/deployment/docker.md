# Docker Deployment

Deploy ApexRAG in production using Docker Compose.

## Quick Start

```bash
# Clone and deploy
git clone https://github.com/abi6374/apexrag.git
cd apex-rag
docker compose up -d

# Or using the Makefile
make docker-up
```

This starts:
- **ApexRAG API** on port 8000
- **Ollama** on port 11434 (with GPU acceleration)
- **Model Puller** (one-shot init container — downloads the model)
- **Postgres** (optional, enabled via profile)
- **Prometheus + Grafana** (optional, enabled via profile)

## Services

### API Server

```yaml
services:
  api:
    build: .
    ports:
      - "${APEX_PORT:-8000}:8000"
    environment:
      APEX_DB_URL: "${APEX_DB_URL:-sqlite+aiosqlite:///data/apex.db}"
      APEX_OLLAMA_HOST: "${APEX_OLLAMA_HOST:-http://ollama:11434}"
      APEX_MODEL: "${APEX_MODEL:-llama3.1}"
      APEX_API_KEY: "${APEX_API_KEY:-}"
      APEX_LOG_LEVEL: "${APEX_LOG_LEVEL:-INFO}"
    volumes:
      - apex_data:/app/data
    depends_on:
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

### Ollama (LLM)

GPU acceleration is configured via device reservations:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### PostgreSQL (Optional)

```bash
docker compose --profile postgres up -d
```

Then set the environment variable:

```bash
export APEX_DB_URL="postgresql+asyncpg://apexrag:apexrag_secret@postgres:5432/apexrag"
```

### Monitoring Stack (Optional)

```bash
docker compose --profile monitoring up -d
```

Starts Prometheus (port 9090) and Grafana (port 3000).

## Environment Configuration

Create a `.env` file in the project root:

```bash
# .env
APEX_MODEL=llama3.1
APEX_API_KEY=your-secret-key
APEX_LOG_LEVEL=INFO
APEX_DB_URL=postgresql+asyncpg://apexrag:apexrag_secret@postgres:5432/apexrag
```

## Commands

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild after code changes
docker compose build
docker compose up -d

# With monitoring
docker compose --profile monitoring up -d
```

## Health Checks

```bash
# API health
curl http://localhost:8000/health

# Readiness (checks DB + Ollama)
curl http://localhost:8000/health/ready

# Dashboard
open http://localhost:8000
```

## Production Considerations

1. **Use PostgreSQL** — SQLite is not suitable for concurrent multi-worker deployments.
2. **Set APEX_API_KEY** — Always enable authentication in production.
3. **Configure CORS** — Restrict `APEX_CORS_ORIGINS` to your frontend domain.
4. **Mount volumes** — Persistent storage for DB and Ollama models.
5. **Use a reverse proxy** — Place Nginx or Traefik in front for SSL termination.
6. **Set resource limits** — Configure CPU/memory limits in Docker Compose.
