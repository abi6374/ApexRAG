FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the package
COPY pyproject.toml README.md ./
COPY apex_rag/ ./apex_rag/
RUN pip install --no-cache-dir -e ".[web]"

# ── Runtime image ───────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed package from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/apex_rag /app/apex_rag

EXPOSE 8000

# Default: serve the REST API
CMD ["uvicorn", "apex_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
