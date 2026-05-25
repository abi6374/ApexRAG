# Production Checklist

A step-by-step checklist for deploying ApexRAG in production.

## Database

- [ ] Use **PostgreSQL** (not SQLite) for concurrent access
- [ ] Set `APEX_DB_URL` to your PostgreSQL connection string
- [ ] Configure `APEX_DB_POOL_SIZE` based on expected concurrency
- [ ] Set `APEX_DB_MAX_OVERFLOW` for burst traffic handling
- [ ] Enable automated backups for the database
- [ ] ApexRAG manages its schema automatically (auto-create on startup)

## Security

- [ ] **Set `APEX_API_KEY`** to a strong, random value
- [ ] Configure `APEX_CORS_ORIGINS` to your frontend domain(s)
- [ ] Set `APEX_RATE_LIMIT` (default: 60/minute)
- [ ] Use HTTPS via a reverse proxy (Nginx, Traefik, Caddy)
- [ ] Never expose Ollama directly to the internet
- [ ] Run dependency audits: `pip-audit`
- [ ] Keep secrets in environment variables, never in code

## LLM Configuration

- [ ] Set `APEX_MODEL` to your production model (e.g., `llama3.1`)
- [ ] Consider a smaller `APEX_SUMMARISER_MODEL` for ingestion speed
- [ ] Test Ollama GPU acceleration (nvidia-smi)
- [ ] Configure `APEX_OLLAMA_TIMEOUT` for large documents
- [ ] Monitor Ollama memory usage (models consume 4-16GB VRAM)

## Performance

- [ ] Tune `APEX_MAX_CONCURRENT_SUMMARIES` to your GPU/CPU capacity
- [ ] Enable `APEX_VERIFY=true` for production accuracy
- [ ] Consider `hybrid=True` for large document sets (>100 pages)
- [ ] Monitor p99 query latency target: <10 seconds
- [ ] Set up Prometheus metrics + Grafana dashboards
- [ ] Configure structured JSON logging (`APEX_LOG_FORMAT=json`)

## Infrastructure

- [ ] Use Docker with the provided `docker-compose.yml`
- [ ] Mount persistent volumes for database and Ollama models
- [ ] Set resource limits (CPU/memory) for all containers
- [ ] Configure log rotation for Docker containers
- [ ] Set up uptime monitoring (e.g., UptimeRobot, Pingdom)
- [ ] Implement health check endpoints in your load balancer

## Observability

- [ ] Enable OpenTelemetry: `pip install apex-rag[telemetry]`
- [ ] Configure OTLP exporter endpoint
- [ ] Set up query latency alerts (>15s threshold)
- [ ] Track cache hit rate (target: >30%)
- [ ] Monitor error rate (target: <1% of queries)
- [ ] Set `APEX_LOG_LEVEL=WARNING` in production to reduce noise

## Backup & Recovery

- [ ] Schedule regular database backups
- [ ] Test restoration procedure
- [ ] Document the recovery process
- [ ] Keep a copy of the ApexRAG version used (pin in requirements)

## Scaling

| Metric | Threshold | Action |
|--------|-----------|--------|
| Query latency > 10s p99 | 3 consecutive checks | Scale up LLM GPU or reduce concurrency |
| Cache hit rate < 20% | 1 day | Review query patterns; no action needed if queries are unique |
| Error rate > 1% | 5 minutes | Check Ollama / database connectivity |
| Document count > 10,000 | Weekly | Consider sharding by document category |
| Concurrent users > 50 | Growing | Add API server replicas behind load balancer |
