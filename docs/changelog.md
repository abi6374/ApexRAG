# Changelog

See the full [`CHANGELOG.md`](https://github.com/abi6374/apexrag/blob/main/CHANGELOG.md)
on GitHub for the complete changelog.

## Latest Release — [1.0.5] — 2026-07

### Added
- **8 Knowledge DAG Projections** — Unified `KnowledgeEdge` model supporting Document, Entity, Citation, Temporal, Version, Policy, Fact, and Reasoning DAGs. Each edge carries `projections` tags for multi-DAG membership.
- **ReasoningDAG** — Orchestrator trace events captured and persisted as typed reasoning edges (`REASONING_CHAIN`, `DERIVES_FROM`, `INFERS`, `USES`). Built by `ReasoningDagBuilder` after each query execution.
- **SSE ReasoningDAG Streaming** — `POST /query/stream/reasoning-graph` endpoint delivers real-time agent traces via SSE followed by the full ReasoningDAG as `{nodes, edges}` JSON.
- **Global Graph API** — `GET /graph` and `GET /graph/{projection}` endpoints expose the knowledge graph across **all** indexed documents.
- **Enriched Graph Nodes** — Node responses now include `node_type`, `page_number`, and human-readable content labels resolved from the AST nodes table (instead of truncated UUIDs).
- **Batch Node Lookup** — `ApexStorage.get_nodes_batch()` for efficient single-query multi-node retrieval.
- **DAG Visualization** — Interactive vis-network graph tab added to both the dashboard (`/`) and document view (`/documents/{doc_id}/index/page`).
- **Comprehensive REST API docs** — Full `docs/rest-api.md` covering all 29 endpoints.
- **External Trace ID** — `ApexOrchestrator.run()` accepts `external_trace_id` parameter enabling external SSE listeners to sync with orchestrator traces.

### Changed
- `_build_graph_response()` is now async with node metadata resolution.
- Updated module docstrings for all new endpoints.
- Bumped version to `v1.0.5`.

[Full changelog on GitHub](https://github.com/abi6374/apexrag/blob/main/CHANGELOG.md)
