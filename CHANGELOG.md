# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] — 2026-05-26

### Added
- **Structural Retrieval Operating System Redesign**: Complete overhaul of the retrieval engine.
- **Universal Document AST**: Canonical internal representation for PDF, MD, and Code, preserving structural lineage.
- **Multi-Agent Orchestrator**: New reasoning loop featuring `QueryPlannerAgent`, `ASTNavigationAgent`, `EvaluationCriticAgent`, and `EvidenceSynthesizerAgent`.
- **Structural Retrieval Graph (SRG)**: Beyond hierarchies, mapping semantic relationships like `REFERENCES_TABLE` and `DEPENDS_ON`.
- **Evidence Packet System**: Strongly typed provenance packets for zero-hallucination synthesis.
- **Deterministic Candidate Reduction**: Pre-filtering using BM25 and structural heading overlap.
- **Multi-Tenant RBAC**: Data isolation and role-based access control at the database and API levels.
- **Code Intelligence**: Specialized `PythonCodeParser` for structural retrieval over source code repositories.
- **Enterprise Observability**: Native OpenTelemetry integration for distributed tracing of agent reasoning.
- **Retrieval Modes**: Adaptive policies for `LEGAL`, `FINANCIAL`, `CODE`, and `FACTUAL` domains.

### Changed
- Promoted project to version 1.0 stability.
- Migrated `StorageEngine` to support multi-tenant `NodeData` and `SemanticModelData`.
- Enhanced `NavigationResult` to `ASTNavigationResult` with full reasoning traces.

## [0.1.8] — 2026-04-10

### Added
- Production-grade FastAPI integration with visual dashboard
- Global agentic search across all indexed documents
- Streaming SSE endpoints for real-time navigation traces
- AggregatorAgent for multi-document synthesis
- Semantic cache with substring matching for faster repeated queries
- Vision/multimodal support in Summariser for image-rich documents
- CORS middleware for browser-based UIs
- `py.typed` marker for PEP 561 type checker compliance
- `NavigationResult` dataclass with trace, confidence, and verification info
- Hybrid search (FTS5 keyword + LLM agentic)

### Changed
- Restructured `pyproject.toml` dependencies into optional extras (`[web]`, `[postgres]`, `[docling]`, `[migrations]`)
- Cleaned up public API surface — internal classes hidden from `__all__`
- Improved semantic cache with substring fallback matching
- Extracted shared SSE streaming helper to eliminate code duplication
- Increased default `max_concurrent_summaries` to 10 for faster ingestion
- Enhanced CI with matrix testing across Python 3.10–3.12
- Added Ruff linting, formatting checks, and MyPy type checking to CI

### Fixed
- Added missing `import asyncio` in `navigation.py`
- Added `images` parameter to `OpenAIProvider`, `GroqProvider`, `AnthropicProvider` for protocol compliance
- Removed unused `refactor.py` migration script
- Replaced fragile `sys.modules` mocking with proper `patch.dict` in tests
- Fixed `NavigationResult` import in `api.py`

### Removed
- `refactor.py` (stale migration script from `src` → `apex_rag` rename)
- Heavy dependencies (`fastapi`, `uvicorn`, `asyncpg`, `alembic`, `httpx`) from core install — now in optional extras

## [0.1.7] — 2026-03-15

### Added
- First public release of apex-rag
- Core document ingestion pipeline (PDF, DOCX, MD, TXT)
- Markdown → structural tree parser with LTree paths
- LLM-powered Semantic Map summarisation during ingestion
- Recursive Navigation Agent with backtrack-and-verify
- Leaf verification with dedicated verifier model
- SQLAlchemy async storage engine (SQLite + PostgreSQL)
- Book-style page index with alphabetical terms
- Rich HTML dashboard with interactive tree explorer
- `ApexIndex` async context manager API
- `DummyLLM` and mock tree for offline testing
- Full test suite with in-memory SQLite
