# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] — 2026-05-26

### Added
- **Structural Retrieval Operating System Redesign**: Complete overhaul of the retrieval engine to favor structure over proximity.
- **Universal Document AST**: Canonical internal representation for PDF, MD, and Code, preserving structural lineage.
- **Multi-Agent Orchestrator**: New reasoning loop featuring `QueryPlannerAgent`, `ASTNavigationAgent`, `EvaluationCriticAgent`, and `EvidenceSynthesizerAgent`.
- **LLM Provider Adapters**: Clean abstraction for OpenAI, Anthropic, Groq, and Ollama with async streaming support.
- **LangChain Integration**: `ApexRAGRetriever` for drop-in usage in existing LangChain ecosystems.
- **Gradio Demo Application**: Out-of-the-box `app.py` for visual document navigation.
- **RAGAS Benchmark Harness**: Specialized evaluation scripts for HotpotQA.
- **Multi-Tenant RBAC**: Data isolation and role-based access control at the database level.
- **Enterprise Observability**: Native OpenTelemetry integration for distributed reasoning traces.

### Changed
- Promoted project to version 1.0 (Production Stable).
- Upgraded `pyproject.toml` with modular optional dependencies.
- Migrated `StorageEngine` to support multi-tenant `NodeData`.

## [0.1.8] — 2026-04-10

### Added
- Production-grade FastAPI integration with visual dashboard.
- Global agentic search across all indexed documents.
- Hybrid search (FTS5 keyword + LLM agentic).

### Fixed
- Improved semantic cache with substring fallback matching.
