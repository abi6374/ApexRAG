# Changelog

See the full [`CHANGELOG.md`](https://github.com/abi6374/apexrag/blob/main/CHANGELOG.md)
on GitHub for the complete changelog.

## Latest Release — [1.0.2] — 2026-05-26

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

[Full changelog on GitHub](https://github.com/abi6374/apexrag/blob/main/CHANGELOG.md)
