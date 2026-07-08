# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] — 2026-07-08

### Changed
- Published updated `README.md` to PyPI — complete API reference, enterprise usage guide, CLI docs, LangChain integration, and environment variable table.

## [1.0.4] — 2026-07-08


### Added
- Published stable release to PyPI as `apex-rag==1.0.4`.
- Full `README.md` rewrite with complete API reference, CLI docs, enterprise usage examples, LangChain integration guide, and environment variable reference.
- `.pypirc` publishing workflow documented in `PUBLISHING.md`.

### Changed
- `README.md` updated throughout to reflect v1.0.4 and the stable `EnterpriseClient` API introduced in v1.0.3.
- Badge links updated to reflect current PyPI version.

## [1.0.3] — 2026-06-28

### Added
- **EnterpriseClient**: New dedicated class for enterprise features (temporal queries, RBAC, version history) accessed via `index.enterprise` property. Keeps `ApexIndex` focused on core ingestion and querying.
- **Deprecation shims**: Removed `__init__.py` symbols raise helpful `ImportError` with the correct import path, easing migration.

### Changed
- **API Stabilization**: Cleaned up the public API surface for a stable 1.x release.
- **Enterprise features extracted**: `temporal_query()`, `get_version_history()`, `get_version_lineage()`, `role_aware_query()` moved from `ApexIndex` to `EnterpriseClient`. `temporal_compare()` kept as deprecated backward-compat wrapper.
- **Dead parameters removed**: `source_date` removed from `ingest_file()` / `ingest()`, `root_node_id` removed from `query()`, `synthesize` removed from `query_global()` — all were accepted but never used.
- **`get_nodes()` removed**: Exact duplicate of `get_tree()`. Use `get_tree()` instead.
- **Hybrid search API unified**: `hybrid=True` parameter replaced by `domain="financial"` which automatically enables hybrid search with domain-tuned freshness decay.
- **Unused imports cleaned**: Removed ~15 dead imports from `client.py`.
- **`__init__.py` exports reduced**: From 21 to 11 symbols. Only user-facing API is now exported (`ApexIndex`, `LLMProvider`, `TenantContext`, `ApexAnswer`, `EvidencePacket`, error classes).
- **EnterpriseClient services lazy-cached**: `VersionResolver`, `TemporalReasoningService`, and `AccessControlAgent` are now lazy singletons on `EnterpriseClient`, avoiding re-creation on each property access.

### Fixed
- **Circular import**: `ApexIndex.enterprise` uses lazy import to avoid circular dependency between `client.py` and `enterprise/client.py`.
- **Test compatibility**: `_patch_storage_create()` in `test_image_ingestion.py` updated to patch `ApexStorage` instead of removed `StorageEngine` import.

## [0.1.8] — 2026-04-10

### Added
- Production-grade FastAPI integration with visual dashboard.
- Global agentic search across all indexed documents.
- Hybrid search (FTS5 keyword + LLM agentic).

### Fixed
- Improved semantic cache with substring fallback matching.
