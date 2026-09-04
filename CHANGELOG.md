# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.7] — 2026-09-04

### Fixed

- **Document version-history crash (critical):** `version_dag.py` referenced
  `RelationType.REPLACED_BY`, which existed in `models/unified_models.py`'s
  `EdgeType` enum but not in `graph/edges/models.py`'s separate
  `RelationType` enum (the one actually imported by `version_dag.py`),
  crashing all document version-history operations
  (`AttributeError: type object 'RelationType' has no attribute
  'REPLACED_BY'`). Also affected: `VALID_DURING` and `SNAPSHOT_OF`, the same
  class of gap in the same enum, used by `temporal_dag.py`/`version_dag.py`
  but never previously hit by any test.
  - Added the three missing members to `graph/edges/models.py`'s
    `RelationType` (kept as the canonical enum for all DAG builders --
    every other builder already imports `RelationType` from this module,
    confirmed by checking all 8 sibling builders; `unified_models.EdgeType`
    remains a separate, narrower enum used for other purposes).
  - Fixed a second, previously-masked bug in the same code path: both
    `version_dag.py` (SUPERSEDES/REPLACED_BY) and `temporal_dag.py`
    (SUCCESSOR/PREDECESSOR) stored the same relationship as two edges in
    opposite directions between the same node pair, which is a 2-cycle --
    correctly rejected at write time by `ApexStorage.save_knowledge_edge()`
    (Principle 11 — DAG Acyclicity). Only the forward edge (SUPERSEDES /
    SUCCESSOR) is now stored; the inverse relationship is available by
    reverse-traversing it rather than duplicating it as a contradictory
    stored edge.

## [1.0.6] — 2026-09-04

### Added

- **`EnterpriseClient.calibrate_conformal()`**: Real split-conformal
  calibration from a held-out labeled `(question, doc_id, gold_answer)`
  set. Previously `ApexIndex.query()` never had a way to calibrate the
  conformal predictor, so `answer.coverage_guarantee` always read `0.0`
  and every retrieved packet passed through unfiltered — a decorative
  guarantee. Calling `index.enterprise.calibrate_conformal(...)` once
  persists a real threshold on the orchestrator, so every subsequent
  `index.query()` call reflects it automatically. See the README's
  "Conformal Calibration" section.

### Fixed

- **Packaging crisis (critical):** 1.0.5 shipped without `apex_rag/models/`
  (specifically `unified_models.py`), making the entire package unimportable.
  The root cause was a `.gitignore` pattern `models/` that recursively excluded
  `apex_rag/models/` from the built sdist and wheel because hatchling respects
  `.gitignore` patterns during builds. Fixed by:
  - Changing `.gitignore` from `models/` to `/models/` (root-level only)
  - Adding `"apex_rag.models"` to the explicit `packages` list in `[tool.hatch.build]`
  - Strengthened CI build verification to check `from apex_rag import ApexIndex`
    from both wheel and sdist artifacts

## [1.0.5] — 2026-07-17

### Fixed
- **Critical regression**: `ingest()` raised `MissingTenantContextError` for all documents due to missing `tenant_context` argument in `save_page_index_entries()` call site (regression introduced in 1.0.3).
- **Additional missing `tenant_context`**: `get_page_index_entries()` call in `ApexIndex.get_page_index()` was also missing `tenant_context` argument.
- **Duplicate index warning**: Removed duplicate index declarations in `NodeVersionRow` (`node_id`, `doc_id`, `tenant_id`) and `RoleProfileRow` (`name`) where column-level `index=True` collided with explicit `Index()` in `__table_args__`, causing spurious warnings on fresh database creation.
- **`CausalRetriever` tenant isolation**: Fixed default `tenant_context` from `None` to `"default"` so graph traversal operations don't raise `MissingTenantContextError`.

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
