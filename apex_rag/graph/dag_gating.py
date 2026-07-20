"""
graph/dag_gating.py — Adaptive DAG-Gating Layer.

Controls which Knowledge DAGs are built eagerly (at ingest time),
lazily (on first query that needs them), or deferred (background).

Three modes controlled by ``graph_construction_mode`` in ApexSettings:
    - ``"adaptive"`` — DocumentDAG eager; EntityDAG, CitationDAG,
      PolicyDAG lazy/query-triggered; FactDAG already deferred.
    - ``"eager"`` — All DAGs built synchronously at ingest time (old behavior).
    - ``"minimal"`` — Only DocumentDAG built.

Lazy DAGs are built once and cached: once edges with the relevant
projection tag exist for a document, they are never rebuilt.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from apex_rag.config import settings

if TYPE_CHECKING:
    from apex_rag.ingestion.apex_storage import ApexStorage
    from apex_rag.models.unified_models import ASTNode, KnowledgeEdge

logger = logging.getLogger("apex_rag.graph.dag_gating")


# ── Tier definitions ─────────────────────────────────────────────────────

# DAGs that are always built eagerly in all modes (including minimal).
_EAGER_DAGS: frozenset[str] = frozenset({"document"})

# DAGs built lazily on first query that needs them (adaptive mode only).
_LAZY_DAGS: frozenset[str] = frozenset({"entity", "citation", "policy"})

# DAGs already deferred to background or query-time (never built by gating).
_DEFERRED_DAGS: frozenset[str] = frozenset({"fact", "reasoning", "temporal", "version"})

# All known DAG projection tags.
_ALL_DAGS: frozenset[str] = _EAGER_DAGS | _LAZY_DAGS | _DEFERRED_DAGS


class DAGGatingService:
    """Controls which DAGs are built and when.

    Uses ``ApexStorage`` to check whether edges for a given projection
    already exist for a document (build-once cache semantics).

    Args:
        storage: The :class:`ApexStorage` instance.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    # ── Mode queries ──────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Return the current graph construction mode from settings."""
        return settings.graph_construction_mode

    def should_build_eager(self, projection: str) -> bool:
        """Should a DAG with the given projection be built at ingest time?

        Args:
            projection: DAG projection tag (e.g. ``"entity"``).

        Returns:
            ``True`` if this DAG should be built eagerly.
        """
        if self.mode == "eager":
            return projection in _ALL_DAGS
        if self.mode == "minimal":
            return projection in _EAGER_DAGS
        # adaptive
        return projection in _EAGER_DAGS

    def should_build_lazy(self, projection: str) -> bool:
        """Should a DAG with the given projection be built on demand?

        Args:
            projection: DAG projection tag.

        Returns:
            ``True`` if this DAG can be lazily built at query time.
        """
        if self.mode == "eager":
            return False  # already built at ingest
        if self.mode == "minimal":
            return False  # never build non-essential DAGs
        # adaptive
        return projection in _LAZY_DAGS

    def lazy_dags_for_mode(self) -> frozenset[str]:
        """Return the set of lazy DAG projections for the current mode."""
        if self.mode != "adaptive":
            return frozenset()
        return _LAZY_DAGS

    # ── Build-once cache check ────────────────────────────────────────

    async def is_dag_built(self, projection: str, doc_id: str, *, tenant_id: str = "default") -> bool:
        """Check whether edges with the given projection already exist for
        the document.

        Args:
            projection: DAG projection tag.
            doc_id:     The document ID.
            tenant_id:  Tenant isolation boundary.

        Returns:
            ``True`` if edges exist (DAG was already built for this doc).
        """
        try:
            edges = await self._storage.get_edges_by_projection(
                projection,
                doc_id=doc_id,
                tenant_context=tenant_id,
                limit=1,
            )
            return len(edges) > 0
        except Exception:
            return False

    # ── Lazy DAG building ─────────────────────────────────────────────

    async def ensure_dags(
        self,
        projections: list[str],
        nodes: list[ASTNode],
        *,
        doc_id: str,
        tenant_id: str = "default",
        trigger_reason: str = "query_triggered",
    ) -> list[dict[str, Any]]:
        """Build any requested DAGs that haven't already been built for
        the document.

        Args:
            projections:    List of DAG projection tags to ensure.
            nodes:          AST nodes for the document.
            doc_id:         The document ID.
            tenant_id:      Tenant isolation boundary.
            trigger_reason: ``\"query_triggered\"`` or ``\"background\"``.

        Returns:
            A list of build result dicts with keys: projection, edges_count,
            duration_ms, trigger_reason.
        """
        results: list[dict[str, Any]] = []
        for proj in projections:
            if proj not in _LAZY_DAGS:
                continue
            # Check cache — skip if already built
            already = await self.is_dag_built(proj, doc_id, tenant_id=tenant_id)
            if already:
                logger.debug(
                    "DAG '%s' already built for doc %s — skipping", proj, doc_id[:8]
                )
                continue

            # Build the DAG
            build_result = await self._build_single_dag(
                proj, nodes, doc_id=doc_id, tenant_id=tenant_id,
                trigger_reason=trigger_reason,
            )
            results.append(build_result)

        return results

    async def _build_single_dag(
        self,
        projection: str,
        nodes: list[ASTNode],
        *,
        doc_id: str,
        tenant_id: str,
        trigger_reason: str,
    ) -> dict[str, Any]:
        """Build a single lazy DAG, persist edges, and emit a trace event.

        Args:
            projection:     DAG projection tag.
            nodes:          AST nodes for the document.
            doc_id:         The document ID.
            tenant_id:      Tenant isolation boundary.
            trigger_reason: Why this DAG is being built.

        Returns:
            Dict with build metadata.
        """
        start = time.perf_counter()
        edges: list[KnowledgeEdge] = []

        try:
            if projection == "document":
                from apex_rag.graph.dags.document_dag import DocumentDagBuilder

                builder = DocumentDagBuilder(self._storage)
                edges = await builder.build(
                    nodes, doc_id=doc_id, tenant_id=tenant_id
                )
            elif projection == "entity":
                from apex_rag.graph.dags.entity_dag import EntityDagBuilder

                builder = EntityDagBuilder(self._storage)
                edges = await builder.build(
                    nodes, doc_id=doc_id, tenant_id=tenant_id
                )
            elif projection == "citation":
                from apex_rag.graph.dags.citation_dag import CitationDagBuilder

                builder = CitationDagBuilder(self._storage)
                edges = await builder.build(
                    nodes, doc_id=doc_id, tenant_id=tenant_id
                )
            elif projection == "policy":
                from apex_rag.graph.dags.policy_dag import PolicyDagBuilder

                builder = PolicyDagBuilder(self._storage)
                edges = await builder.build(
                    nodes, doc_id=doc_id, tenant_id=tenant_id
                )
            else:
                logger.warning("Unknown lazy DAG projection: %s", projection)
        except Exception as exc:
            logger.error(
                "Failed to build lazy DAG '%s' for doc %s: %s",
                projection, doc_id[:8], exc,
            )
            duration = (time.perf_counter() - start) * 1000
            return {
                "projection": projection,
                "edges_count": 0,
                "duration_ms": round(duration, 1),
                "trigger_reason": trigger_reason,
                "error": str(exc),
            }

        # Persist edges
        for edge in edges:
            try:
                await self._storage.save_knowledge_edge(edge)
            except Exception as exc:
                logger.warning(
                    "Failed to save edge %s for DAG '%s': %s",
                    edge.edge_id[:8], projection, exc,
                )

        duration = (time.perf_counter() - start) * 1000

        logger.info(
            "DAG '%s' built for doc %s: %d edges in %.1fms (trigger: %s)",
            projection, doc_id[:8], len(edges), duration, trigger_reason,
        )

        # Emit trace event via trace_manager
        try:
            from apex_rag.observability.trace_manager import trace_manager

            trace_manager.publish(
                trace_id=f"dag-{doc_id[:8]}-{projection}",
                trace_type="graph",
                event_name="DAG_BUILD",
                data={
                    "projection": projection,
                    "doc_id": doc_id,
                    "edges_count": len(edges),
                    "duration_ms": round(duration, 1),
                    "trigger_reason": trigger_reason,
                },
            )
        except Exception:
            pass

        return {
            "projection": projection,
            "edges_count": len(edges),
            "duration_ms": round(duration, 1),
            "trigger_reason": trigger_reason,
        }

    # ── Background DAG builder ────────────────────────────────────────

    async def build_background_dags(
        self,
        nodes: list[ASTNode],
        *,
        doc_id: str,
        tenant_id: str = "default",
    ) -> None:
        """Build deferred/background DAGs asynchronously after ingest.

        Currently handles TemporalDAG and VersionDAG which are
        lightweight enough to run eagerly but not critical for
        first-query latency.  Runs in a fire-and-forget async task.

        Args:
            nodes:    AST nodes for the document.
            doc_id:   The document ID.
            tenant_id: Tenant isolation boundary.
        """
        # Build TemporalDAG if not already built
        try:
            already = await self.is_dag_built("temporal", doc_id, tenant_id=tenant_id)
            if not already:
                start = time.perf_counter()
                from apex_rag.graph.dags.temporal_dag import TemporalDagBuilder

                builder = TemporalDagBuilder(self._storage)
                edges = await builder.build(
                    nodes, doc_id=doc_id, tenant_id=tenant_id
                )
                for edge in edges:
                    try:
                        await self._storage.save_knowledge_edge(edge)
                    except Exception:
                        pass
                duration = (time.perf_counter() - start) * 1000

                if edges:
                    logger.info(
                        "TemporalDAG (background): %d edges in %.1fms",
                        len(edges), duration,
                    )
        except Exception as exc:
            logger.warning("Background TemporalDAG build failed: %s", exc)

        # VersionDAG is not built during ingest (requires version_rows).
        # It is built separately via VersionDagBuilder.build_from_versions().

    # ── Synchronous DAG builder for ingest ───────────────────────────

    async def build_eager_dags(
        self,
        nodes: list[ASTNode],
        *,
        doc_id: str,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Build all DAGs that are marked eager for the current mode.

        Args:
            nodes:    AST nodes for the document.
            doc_id:   The document ID.
            tenant_id: Tenant isolation boundary.

        Returns:
            Build results for each eager DAG.
        """
        results: list[dict[str, Any]] = []

        # DocumentDAG — always built in all modes
        doc_result = await self._build_single_dag(
            "document", nodes, doc_id=doc_id, tenant_id=tenant_id,
            trigger_reason="eager",
        )
        results.append(doc_result)

        # In eager mode, also build lazy DAGs + TemporalDAG synchronously
        if self.mode == "eager":
            for proj in sorted(_LAZY_DAGS):
                result = await self._build_single_dag(
                    proj, nodes, doc_id=doc_id, tenant_id=tenant_id,
                    trigger_reason="eager",
                )
                results.append(result)
            # TemporalDAG was built synchronously in old code — preserve that.
            # Use build_background_dags which has the TemporalDagBuilder logic.
            # It will skip if already built.
            await self.build_background_dags(
                nodes, doc_id=doc_id, tenant_id=tenant_id,
            )

        # In minimal mode, build nothing else
        # In adaptive mode, lazy DAGs are NOT built here

        return results
