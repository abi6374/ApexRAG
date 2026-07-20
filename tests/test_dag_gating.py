"""
Tests for the adaptive/lazy/eager Knowledge DAG gating layer.

Covers:
    - Adaptive mode: DocumentDAG built eagerly, lazy DAGs deferred
    - Eager mode: all DAGs built synchronously (regression safety)
    - Minimal mode: only DocumentDAG built
    - Build-once cache semantics (second query doesn't rebuild)
    - Query-need classifier (DAGRouter)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from apex_rag.agents.planner.dag_router import DAGRouter
from apex_rag.graph.dag_gating import DAGGatingService, _EAGER_DAGS, _LAZY_DAGS


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock ApexStorage."""
    storage = MagicMock()
    storage.get_edges_by_projection = AsyncMock(return_value=[])
    storage.save_knowledge_edge = AsyncMock()
    storage.get_nodes_by_doc = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_nodes() -> list[MagicMock]:
    """Create mock AST nodes."""
    nodes = []
    for i in range(3):
        node = MagicMock()
        node.node_id = f"node-{i:03d}"
        node.content = f"Test content {i}"
        node.doc_id = "test-doc-001"
        node.parent_id = "node-000" if i > 0 else None
        node.depth = i
        node.children = ["node-001"] if i == 0 else []
        node.source_date = None
        node.embedding = [0.1, 0.2, 0.3]
        node.node_type = "PARAGRAPH"
        nodes.append(node)
    return nodes


@pytest.fixture
def dag_gating(mock_storage: MagicMock) -> DAGGatingService:
    """Create DAGGatingService with mocked storage."""
    return DAGGatingService(mock_storage)


# ═══════════════════════════════════════════════════════════════════════
# DAGGatingService — Mode Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDAGGatingMode:
    """Test that mode-based gating works correctly."""

    @pytest.mark.parametrize("mode,projection,expected", [
        ("adaptive", "document", True),
        ("adaptive", "entity", False),
        ("adaptive", "citation", False),
        ("adaptive", "policy", False),
        ("eager", "document", True),
        ("eager", "entity", True),
        ("eager", "citation", True),
        ("eager", "policy", True),
        ("minimal", "document", True),
        ("minimal", "entity", False),
        ("minimal", "citation", False),
        ("minimal", "policy", False),
    ])
    def test_should_build_eager(self, mode: str, projection: str, expected: bool) -> None:
        """Verify should_build_eager returns correct values per mode."""
        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = mode
            storage = MagicMock()
            gating = DAGGatingService(storage)
            assert gating.should_build_eager(projection) == expected

    def test_lazy_dags_for_mode_adaptive(self) -> None:
        """Adaptive mode should return all lazy DAGs."""
        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = "adaptive"
            gating = DAGGatingService(MagicMock())
            assert gating.lazy_dags_for_mode() == _LAZY_DAGS

    def test_lazy_dags_for_mode_eager(self) -> None:
        """Eager mode should return no lazy DAGs."""
        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = "eager"
            gating = DAGGatingService(MagicMock())
            assert gating.lazy_dags_for_mode() == frozenset()

    def test_lazy_dags_for_mode_minimal(self) -> None:
        """Minimal mode should return no lazy DAGs."""
        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = "minimal"
            gating = DAGGatingService(MagicMock())
            assert gating.lazy_dags_for_mode() == frozenset()


# ═══════════════════════════════════════════════════════════════════════
# DAGGatingService — Build Eager Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDAGGatingBuildEager:
    """Test that build_eager_dags builds the correct DAGs per mode."""

    @pytest.mark.asyncio
    async def test_adaptive_mode_builds_only_document_dag(
        self, dag_gating: DAGGatingService, mock_nodes: list[MagicMock]
    ) -> None:
        """Adaptive mode: only DocumentDAG should be built eagerly."""
        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = "adaptive"
            results = await dag_gating.build_eager_dags(
                mock_nodes, doc_id="test-doc-001", tenant_id="default"
            )

            # Only 1 result (DocumentDAG)
            assert len(results) == 1
            assert results[0]["projection"] == "document"
            assert results[0]["trigger_reason"] == "eager"

    @pytest.mark.asyncio
    async def test_eager_mode_builds_all_dags(
        self, dag_gating: DAGGatingService, mock_nodes: list[MagicMock]
    ) -> None:
        """Eager mode: all DAGs should be built."""
        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = "eager"
            results = await dag_gating.build_eager_dags(
                mock_nodes, doc_id="test-doc-001", tenant_id="default"
            )

            # Should build document + entity + citation + policy = 4
            # (TemporalDAG is also built in eager mode but not counted in results
            #  since build_background_dags doesn't return build results)
            assert len(results) == 1 + len(_LAZY_DAGS)
            projections = {r["projection"] for r in results}
            assert "document" in projections
            assert all(p in projections for p in ["entity", "citation", "policy"])

    @pytest.mark.asyncio
    async def test_minimal_mode_builds_only_document_dag(
        self, dag_gating: DAGGatingService, mock_nodes: list[MagicMock]
    ) -> None:
        """Minimal mode: only DocumentDAG."""
        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = "minimal"
            results = await dag_gating.build_eager_dags(
                mock_nodes, doc_id="test-doc-001", tenant_id="default"
            )

            assert len(results) == 1
            assert results[0]["projection"] == "document"


# ═══════════════════════════════════════════════════════════════════════
# DAGGatingService — Build-once Cache Semantics
# ═══════════════════════════════════════════════════════════════════════


class TestDAGGatingCache:
    """Test build-once cache semantics for lazy DAGs."""

    @pytest.mark.asyncio
    async def test_lazy_dag_built_once_and_cached(
        self, dag_gating: DAGGatingService, mock_nodes: list[MagicMock]
    ) -> None:
        """A lazy DAG built on first query should not be rebuilt on second."""
        # First call: DAG not built yet (simulate by returning no edges)
        dag_gating._storage.get_edges_by_projection = AsyncMock(return_value=[])

        with patch("apex_rag.graph.dag_gating.settings") as mock_settings:
            mock_settings.graph_construction_mode = "adaptive"

            results = await dag_gating.ensure_dags(
                ["entity"], mock_nodes,
                doc_id="test-doc-001", tenant_id="default",
            )
            # Should have built entity DAG
            entity_results = [r for r in results if r["projection"] == "entity"]
            assert len(entity_results) == 1
            assert entity_results[0]["trigger_reason"] == "query_triggered"

            # Now simulate DAG is built — next call should skip
            dag_gating._storage.get_edges_by_projection = AsyncMock(
                return_value=[{"edge_id": "mock-edge"}]
            )

            results2 = await dag_gating.ensure_dags(
                ["entity"], mock_nodes,
                doc_id="test-doc-001", tenant_id="default",
            )
            # Should have zero new builds (already cached)
            entity_results2 = [r for r in results2 if r["projection"] == "entity"]
            assert len(entity_results2) == 0

    @pytest.mark.asyncio
    async def test_second_identical_query_does_not_rebuild(
        self, dag_gating: DAGGatingService, mock_nodes: list[MagicMock]
    ) -> None:
        """Verify that a second query with the same need doesn't rebuild."""
        dag_gating._storage.get_edges_by_projection = AsyncMock(return_value=[])
        build_spy = AsyncMock(wraps=dag_gating._build_single_dag)

        with patch.object(dag_gating, "_build_single_dag", build_spy):
            results = await dag_gating.ensure_dags(
                ["entity", "citation"], mock_nodes,
                doc_id="test-doc-001",
            )
            # Should have triggered builds
            assert len(results) == 2

            # Second call — now return edges to indicate already built
            dag_gating._storage.get_edges_by_projection = AsyncMock(
                return_value=[{"edge_id": "mock-edge"}]
            )

            results2 = await dag_gating.ensure_dags(
                ["entity", "citation"], mock_nodes,
                doc_id="test-doc-001",
            )
            assert len(results2) == 0  # Nothing rebuilt

            # _build_single_dag should have been called exactly 2 times (first call only)
            assert build_spy.call_count == 2


# ═══════════════════════════════════════════════════════════════════════
# DAGRouter — Query Classification Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDAGRouter:
    """Test the query-need classifier."""

    def setup_method(self) -> None:
        self.router = DAGRouter()

    @pytest.mark.parametrize("query,expected", [
        # Entity queries
        ("Who is the CEO?", {"entity"}),
        ("What entities are mentioned?", {"entity"}),
        ("Which company employs John?", {"entity"}),
        ("List all organizations", {"entity"}),
        # Citation queries
        ("What does Section 3.2 cite?", {"citation"}),
        ("Find references in the bibliography", {"citation"}),
        ("According to Smith et al., what is...", {"citation"}),
        # "See page" doesn't match the citation regex (needs §, section, chapter)
        ("See page regression for details", frozenset()),
        # Policy queries
        ("What are the compliance requirements?", {"policy"}),
        ("Which GDPR regulations apply?", {"policy"}),
        ("All employees shall comply with...", {"entity", "policy"}),
        ("Legal obligations under HIPAA", {"policy"}),
        # Mixed queries
        ("Who must comply with ISO standards?", {"entity", "policy"}),
        ("Cite the company policy on GDPR", {"citation", "entity", "policy"}),
        # Non-triggering queries
        ("What is the revenue for Q3?", frozenset()),
        ("Summarize the document", frozenset()),
        ("Explain the main topics", frozenset()),
    ])
    def test_classify(self, query: str, expected: set[str]) -> None:
        """Verify that DAGRouter correctly classifies queries."""
        result = self.router.classify(query)
        assert result == expected, f"Query '{query}' expected {expected} but got {result}"

    def test_classify_with_planner_data(self) -> None:
        """Test that planner data enriches classification."""
        # LEGAL query type should trigger policy even without keywords
        result = self.router.classify(
            "What are the requirements?",
            planner_data={"query_type": "LEGAL", "entity_hints": {}},
        )
        assert "policy" in result

    def test_classify_from_plan(self) -> None:
        """Test classify_from_plan with a mock plan object."""
        plan = MagicMock()
        plan.query_type = "FINANCIAL"
        plan.entity_hints = {"hint1": ["entity1"]}
        plan.to_dict = MagicMock(return_value={"query_type": "FINANCIAL", "entity_hints": {}})

        result = self.router.classify_from_plan(plan, "What is the revenue?")
        assert "policy" in result  # FINANCIAL query type
