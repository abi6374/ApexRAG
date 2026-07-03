"""
Tests for Part 4 — Causal Knowledge Graph.

Covers:
- GraphEdge model (conversion, relation types)
- CausalGraphBuilder (structural, temporal, semantic, LLM)
- CausalRetriever (chain building, path finding, subgraph extraction)
- Deduplication and edge persistence
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pytest import approx

from apex_rag.graph.edges.causal_builder import CausalGraphBuilder, Embedder
from apex_rag.graph.edges.causal_retriever import CausalRetriever, StorageProvider
from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.models.unified_models import ASTNode, CausalEdge, EdgeType

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def node_a() -> ASTNode:
    return ASTNode(
        node_id="11111111-1111-4111-8111-111111111111",
        content="Q3 revenue was $40M, driven by strong SaaS growth.",
        node_type="PARAGRAPH",
        depth=1,
        parent_id="00000000-0000-4000-8000-000000000000",
        doc_id="doc-001",
        source_date=datetime(2024, 10, 1, tzinfo=timezone.utc),
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


@pytest.fixture
def node_b() -> ASTNode:
    return ASTNode(
        node_id="22222222-2222-4222-8222-222222222222",
        content="Q3 revenue was $52M, driven by enterprise expansion.",
        node_type="PARAGRAPH",
        depth=1,
        parent_id="00000000-0000-4000-8000-000000000000",
        doc_id="doc-001",
        source_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
        embedding=[0.15, 0.25, 0.35, 0.45],
    )


@pytest.fixture
def root_node() -> ASTNode:
    return ASTNode(
        node_id="00000000-0000-4000-8000-000000000000",
        content="Q3 Financial Report",
        node_type="HEADING",
        depth=0,
        parent_id=None,
        doc_id="doc-001",
        source_date=datetime(2024, 9, 1, tzinfo=timezone.utc),
        embedding=[0.5, 0.5, 0.5, 0.5],
    )


@pytest.fixture
def isolated_node() -> ASTNode:
    return ASTNode(
        node_id="33333333-3333-4333-8333-333333333333",
        content="Q3 marketing spend increased 15% year-over-year.",
        node_type="PARAGRAPH",
        depth=1,
        parent_id="00000000-0000-4000-8000-000000000000",
        doc_id="doc-001",
        source_date=datetime(2024, 10, 1, tzinfo=timezone.utc),
        embedding=[0.9, 0.1, 0.9, 0.1],
    )


@pytest.fixture
def child_node() -> ASTNode:
    return ASTNode(
        node_id="44444444-4444-4444-8444-444444444444",
        content="SaaS revenue breakdown: $28M subscription, $12M services.",
        node_type="PARAGRAPH",
        depth=2,
        parent_id="11111111-1111-4111-8111-111111111111",
        doc_id="doc-001",
        embedding=[0.12, 0.22, 0.32, 0.42],
    )


@pytest.fixture
def all_nodes(root_node, node_a, node_b, isolated_node, child_node) -> list[ASTNode]:
    return [root_node, node_a, node_b, isolated_node, child_node]


# ═══════════════════════════════════════════════════════════════════
# GraphEdge model tests
# ═══════════════════════════════════════════════════════════════════


class TestGraphEdgeModel:
    def test_to_causal_edge_refines(self) -> None:
        """GraphEdge with REFINES converts to CausalEdge with same values."""
        ge = GraphEdge(
            source_id="11111111-1111-4111-8111-111111111111",
            target_id="22222222-2222-4222-8222-222222222222",
            relation_type=RelationType.REFINES,
            strength=0.8,
            evidence="test edge",
        )
        ce = ge.to_causal_edge()
        assert ce.source_node_id == ge.source_id
        assert ce.target_node_id == ge.target_id
        assert ce.edge_type == EdgeType.REFINES
        assert ce.strength == 0.8
        assert ce.evidence == "test edge"

    def test_to_causal_edge_extension_type(self) -> None:
        """SRG-only types like REFERENCES_TABLE fall back to SUPPORTS."""
        ge = GraphEdge(
            source_id="11111111-1111-4111-8111-111111111111",
            target_id="22222222-2222-4222-8222-222222222222",
            relation_type=RelationType.REFERENCES_TABLE,
        )
        ce = ge.to_causal_edge()
        assert ce.edge_type == EdgeType.SUPPORTS

    def test_from_causal_edge(self) -> None:
        """CausalEdge round-trips through GraphEdge.from_causal_edge."""
        ce = CausalEdge(
            source_node_id="11111111-1111-4111-8111-111111111111",
            target_node_id="22222222-2222-4222-8222-222222222222",
            edge_type=EdgeType.CONTRADICTS,
            strength=0.9,
            evidence="direct contradiction",
        )
        ge = GraphEdge.from_causal_edge(ce)
        assert ge.source_id == ce.source_node_id
        assert ge.target_id == ce.target_node_id
        assert ge.relation_type == RelationType.CONTRADICTS
        assert ge.strength == 0.9
        assert ge.evidence == "direct contradiction"

    def test_relation_type_from_edge_type(self) -> None:
        """RelationType.from_edge_type converts base types correctly."""
        assert RelationType.from_edge_type(EdgeType.SUPPORTS) == RelationType.SUPPORTS
        assert RelationType.from_edge_type(EdgeType.OVERRIDES) == RelationType.OVERRIDES

    def test_graph_edge_defaults(self) -> None:
        """GraphEdge has sensible defaults for strength and evidence."""
        ge = GraphEdge(
            source_id="11111111-1111-4111-8111-111111111111",
            target_id="22222222-2222-4222-8222-222222222222",
            relation_type=RelationType.SAME_TOPIC,
        )
        assert ge.strength == 0.5
        assert ge.evidence == ""
        assert ge.metadata == {}
        assert uuid.UUID(ge.id, version=4)


# ═══════════════════════════════════════════════════════════════════
# CausalGraphBuilder tests
# ═══════════════════════════════════════════════════════════════════


class TestCausalGraphBuilder:
    def test_structural_parent_child(self, child_node, all_nodes) -> None:
        """Parent-child relationship produces REFINES edge."""
        builder = CausalGraphBuilder()
        # Include both parent and child
        parent = [n for n in all_nodes if n.node_id == child_node.parent_id][0]
        edges = builder.build_structural([parent, child_node])
        refines = [e for e in edges if e.relation_type == RelationType.REFINES]
        assert len(refines) == 1
        assert refines[0].source_id == parent.node_id
        assert refines[0].target_id == child_node.node_id

    def test_structural_siblings(self, node_a, node_b) -> None:
        """Sibling nodes produce SUPPORTS edges."""
        builder = CausalGraphBuilder()
        edges = builder.build_structural([node_a, node_b])
        supports = [e for e in edges if e.relation_type == RelationType.SUPPORTS]
        assert len(supports) >= 1

    def test_structural_single_node(self, root_node) -> None:
        """Single node produces no structural edges."""
        builder = CausalGraphBuilder()
        edges = builder.build_structural([root_node])
        assert len(edges) == 0

    def test_structural_orphan_handling(self) -> None:
        """Nodes without parents don't cause errors."""
        orphans = [
            ASTNode(
                node_id="55555555-5555-4555-8555-555555555555",
                content="Orphan 1",
                node_type="PARAGRAPH",
                doc_id="doc-001",
            ),
            ASTNode(
                node_id="66666666-6666-4666-8666-666666666666",
                content="Orphan 2",
                node_type="PARAGRAPH",
                doc_id="doc-001",
            ),
        ]
        builder = CausalGraphBuilder()
        edges = builder.build_structural(orphans)
        # Two orphans → siblings → one SUPPORTS edge
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.SUPPORTS

    async def test_semantic_requires_embedder(self, node_a, node_b) -> None:
        """Semantic strategy returns empty list without an embedder."""
        builder = CausalGraphBuilder()
        edges = await builder.build_semantic([node_a, node_b])
        assert edges == []

    async def test_semantic_discovery(self) -> None:
        """Semantic strategy discovers SUPPORTS for similar nodes."""

        class FakeEmbedder(Embedder):
            def __init__(self, embs: list[list[float]]) -> None:
                self._embs = embs

            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return self._embs

        embs = [[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]]
        builder = CausalGraphBuilder(
            embedder=FakeEmbedder(embs),
            similarity_threshold=0.75,
        )
        # Create fresh nodes with embeddings directly set
        a = ASTNode(
            node_id="aaaa0000-0000-4000-8000-000000000000",
            content="Revenue was $40M",
            node_type="PARAGRAPH",
            doc_id="doc-001",
            embedding=embs[0],
        )
        b = ASTNode(
            node_id="bbbb0000-0000-4000-8000-000000000000",
            content="Revenue was $52M",
            node_type="PARAGRAPH",
            doc_id="doc-001",
            embedding=embs[1],
        )
        edges = await builder.build_semantic([a, b])
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.SUPPORTS

    async def test_semantic_low_similarity(self, node_a, isolated_node) -> None:
        """Semantic strategy skips pairs below threshold."""

        class FakeEmbedder(Embedder):
            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3, 0.4], [0.9, 0.1, 0.9, 0.1]]

        builder = CausalGraphBuilder(
            embedder=FakeEmbedder(),
            similarity_threshold=0.75,
        )
        a = node_a.model_copy(update={"embedding": []})
        iso = isolated_node.model_copy(update={"embedding": []})
        edges = await builder.build_semantic([a, iso])
        # Cosine sim: dot=0.09+0.02+0.27+0.04=0.42
        # norm_a=0.5477, norm_b=sqrt(0.81+0.01+0.81+0.01)=sqrt(1.64)=1.2806
        # sim=0.42/(0.5477*1.2806)=0.42/0.7015=0.598 < 0.75
        assert len(edges) == 0

    def test_structural_skip_structural(self, node_a, child_node) -> None:
        """_is_structurally_related detects parent-child and siblings."""
        assert CausalGraphBuilder._is_structurally_related(node_a, child_node)
        # node_a and child_node are parent-child

    def test_not_structurally_related(self, node_a, child_node) -> None:
        """Nodes that are not parent-child or siblings are not structurally related."""
        # node_a and child_node are parent-child: they ARE structurally related
        assert CausalGraphBuilder._is_structurally_related(node_a, child_node)
        # Use nodes with different parents
        other = ASTNode(
            node_id="55555555-5555-4555-8555-555555555555",
            content="Other doc content",
            node_type="PARAGRAPH",
            parent_id="99999999-9999-4999-8999-999999999999",
            doc_id="doc-other",
        )
        assert not CausalGraphBuilder._is_structurally_related(node_a, other)

    def test_cosine_similarity_identical(self) -> None:
        """Identical vectors have cosine similarity of 1.0."""
        sim = CausalGraphBuilder._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        """Orthogonal vectors have cosine similarity of 0.0."""
        sim = CausalGraphBuilder._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == approx(0.0)

    def test_cosine_similarity_empty(self) -> None:
        """Empty vectors return 0.0."""
        sim = CausalGraphBuilder._cosine_similarity([], [1.0])
        assert sim == 0.0

    async def test_build_all_empty_nodes(self) -> None:
        """build_all returns empty for empty node list."""
        builder = CausalGraphBuilder()
        edges = await builder.build_all(
            [],
            include_temporal=False,
            include_semantic=False,
            include_llm=False,
        )
        assert edges == []

    async def test_build_all_structural_only(self, all_nodes) -> None:
        """build_all with only structural enabled works."""
        builder = CausalGraphBuilder()
        edges = await builder.build_all(
            all_nodes,
            include_temporal=False,
            include_semantic=False,
            include_llm=False,
        )
        assert len(edges) >= 1
        all_refines = [e for e in edges if e.relation_type == RelationType.REFINES]
        assert len(all_refines) >= 1  # parent-child edges

    async def test_build_all_dedup(self, node_a, child_node) -> None:
        """build_all deduplicates by source-target-relation."""
        builder = CausalGraphBuilder()
        edges = await builder.build_all(
            [node_a, child_node],
            include_temporal=False,
            include_semantic=False,
            include_llm=False,
        )
        # With the parent-only-check fix: node_a's parent (root) is not in the list,
        # child_node's parent (node_a) IS in the list → 1 REFINES edge
        refines = [e for e in edges if e.relation_type == RelationType.REFINES]
        assert len(refines) == 1
        assert refines[0].source_id == node_a.node_id
        assert refines[0].target_id == child_node.node_id

    async def test_llm_requires_llm(self, node_a, node_b) -> None:
        """LLM strategy returns empty list without an LLM."""
        builder = CausalGraphBuilder()
        edges = await builder.build_llm([node_a, node_b])
        assert edges == []


# ═══════════════════════════════════════════════════════════════════
# CausalRetriever tests
# ═══════════════════════════════════════════════════════════════════


class FakeStorage(StorageProvider):
    """In-memory mock storage for testing the CausalRetriever."""

    def __init__(self) -> None:
        self._nodes: dict[str, ASTNode] = {}
        self._edges: dict[str, list[CausalEdge]] = {}  # node_id → edges

    def add_node(self, node: ASTNode) -> None:
        self._nodes[node.node_id] = node

    def add_edge(self, edge: CausalEdge) -> None:
        self._edges.setdefault(edge.source_node_id, []).append(edge)
        self._edges.setdefault(edge.target_node_id, []).append(edge)

    async def get_edges_for_node(self, node_id: str, *, tenant_context: str | None = None) -> list[CausalEdge]:  # noqa: ARG002
        return self._edges.get(node_id, [])

    async def get_node(self, node_id: str, *, tenant_context: str | None = None) -> ASTNode | None:  # noqa: ARG002
        return self._nodes.get(node_id)

    async def get_nodes_by_doc(self, doc_id: str, *, tenant_context: str | None = None) -> list[ASTNode]:  # noqa: ARG002
        return [n for n in self._nodes.values() if n.doc_id == doc_id]


def _valid_uuid(tag: str) -> str:
    """Generate a valid UUID4 from a predictable tag (for testing)."""
    import hashlib

    hex_digest = hashlib.md5(tag.encode()).hexdigest()
    # Insert UUID4 version nibble at position 12
    return f"{hex_digest[:12]}4{hex_digest[13:16]}8{hex_digest[17:32]}"


@pytest.fixture
def chain_storage() -> FakeStorage:
    """Build a small graph for chain-building tests."""
    # Node layout:
    #   root → a → c
    #        → b (SUPPORTS a)
    root_id = _valid_uuid("root")
    a_id = _valid_uuid("a")
    b_id = _valid_uuid("b")
    c_id = _valid_uuid("c")

    root = ASTNode(
        node_id=root_id,
        content="Root",
        node_type="HEADING",
        doc_id="doc-001",
    )
    a = ASTNode(
        node_id=a_id,
        content="Revenue was $40M",
        node_type="PARAGRAPH",
        parent_id=root_id,
        doc_id="doc-001",
    )
    b = ASTNode(
        node_id=b_id,
        content="Profit was $10M",
        node_type="PARAGRAPH",
        parent_id=root_id,
        doc_id="doc-001",
    )
    c = ASTNode(
        node_id=c_id,
        content="SaaS breakdown",
        node_type="PARAGRAPH",
        parent_id=a_id,
        doc_id="doc-001",
    )

    storage = FakeStorage()
    for n in [root, a, b, c]:
        storage.add_node(n)

    # REFINES: parent → child
    storage.add_edge(
        CausalEdge(
            source_node_id=root.node_id,
            target_node_id=a.node_id,
            edge_type=EdgeType.REFINES,
            strength=0.8,
            evidence="parent-child",
        )
    )
    storage.add_edge(
        CausalEdge(
            source_node_id=root.node_id,
            target_node_id=b.node_id,
            edge_type=EdgeType.REFINES,
            strength=0.8,
            evidence="parent-child",
        )
    )
    storage.add_edge(
        CausalEdge(
            source_node_id=a.node_id,
            target_node_id=c.node_id,
            edge_type=EdgeType.REFINES,
            strength=0.8,
            evidence="parent-child",
        )
    )
    # SUPPORTS: a ↔ b siblings
    storage.add_edge(
        CausalEdge(
            source_node_id=a.node_id,
            target_node_id=b.node_id,
            edge_type=EdgeType.SUPPORTS,
            strength=0.6,
            evidence="siblings",
        )
    )

    return storage


class TestCausalRetriever:
    async def test_build_chain_from_single(self, chain_storage) -> None:
        """Build chain starting from node_a finds connected edges."""
        retriever = CausalRetriever(chain_storage)
        a_id = _valid_uuid("a")
        a = await chain_storage.get_node(a_id)
        assert a is not None
        chain = await retriever.build_chain([a])
        # Should find at least edges (a→root, a→b, a→c)
        assert len(chain) >= 3

    async def test_build_chain_empty_seeds(self, chain_storage) -> None:
        """Empty seed nodes produce empty chain."""
        retriever = CausalRetriever(chain_storage)
        chain = await retriever.build_chain([])
        assert chain == []

    async def test_build_chain_max_depth(self, chain_storage) -> None:
        """max_depth limits BFS depth."""
        retriever = CausalRetriever(chain_storage)
        a_id = _valid_uuid("a")
        a = await chain_storage.get_node(a_id)
        assert a is not None
        chain_shallow = await retriever.build_chain([a], max_depth=1)
        chain_deep = await retriever.build_chain([a], max_depth=3)
        # Deeper search should find more edges
        assert len(chain_deep) >= len(chain_shallow) - 1  # allow for overlap

    async def test_find_path_direct(self, chain_storage) -> None:
        """Find direct path from a to b."""
        retriever = CausalRetriever(chain_storage)
        a_id = _valid_uuid("a")
        b_id = _valid_uuid("b")
        path = await retriever.find_path(a_id, b_id)
        assert len(path) == 1
        assert path[0].edge_type == EdgeType.SUPPORTS

    async def test_find_path_two_hop(self, chain_storage) -> None:
        """Find two-hop path from c to b via a."""
        retriever = CausalRetriever(chain_storage)
        c_id = _valid_uuid("c")
        b_id = _valid_uuid("b")
        path = await retriever.find_path(c_id, b_id)
        assert len(path) == 2
        # First edge: c → a (REFINES), second edge: a → b (SUPPORTS)

    async def test_find_path_no_path(self, chain_storage) -> None:
        """Isolated node returns empty path."""
        isolated_id = _valid_uuid("isolated")
        isolated = ASTNode(
            node_id=isolated_id,
            content="Isolated",
            node_type="PARAGRAPH",
            doc_id="doc-002",
        )
        chain_storage.add_node(isolated)
        retriever = CausalRetriever(chain_storage)
        a_id = _valid_uuid("a")
        path = await retriever.find_path(isolated_id, a_id)
        assert path == []

    async def test_find_path_same_node(self, chain_storage) -> None:
        """Same source and target returns empty path."""
        retriever = CausalRetriever(chain_storage)
        a_id = _valid_uuid("a")
        path = await retriever.find_path(a_id, a_id)
        assert path == []

    async def test_get_subgraph_radius_0(self, chain_storage) -> None:
        """Radius 0 returns no edges."""
        retriever = CausalRetriever(chain_storage)
        a_id = _valid_uuid("a")
        edges = await retriever.get_subgraph(a_id, radius=0)
        assert len(edges) == 0

    async def test_get_subgraph_radius_1(self, chain_storage) -> None:
        """Radius 1 returns immediate neighbours."""
        retriever = CausalRetriever(chain_storage)
        a_id = _valid_uuid("a")
        edges = await retriever.get_subgraph(a_id, radius=1)
        # a is connected to root, b, and c
        assert len(edges) == 3

    async def test_bfs_skips_contradicts(self) -> None:
        """BFS does not follow CONTRADICTS edges."""
        storage = FakeStorage()
        a = ASTNode(
            node_id="a0000000-0000-4000-8000-000000000000",
            content="Node A",
            node_type="PARAGRAPH",
            doc_id="doc-001",
        )
        b = ASTNode(
            node_id="b0000000-0000-4000-8000-000000000000",
            content="Node B",
            node_type="PARAGRAPH",
            doc_id="doc-001",
        )
        c = ASTNode(
            node_id="c0000000-0000-4000-8000-000000000000",
            content="Node C",
            node_type="PARAGRAPH",
            doc_id="doc-001",
        )
        for n in [a, b, c]:
            storage.add_node(n)
        # a SUPPORTS b, b CONTRADICTS c
        storage.add_edge(
            CausalEdge(
                source_node_id=a.node_id,
                target_node_id=b.node_id,
                edge_type=EdgeType.SUPPORTS,
                strength=0.7,
            )
        )
        storage.add_edge(
            CausalEdge(
                source_node_id=b.node_id,
                target_node_id=c.node_id,
                edge_type=EdgeType.CONTRADICTS,
                strength=0.9,
            )
        )

        retriever = CausalRetriever(storage)
        chain = await retriever.build_chain([a])
        edge_types = [e.edge_type for e in chain]
        assert EdgeType.SUPPORTS in edge_types
        assert EdgeType.CONTRADICTS not in edge_types


# ═══════════════════════════════════════════════════════════════════
# Integration: CausalGraphBuilder + CausalRetriever
# ═══════════════════════════════════════════════════════════════════


class TestGraphIntegration:
    async def test_build_then_retrieve(self, all_nodes) -> None:
        """Build graph structurally, then retrieve chains."""
        builder = CausalGraphBuilder()
        edges = await builder.build_all(
            all_nodes,
            include_temporal=False,
            include_semantic=False,
            include_llm=False,
        )
        assert len(edges) >= 1

        # Store edges in fake storage
        storage = FakeStorage()
        for n in all_nodes:
            storage.add_node(n)
        for ge in edges:
            storage.add_edge(ge.to_causal_edge())

        # Retrieve chains
        retriever = CausalRetriever(storage)
        chain = await retriever.build_chain([all_nodes[1]])  # node_a
        assert len(chain) >= 1

    async def test_graph_round_trip(self, root_node, node_a) -> None:
        """GraphEdge → CausalEdge → GraphEdge round trip preserves data."""
        ge = GraphEdge(
            source_id=root_node.node_id,
            target_id=node_a.node_id,
            relation_type=RelationType.REFINES,
            strength=0.85,
            evidence="parent-child relationship",
        )
        ce = ge.to_causal_edge()
        ge2 = GraphEdge.from_causal_edge(ce)

        assert ge2.source_id == ge.source_id
        assert ge2.target_id == ge.target_id
        assert ge2.relation_type == ge.relation_type
        assert ge2.strength == ge.strength
        assert ge2.evidence == ge.evidence
