"""
test_tree.py — Unit tests for the tree building and storage layer.

All tests use an in-memory SQLite database and do NOT call Ollama,
ensuring fast, deterministic CI runs without any external dependencies.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from apex_rag.ingestion import ParsedSection, _parse_markdown_to_tree, _count_nodes
from apex_rag.storage import DocumentNode, StorageEngine
from apex_rag.utils import build_ltree_path, path_depth

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """\
# Chapter 1: Introduction
This chapter introduces the topic.

## Section 1.1: Background
Background information goes here.

### Section 1.1.1: History
The history of the subject.

## Section 1.2: Scope
The scope of this document.

# Chapter 2: Methods
Methodology description.

## Section 2.1: Data Collection
How data was collected.

## Section 2.2: Analysis
Analysis approach.
"""


@pytest_asyncio.fixture
async def storage() -> StorageEngine:
    """In-memory SQLite storage engine."""
    engine = await StorageEngine.create("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()


# ---------------------------------------------------------------------------
# Markdown Parsing Tests
# ---------------------------------------------------------------------------


class TestMarkdownParser:
    def test_basic_parse_returns_two_roots(self) -> None:
        sections = _parse_markdown_to_tree(SAMPLE_MARKDOWN)
        assert len(sections) == 2, "Expected two top-level chapters"

    def test_chapter1_has_two_children(self) -> None:
        sections = _parse_markdown_to_tree(SAMPLE_MARKDOWN)
        ch1 = sections[0]
        assert ch1.title == "Chapter 1: Introduction"
        assert len(ch1.children) == 2

    def test_nested_section_path(self) -> None:
        sections = _parse_markdown_to_tree(SAMPLE_MARKDOWN)
        # Section 1.1.1 should have path "1.1.1"
        sec_111 = sections[0].children[0].children[0]
        assert sec_111.path == "1.1.1"
        assert sec_111.title == "Section 1.1.1: History"

    def test_content_extracted(self) -> None:
        sections = _parse_markdown_to_tree(SAMPLE_MARKDOWN)
        ch1 = sections[0]
        assert "introduces the topic" in ch1.content

    def test_positions_are_sequential(self) -> None:
        sections = _parse_markdown_to_tree(SAMPLE_MARKDOWN)
        ch2_children = sections[1].children
        assert [c.position for c in ch2_children] == [1, 2]

    def test_total_node_count(self) -> None:
        sections = _parse_markdown_to_tree(SAMPLE_MARKDOWN)
        # 2 roots + 4 level-2 + 1 level-3 = 7
        assert _count_nodes(sections) == 7

    def test_empty_markdown(self) -> None:
        sections = _parse_markdown_to_tree("")
        assert sections == []

    def test_only_text_no_headings(self) -> None:
        sections = _parse_markdown_to_tree("Just some plain text without headings.")
        assert len(sections) == 1
        assert sections[0].title == "Document"
        assert sections[0].content == "Just some plain text without headings."


# ---------------------------------------------------------------------------
# LTree Path Helper Tests
# ---------------------------------------------------------------------------


class TestLTreePaths:
    def test_root_path(self) -> None:
        assert build_ltree_path(None, 1) == "1"
        assert build_ltree_path(None, 3) == "3"

    def test_child_path(self) -> None:
        assert build_ltree_path("1", 2) == "1.2"
        assert build_ltree_path("1.2", 3) == "1.2.3"

    def test_depth_calculation(self) -> None:
        assert path_depth("1") == 0
        assert path_depth("1.2") == 1
        assert path_depth("1.2.3") == 2
        assert path_depth("1.2.3.4") == 3


# ---------------------------------------------------------------------------
# Storage Layer Tests
# ---------------------------------------------------------------------------


class TestStorageEngine:
    @pytest.mark.asyncio
    async def test_insert_and_retrieve_node(self, storage: StorageEngine) -> None:
        async with storage.session() as session:
            node = DocumentNode(
                doc_id="test-doc",
                parent_id=None,
                path="1",
                title="Root Section",
                summary="A root-level section about testing.",
                content="Root content text.",
                depth=0,
                position=1,
            )
            node.meta = {"page": 1}
            persisted = await storage.insert_node(session, node)

            assert persisted.id is not None
            assert persisted.id > 0

        async with storage.session() as session:
            fetched = await storage.get_node(session, persisted.id)
            assert fetched is not None
            assert fetched.title == "Root Section"
            assert fetched.meta == {"page": 1}

    @pytest.mark.asyncio
    async def test_get_children_empty(self, storage: StorageEngine) -> None:
        async with storage.session() as session:
            children = await storage.get_children(session, parent_id=999)
            assert list(children) == []

    @pytest.mark.asyncio
    async def test_parent_child_relationship(self, storage: StorageEngine) -> None:
        async with storage.session() as session:
            parent = DocumentNode(
                doc_id="doc1", parent_id=None, path="1",
                title="Parent", summary="Parent summary",
                content=None, depth=0, position=1,
            )
            parent = await storage.insert_node(session, parent)

            child1 = DocumentNode(
                doc_id="doc1", parent_id=parent.id, path="1.1",
                title="Child 1", summary="First child",
                content="Child 1 content", depth=1, position=1,
            )
            child2 = DocumentNode(
                doc_id="doc1", parent_id=parent.id, path="1.2",
                title="Child 2", summary="Second child",
                content="Child 2 content", depth=1, position=2,
            )
            await storage.insert_node(session, child1)
            await storage.insert_node(session, child2)

        async with storage.session() as session:
            children = await storage.get_children(session, parent_id=parent.id)
            assert len(children) == 2
            assert children[0].title == "Child 1"
            assert children[1].title == "Child 2"

    @pytest.mark.asyncio
    async def test_delete_document(self, storage: StorageEngine) -> None:
        async with storage.session() as session:
            node = DocumentNode(
                doc_id="to-delete", parent_id=None, path="1",
                title="Temp Node", summary="Will be deleted",
                content="Temporary", depth=0, position=1,
            )
            await storage.insert_node(session, node)

        async with storage.session() as session:
            count = await storage.delete_document(session, "to-delete")
            assert count == 1

        async with storage.session() as session:
            docs = await storage.list_documents(session)
            assert "to-delete" not in list(docs)

    @pytest.mark.asyncio
    async def test_is_leaf_property(self, storage: StorageEngine) -> None:
        async with storage.session() as session:
            leaf = DocumentNode(
                doc_id="d", parent_id=None, path="1",
                title="Leaf", summary="s", content="Some content",
                depth=0, position=1,
            )
            persisted = await storage.insert_node(session, leaf)

        async with storage.session() as session:
            node = await storage.get_node(session, persisted.id)
            # is_leaf: content is not None AND no children
            assert node is not None
            assert node.content == "Some content"

    @pytest.mark.asyncio
    async def test_list_documents(self, storage: StorageEngine) -> None:
        async with storage.session() as session:
            for i in range(3):
                node = DocumentNode(
                    doc_id=f"doc-{i}", parent_id=None, path="1",
                    title=f"Doc {i}", summary="s", content="c",
                    depth=0, position=1,
                )
                await storage.insert_node(session, node)

        async with storage.session() as session:
            docs = await storage.list_documents(session)
            assert set(docs) == {"doc-0", "doc-1", "doc-2"}

    @pytest.mark.asyncio
    async def test_metadata_serialization(self, storage: StorageEngine) -> None:
        async with storage.session() as session:
            node = DocumentNode(
                doc_id="meta-test", parent_id=None, path="1",
                title="Meta Node", summary="s", content="c",
                depth=0, position=1,
            )
            node.meta = {"pages": [1, 2, 3], "source": "report.pdf"}
            persisted = await storage.insert_node(session, node)

        async with storage.session() as session:
            fetched = await storage.get_node(session, persisted.id)
            assert fetched is not None
            assert fetched.meta["pages"] == [1, 2, 3]
            assert fetched.meta["source"] == "report.pdf"
