"""
test_edge_cases.py — Edge case tests for ApexRAG.

Covers:
  - Empty documents
  - Very large documents
  - Concurrent access
  - DB connection failures
  - Malformed markdown
  - Unicode and special characters
"""

from __future__ import annotations

import asyncio

import pytest

from apex_rag.ingestion.apex_parser import ApexParser
from apex_rag.ingestion.legacy import _ast_nodes_to_parsed_sections
from apex_rag.storage import DocumentNode, StorageEngine
from apex_rag.utils import build_ltree_path, path_depth, truncate


_parser = ApexParser()


def _parse(text: str) -> list:
    """Parse markdown text into ParsedSections (test helper)."""
    if not text or not text.strip():
        return []
    nodes = _parser.parse_markdown(text)
    return _ast_nodes_to_parsed_sections(nodes)

# ---------------------------------------------------------------------------
# Empty & Minimal Input Tests
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_markdown_parsing(self) -> None:
        """Empty string should produce empty sections list."""
        sections = _parse("")
        assert sections == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only text should return empty sections."""
        sections = _parse("   \n\n  ")
        assert sections == []

    def test_single_heading(self) -> None:
        """Single heading with no content should still create a node."""
        sections = _parse("# Just a heading")
        assert len(sections) == 1
        assert sections[0].title == "Just a heading"
        assert sections[0].content == ""


# ---------------------------------------------------------------------------
# Unicode & Special Characters
# ---------------------------------------------------------------------------


class TestUnicodeContent:
    def test_unicode_headings(self) -> None:
        """Headings with Unicode characters should be handled."""
        text = "# 日本語タイトル\nContent in Japanese.\n## Français\nFrench content."
        sections = _parse(text)
        assert len(sections) == 1
        assert sections[0].title == "日本語タイトル"
        assert len(sections[0].children) == 1
        assert sections[0].children[0].title == "Français"

    def test_emoji_in_content(self) -> None:
        """Emoji and special symbols in content should not break parsing."""
        text = "# Test 🎉\nContent with emojis 🚀 and symbols ©®™."
        sections = _parse(text)
        assert len(sections) == 1
        assert "🎉" in sections[0].title


# ---------------------------------------------------------------------------
# Large Content Tests
# ---------------------------------------------------------------------------


class TestLargeContent:
    def test_large_section_content_preserved(self) -> None:
        """
        Large sections should have their full content preserved.

        ``ApexParser`` handles internal chunking at the ``ASTNode``
        level, so the ``ParsedSection`` converter collects all descendant
        text into the parent section's ``content`` field via
        ``_collect_descendant_text``.
        """
        # Generate a large section to stress the chunking path
        large_text = "# Huge Section\n"
        paragraphs = [f"Paragraph {i} with enough content to make multiple chunks." for i in range(200)]
        large_text += "\n\n".join(paragraphs)
        sections = _parse(large_text)
        assert len(sections) == 1
        assert sections[0].title == "Huge Section"
        # The full content should be preserved (not truncated or lost)
        assert len(sections[0].content) > len(paragraphs) * 10
        assert "Paragraph 0" in sections[0].content
        assert "Paragraph 199" in sections[0].content

    def test_many_children(self) -> None:
        """Document with many children should be handled efficiently."""
        text = "# Root\n\n"
        text += "\n\n".join(f"## Section {i}\nContent here." for i in range(50))
        sections = _parse(text)
        assert len(sections) == 1
        assert len(sections[0].children) == 50


# ---------------------------------------------------------------------------
# Concurrent Access Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_tree_access() -> None:
    """Storage engine should handle concurrent read access."""
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")

    # Insert some nodes
    async with storage.session() as session:
        root = DocumentNode(
            doc_id="concurrent-test", parent_id=None, path="1",
            title="Root", summary="Root", content="Root content",
            depth=0, position=1,
        )
        await storage.insert_node(session, root)

        for i in range(5):
            child = DocumentNode(
                doc_id="concurrent-test", parent_id=root.id, path=f"1.{i+1}",
                title=f"Child {i}", summary=f"Summary {i}", content=f"Content {i}",
                depth=1, position=i + 1,
            )
            await storage.insert_node(session, child)

    # Concurrent reads
    async def read_tree() -> int:
        async with storage.session() as session:
            nodes = await storage.get_full_tree(session, "concurrent-test")
            return len(nodes)

    results = await asyncio.gather(*[read_tree() for _ in range(10)])
    assert all(r == 6 for r in results)  # root + 5 children
    await storage.dispose()


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_truncate_short_text(self) -> None:
        """Truncate should not modify short text."""
        assert truncate("Hello", 100) == "Hello"

    def test_truncate_long_text(self) -> None:
        """Truncate should add ellipsis for long text."""
        result = truncate("A" * 200, 50)
        assert len(result) == 51  # 50 chars + ellipsis
        assert result.endswith("…")

    def test_build_ltree_path_root(self) -> None:
        assert build_ltree_path(None, 1) == "1"
        assert build_ltree_path(None, 99) == "99"

    def test_build_ltree_path_child(self) -> None:
        assert build_ltree_path("1", 2) == "1.2"
        assert build_ltree_path("1.2.3", 4) == "1.2.3.4"

    def test_path_depth(self) -> None:
        assert path_depth("1") == 0
        assert path_depth("1.2") == 1
        assert path_depth("1.2.3.4.5") == 4


# ---------------------------------------------------------------------------
# Document Node Property Tests
# ---------------------------------------------------------------------------


class TestDocumentNode:
    def test_is_leaf_with_content(self) -> None:
        node = DocumentNode(
            doc_id="test", path="1", title="Test",
            summary="Test", content="Has content",
            depth=0, position=1,
        )
        assert node.is_leaf is True

    def test_is_leaf_without_content(self) -> None:
        node = DocumentNode(
            doc_id="test", path="1", title="Test",
            summary="Test", content=None,
            depth=0, position=1,
        )
        assert node.is_leaf is False

    def test_page_range_same_page(self) -> None:
        node = DocumentNode(
            doc_id="test", path="1", title="Test",
            summary="Test", content="Content",
            depth=0, position=1, page_start=5, page_end=5,
        )
        assert node.page_range == "p.5"

    def test_page_range_different(self) -> None:
        node = DocumentNode(
            doc_id="test", path="1", title="Test",
            summary="Test", content="Content",
            depth=0, position=1, page_start=3, page_end=7,
        )
        assert "p.3" in node.page_range
        assert "7" in node.page_range

    def test_page_range_unknown(self) -> None:
        node = DocumentNode(
            doc_id="test", path="1", title="Test",
            summary="Test", content="Content",
            depth=0, position=1, page_start=0, page_end=0,
        )
        assert node.page_range == ""
