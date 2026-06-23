"""
Tests for Part 2 — AST ingestion engine.

Covers:
    - ApexParser (markdown, Python, tables, no headings)
    - SemanticModelBuilder (signpost generation)
    - EmbeddingEngine (batch embedding)
    - ApexStorage (SQLAlchemy CRUD)
    - Page marker extraction & chunking
    - Page index CRUD, semantic cache, global search

Test count: 35+
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from apex_rag.ingestion.apex_parser import ApexParser, _chunk_large_sections
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.ingestion.embedding_engine import EmbeddingEngine
from apex_rag.ingestion.semantic_model_builder import SemanticModelBuilder
from apex_rag.models.unified_models import (
    ASTNode,
    CausalEdge,
    EdgeType,
    EvidencePacket,
    NodeType,
    TemporalMetadata,
)


# ═══════════════════════════════════════════════════════════════
# ApexParser — Markdown tests
# ═══════════════════════════════════════════════════════════════


class TestApexParserMarkdown:
    """Tests for parsing Markdown / plain text into AST nodes."""

    def test_simple_paragraph(self) -> None:
        """A single paragraph becomes a PARAGRAPH node under an implicit root."""
        parser = ApexParser()
        nodes = parser.parse_markdown("Hello, world.")
        assert len(nodes) >= 1
        # Should have an implicit root heading + paragraph
        paras = [n for n in nodes if n.node_type == NodeType.PARAGRAPH]
        assert len(paras) == 1
        assert paras[0].content == "Hello, world."

    def test_three_heading_levels(self) -> None:
        """A document with three heading levels produces correct depths and hierarchy."""
        md = """# Chapter 1

Content under chapter.

## Section 1.1

Content under section.

### Subsection 1.1.1

Deep content.

## Section 1.2

More content.
"""
        parser = ApexParser()
        nodes = parser.parse_markdown(md)
        headings = [n for n in nodes if n.node_type == NodeType.HEADING]

        # Should have 4 headings: Chapter 1, Section 1.1, Subsection 1.1.1, Section 1.2
        assert len(headings) >= 4, f"Expected at least 4 headings, got {len(headings)}"

        # Find our specific headings
        ch1 = next(n for n in headings if "Chapter 1" in n.content)
        sec1 = next(n for n in headings if "Section 1.1" in n.content)
        sub1 = next(n for n in headings if "Subsection 1.1.1" in n.content)
        sec2 = next(n for n in headings if "Section 1.2" in n.content)

        # Depth checks
        assert ch1.depth == 0, f"Chapter should be depth 0, got {ch1.depth}"
        assert sec1.depth == 1, f"Section should be depth 1, got {sec1.depth}"
        assert sub1.depth == 2, f"Subsection should be depth 2, got {sub1.depth}"
        assert sec2.depth == 1, f"Section 1.2 should be depth 1, got {sec2.depth}"

        # Parent-child relationships
        assert sec1.parent_id == ch1.node_id, "Section 1.1 should be child of Chapter 1"
        assert sub1.parent_id == sec1.node_id, "Subsection should be child of Section 1.1"
        assert sec2.parent_id == ch1.node_id, "Section 1.2 should be child of Chapter 1"

        # Children list
        assert ch1.node_id in [n.node_id for n in nodes], "Chapter should have node_id"
        assert sec1.node_id in ch1.children, "Ch1 children should include Sec1"

    def test_code_block(self) -> None:
        """A fenced code block becomes a CODE node."""
        md = """# Code Example

Here is some code:

```python
def hello():
    print("world")
```

End of example."""
        parser = ApexParser()
        nodes = parser.parse_markdown(md)
        code_nodes = [n for n in nodes if n.node_type == NodeType.CODE]
        assert len(code_nodes) >= 1, "Expected at least one CODE node"
        code_content = code_nodes[0].content
        assert "def hello():" in code_content
        assert 'print("world")' in code_content

    def test_table_four_columns(self) -> None:
        """A table with 4 columns becomes a TABLE node preserving cell content."""
        md = """# Data Table

| Name | Age | City | Country |
|------|-----|------|---------|
| Alice| 30  | NYC  | USA     |
| Bob  | 25  | London | UK    |
| Carol| 35  | Tokyo| Japan   |
"""
        parser = ApexParser()
        nodes = parser.parse_markdown(md)
        table_nodes = [n for n in nodes if n.node_type == NodeType.TABLE]
        assert len(table_nodes) >= 1, "Expected at least one TABLE node"
        table_text = table_nodes[0].content
        assert "Name" in table_text
        assert "Age" in table_text
        assert "City" in table_text
        assert "Country" in table_text
        assert "Alice" in table_text
        assert "Bob" in table_text
        assert "Carol" in table_text

    def test_no_headings(self) -> None:
        """A document with no headings gets an implicit root heading."""
        parser = ApexParser()
        nodes = parser.parse_markdown(
            "Just some plain text without any headings whatsoever.\n\nSecond paragraph."
        )
        assert len(nodes) >= 2
        headings = [n for n in nodes if n.node_type == NodeType.HEADING]
        assert len(headings) >= 1, "Expected at least one heading (implicit root)"

        # All paragraphs should be children of the implicit root
        paras = [n for n in nodes if n.node_type == NodeType.PARAGRAPH]
        root = headings[0]
        for p in paras:
            assert p.parent_id == root.node_id, f"Paragraph should be child of implicit root"

    def test_mixed_content(self) -> None:
        """A document with headings, paragraphs, code, and tables parses correctly."""
        md = """# Overview

Introduction paragraph.

## Data

| Year | Value |
|------|-------|
| 2023 | 100   |

## Code

```bash
echo "hello"
```

That's all.
"""
        parser = ApexParser()
        nodes = parser.parse_markdown(md)

        headings = [n for n in nodes if n.node_type == NodeType.HEADING]
        paras = [n for n in nodes if n.node_type == NodeType.PARAGRAPH]
        tables = [n for n in nodes if n.node_type == NodeType.TABLE]
        code_blocks = [n for n in nodes if n.node_type == NodeType.CODE]

        assert len(headings) >= 2
        assert len(paras) >= 1
        assert len(tables) >= 1
        assert len(code_blocks) >= 1

    def test_empty_document(self) -> None:
        """An empty string produces a single implicit root node."""
        parser = ApexParser()
        nodes = parser.parse_markdown("")
        assert len(nodes) >= 1

    def test_parse_text_with_source_date(self) -> None:
        """source_date is propagated to all nodes."""
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        parser = ApexParser()
        nodes = parser.parse_markdown("# Test\n\nContent.", source_date=dt)
        for node in nodes:
            assert node.source_date == dt, f"source_date mismatch for {node.node_id}"
            assert node.ingestion_date is not None

    def test_doc_id_propagation(self) -> None:
        """doc_id is correctly set on all nodes."""
        parser = ApexParser(default_doc_id="my-doc")
        nodes = parser.parse_markdown("# Title\n\nBody.")
        for node in nodes:
            assert node.doc_id == "my-doc", f"doc_id mismatch for {node.node_id}"

    def test_node_id_is_uuid(self) -> None:
        """Every node gets a valid UUID4 node_id."""
        import uuid as uuid_mod

        parser = ApexParser()
        nodes = parser.parse_markdown("# H1\n\nP1.\n\n## H2\n\nP2.")
        for node in nodes:
            parsed = uuid_mod.UUID(node.node_id, version=4)
            assert str(parsed) == node.node_id


# ═══════════════════════════════════════════════════════════════
# Page marker extraction tests
# ═══════════════════════════════════════════════════════════════


class TestPageMarkerExtraction:
    """Tests for page marker extraction from Markdown."""

    def test_page_marker_html_comment(self) -> None:
        """<!-- Page 3 --> assigns page_number=3 to subsequent heading."""
        md = "Some text.\n<!-- Page 3 -->\n# Chapter Three\n\nContent."
        parser = ApexParser()
        nodes = parser.parse_markdown(md)
        ch3 = next(n for n in nodes if "Chapter Three" in n.content)
        assert ch3.page_number == 3, f"Expected page_number=3, got {ch3.page_number}"

    def test_page_marker_bracket(self) -> None:
        """[Page 5] assigns page_number=5 to subsequent heading."""
        md = "# Chapter One\n\nText.\n[Page 5]\n## Section 5A\n\nDetails."
        parser = ApexParser()
        nodes = parser.parse_markdown(md)
        sec5a = next(n for n in nodes if "Section 5A" in n.content)
        assert sec5a.page_number == 5, f"Expected page_number=5, got {sec5a.page_number}"

    def test_multiple_page_markers(self) -> None:
        """Multiple page markers in sequence track the highest page number."""
        md = "<!-- Page 1 -->\n# Intro\n\n<!-- Page 2 -->\n<!-- Page 3 -->\n# Main\n\nContent."
        parser = ApexParser()
        nodes = parser.parse_markdown(md)
        main = next(n for n in nodes if "Main" in n.content)
        assert main.page_number == 3, f"Expected page_number=3, got {main.page_number}"

    def test_page_marker_on_table(self) -> None:
        """Page marker before a table assigns page_number to that table node."""
        md = "<!-- Page 7 -->\n# Data\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        parser = ApexParser()
        nodes = parser.parse_markdown(md)
        tables = [n for n in nodes if n.node_type == NodeType.TABLE]
        assert len(tables) >= 1
        assert tables[0].page_number == 7, f"Expected page_number=7, got {tables[0].page_number}"

    def test_no_page_marker(self) -> None:
        """Document without page markers has page_number=None."""
        parser = ApexParser()
        nodes = parser.parse_markdown("# Title\n\nNo markers here.")
        for n in nodes:
            assert n.page_number is None, f"Expected None, got {n.page_number}"


# ═══════════════════════════════════════════════════════════════
# Large-section chunking tests
# ═══════════════════════════════════════════════════════════════


class TestLargeSectionChunking:
    """Tests for _chunk_large_sections splitting."""

    def test_small_section_not_chunked(self) -> None:
        """A section under max_chars is not chunked."""
        node = ASTNode(
            content="Short content",
            node_type=NodeType.PARAGRAPH,
            doc_id="d1",
        )
        nodes = [node]
        result = _chunk_large_sections(nodes, "d1", None, datetime.now(timezone.utc), 0, max_chars=3000)
        # No new nodes should be created
        assert len(result) == 1
        # Content should remain on original node
        assert result[0].content == "Short content"
        assert result[0].children == []

    def test_large_section_chunked(self) -> None:
        """A section exceeding max_chars is split into child nodes."""
        # Create a node with content > 100 chars
        large_content = "\n\n".join([f"Paragraph {i} contains enough text to fill a section." for i in range(20)])
        node = ASTNode(
            content=large_content,
            node_type=NodeType.PARAGRAPH,
            doc_id="d1",
        )
        nodes = [node]
        result = _chunk_large_sections(nodes, "d1", None, datetime.now(timezone.utc), 0, max_chars=100)

        # Should have more than 1 chunk
        chunks = [n for n in result if n.parent_id == node.node_id]
        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

        # Original node's content should be empty (pushed to children)
        updated_parent = next(n for n in result if n.node_id == node.node_id)
        assert updated_parent.content == ""
        assert len(updated_parent.children) >= 2

    def test_chunk_respects_max_chars(self) -> None:
        """Each chunk should be under max_chars."""
        large_content = "\n\n".join([f"Long paragraph number {i} with filler text to exceed limits." for i in range(30)])
        node = ASTNode(
            content=large_content,
            node_type=NodeType.HEADING,
            doc_id="d1",
        )
        nodes = [node]
        result = _chunk_large_sections(nodes, "d1", None, datetime.now(timezone.utc), 0, max_chars=200)
        chunks = [n for n in result if n.parent_id == node.node_id]
        for chunk in chunks:
            assert len(chunk.content) <= 220, f"Chunk exceeds max_chars: {len(chunk.content)}"

    def test_chunk_assigns_children_to_parent(self) -> None:
        """Chunked nodes are assigned as children of the original parent."""
        large_content = "\n\n".join([f"Paragraph {i} with enough text." for i in range(15)])
        node = ASTNode(
            content=large_content,
            node_type=NodeType.PARAGRAPH,
            doc_id="d1",
        )
        nodes = [node]
        result = _chunk_large_sections(nodes, "d1", None, datetime.now(timezone.utc), 0, max_chars=100)
        updated = next(n for n in result if n.node_id == node.node_id)
        assert len(updated.children) >= 2
        for child_id in updated.children:
            child = next((n for n in result if n.node_id == child_id), None)
            assert child is not None
            assert child.parent_id == node.node_id

    def test_chunk_with_page_number(self) -> None:
        """Chunked child nodes inherit page_number from the ingestion context."""
        large_content = "\n\n".join([f"Paragraph {i} data." for i in range(20)])
        node = ASTNode(
            content=large_content,
            node_type=NodeType.PARAGRAPH,
            doc_id="d1",
        )
        nodes = [node]
        result = _chunk_large_sections(nodes, "d1", None, datetime.now(timezone.utc), 7, max_chars=100)
        chunks = [n for n in result if n.parent_id == node.node_id]
        for chunk in chunks:
            assert chunk.page_number == 7, f"Expected page_number=7, got {chunk.page_number}"


# ═══════════════════════════════════════════════════════════════
# ApexParser — Python source tests
# ═══════════════════════════════════════════════════════════════


class TestApexParserPython:
    """Tests for parsing Python source code into AST nodes."""

    PYTHON_CODE = '''"""Module docstring."""

import os
import sys


class DataProcessor:
    """Processes data from various sources."""

    def __init__(self, source: str) -> None:
        self.source = source

    def load(self) -> list[str]:
        """Load data from the source."""
        return []

    def transform(self, data: list[str]) -> list[str]:
        """Transform the loaded data."""
        return data


class ReportGenerator:
    """Generates reports from processed data."""

    def generate(self, data: list[str], title: str = "Report") -> str:
        """Generate a report string from data."""
        return "\\n".join(data)

    async def stream(self, data: list[str]) -> str:
        """Stream-generated report."""
        return "streamed"

    def _helper(self) -> None:
        pass
'''

    def test_two_classes_five_methods(self) -> None:
        """A Python file with 2 classes and 5 methods produces correct nodes."""
        parser = ApexParser()
        nodes = parser.parse_python(self.PYTHON_CODE)

        headings = [n for n in nodes if n.node_type == NodeType.HEADING]
        code_nodes = [n for n in nodes if n.node_type == NodeType.CODE]

        # Should have 2 class nodes (HEADING) and 6 function nodes (5 methods + _helper)
        # Actually, we should have 2 classes as HEADING nodes
        assert len(headings) >= 2, f"Expected at least 2 class nodes, got {len(headings)}"

        class_names = [n.content for n in headings]
        assert any("DataProcessor" in c for c in class_names)
        assert any("ReportGenerator" in c for c in class_names)

        # Methods should be CODE nodes under the classes
        assert len(code_nodes) >= 5, f"Expected at least 5 method nodes, got {len(code_nodes)}"

        # Check parent-child relationships
        processor = next(n for n in headings if "DataProcessor" in n.content)
        generator = next(n for n in headings if "ReportGenerator" in n.content)

        # Methods should be children of their class
        for method in code_nodes:
            assert method.parent_id is not None, "Method should have a parent"
            parent_is_class = method.parent_id == processor.node_id or method.parent_id == generator.node_id
            assert parent_is_class, f"Method {method.content[:20]} should be child of a class"

    def test_empty_python_file(self) -> None:
        """An empty Python file produces no structural AST nodes."""
        parser = ApexParser()
        nodes = parser.parse_python("")
        # Empty Python source has no classes or functions
        assert len(nodes) == 0

    def test_syntax_error_fallback(self) -> None:
        """Invalid Python syntax falls back to plain text parsing."""
        parser = ApexParser()
        nodes = parser.parse_python("This is >>> not <<< valid Python code!!!")
        assert len(nodes) >= 1


# ═══════════════════════════════════════════════════════════════
# SemanticModelBuilder tests
# ═══════════════════════════════════════════════════════════════


class TestSemanticModelBuilder:
    """Tests for the SemanticModelBuilder signpost generation."""

    def test_constructor(self) -> None:
        """Can instantiate with a provider."""
        builder = SemanticModelBuilder(llm=_DummyProvider())
        assert builder is not None
        # Initialized with default max_concurrent=8
        assert builder._semaphore._value == 8

    @pytest.mark.asyncio
    async def test_build_signposts_empty(self) -> None:
        """Empty node list returns empty dict."""
        builder = SemanticModelBuilder(llm=_DummyProvider())
        result = await builder.build_signposts([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_build_signposts_no_children(self) -> None:
        """Nodes with no children produce no signposts (not non-leaf)."""
        builder = SemanticModelBuilder(llm=_DummyProvider())
        nodes = [
            ASTNode(content="Paragraph", node_type=NodeType.PARAGRAPH, doc_id="d1"),
        ]
        result = await builder.build_signposts(nodes)
        assert result == {}

    @pytest.mark.asyncio
    async def test_build_signposts_heading_with_children(self) -> None:
        """A heading with children produces a signpost."""
        builder = SemanticModelBuilder(llm=_DummyProvider())
        node = ASTNode(
            content="Chapter 1: Introduction",
            node_type=NodeType.HEADING,
            children=["child-id"],
            doc_id="d1",
        )
        result = await builder.build_signposts([node])
        assert node.node_id in result
        assert len(result[node.node_id]) > 0

    @pytest.mark.asyncio
    async def test_build_signpost_single_node(self) -> None:
        """build_signpost works for a single node."""
        builder = SemanticModelBuilder(llm=_DummyProvider())
        node = ASTNode(
            content="Appendix",
            node_type=NodeType.HEADING,
            children=["child-1"],
            doc_id="d1",
        )
        signpost = await builder.build_signpost(node)
        assert len(signpost) > 0

    def test_node_preview(self) -> None:
        """_node_preview returns heading and content preview."""
        builder = SemanticModelBuilder(llm=_DummyProvider())
        node = ASTNode(
            content="My Heading",
            node_type=NodeType.HEADING,
            doc_id="d1",
        )
        heading, preview = builder._node_preview(node)
        assert heading == "My Heading"
        assert preview is not None


class _DummyProvider:
    """A minimal provider that returns a fixed signpost, for testing."""

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        return "This section covers the key concepts and definitions."


# ═══════════════════════════════════════════════════════════════
# ApexParser — File parsing tests
# ═══════════════════════════════════════════════════════════════


class TestApexParserFile:
    """Tests for parsing from file paths."""

    def test_parse_markdown_file(self) -> None:
        """Parse a .md file from disk."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# File Test\n\nContent from file.")
            f.flush()
            fname = f.name

        try:
            parser = ApexParser()
            nodes = parser.parse_markdown(
                Path(fname).read_text(encoding="utf-8")
            )
            assert len(nodes) >= 2
            headings = [n for n in nodes if n.node_type == NodeType.HEADING]
            assert any("File Test" in n.content for n in headings)
        finally:
            os.unlink(fname)

    def test_parse_python_file(self) -> None:
        """Parse a .py file from disk."""
        code = "class A:\\n    def method(self): pass\\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            f.flush()
            fname = f.name

        try:
            parser = ApexParser()
            nodes = parser.parse_python(Path(fname).read_text(encoding="utf-8"))
            assert len(nodes) >= 1
        finally:
            os.unlink(fname)

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for non-existent files."""
        parser = ApexParser()
        with pytest.raises(FileNotFoundError):
            # parse_file is async, but the error happens before any await
            import asyncio
            asyncio.run(parser.parse_file("/nonexistent/file.md"))


# ═══════════════════════════════════════════════════════════════
# EmbeddingEngine tests
# ═══════════════════════════════════════════════════════════════


class TestEmbeddingEngine:
    """Tests for the EmbeddingEngine."""

    @pytest.mark.asyncio
    async def test_embed_nodes(self) -> None:
        """Embedding engine populates the embedding field on all nodes."""
        nodes = [
            ASTNode(content="Hello world", node_type=NodeType.PARAGRAPH, doc_id="d1"),
            ASTNode(content="Second node", node_type=NodeType.PARAGRAPH, doc_id="d1"),
        ]
        engine = EmbeddingEngine()  # No embedder → uses fingerprint fallback
        result = await engine.embed_nodes(nodes)
        assert len(result) == 2
        for node in result:
            assert len(node.embedding) == 384, f"Expected 384-d embedding, got {len(node.embedding)}"
            assert any(v != 0.0 for v in node.embedding[:5]), "Embedding should be non-zero"

    @pytest.mark.asyncio
    async def test_embed_nodes_empty(self) -> None:
        """Empty node list returns empty list."""
        engine = EmbeddingEngine()
        result = await engine.embed_nodes([])
        assert result == []

    def test_fingerprint_deterministic(self) -> None:
        """Same text produces the same fingerprint."""
        engine = EmbeddingEngine()
        emb1 = engine._fingerprint("Hello world")
        emb2 = engine._fingerprint("Hello world")
        assert emb1 == emb2

    def test_fingerprint_different_text(self) -> None:
        """Different texts produce different fingerprints."""
        engine = EmbeddingEngine()
        emb1 = engine._fingerprint("Hello world")
        emb2 = engine._fingerprint("Goodbye world")
        assert emb1 != emb2

    @pytest.mark.asyncio
    async def test_embed_texts(self) -> None:
        """embed_texts works correctly."""
        engine = EmbeddingEngine()
        embeddings = await engine.embed_texts(["text one", "text two"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    @pytest.mark.asyncio
    async def test_embed_nodes_mutates_in_place(self) -> None:
        """embed_nodes mutates the embedding field of the passed nodes."""
        node = ASTNode(content="Test", node_type=NodeType.PARAGRAPH, doc_id="d1")
        engine = EmbeddingEngine()
        await engine.embed_nodes([node])
        assert len(node.embedding) == 384


# ═══════════════════════════════════════════════════════════════
# ApexStorage tests
# ═══════════════════════════════════════════════════════════════


class TestApexStorage:
    """Tests for ApexStorage CRUD operations."""

    @pytest.fixture
    async def storage(self) -> ApexStorage:
        """Create an in-memory SQLite storage for testing."""
        s = await ApexStorage.create("sqlite+aiosqlite://", echo=False)
        yield s
        await s.drop_all()
        await s.dispose()

    @pytest.mark.asyncio
    async def test_save_and_get_node(self, storage: ApexStorage) -> None:
        """Save a node and retrieve it by ID."""
        node = ASTNode(
            content="Test content",
            node_type=NodeType.PARAGRAPH,
            doc_id="doc-123",
        )
        await storage.save_node(node, tenant_context="default")

        retrieved = await storage.get_node(node.node_id, tenant_context="default")
        assert retrieved is not None
        assert retrieved.content == "Test content"
        assert retrieved.node_type == NodeType.PARAGRAPH
        assert retrieved.doc_id == "doc-123"

    @pytest.mark.asyncio
    async def test_save_nodes_bulk(self, storage: ApexStorage) -> None:
        """Save multiple nodes at once."""
        nodes = [
            ASTNode(content=f"Node {i}", node_type=NodeType.PARAGRAPH, doc_id="doc-1")
            for i in range(5)
        ]
        await storage.save_nodes(nodes, tenant_context="default")
        count = await storage.count_nodes("doc-1")
        assert count == 5

    @pytest.mark.asyncio
    async def test_get_nodes_by_doc(self, storage: ApexStorage) -> None:
        """Get all nodes for a given document."""
        nodes = [
            ASTNode(content="A", node_type=NodeType.PARAGRAPH, doc_id="doc-a"),
            ASTNode(content="B", node_type=NodeType.PARAGRAPH, doc_id="doc-a"),
            ASTNode(content="C", node_type=NodeType.PARAGRAPH, doc_id="doc-b"),
        ]
        await storage.save_nodes(nodes, tenant_context="default")

        doc_a_nodes = await storage.get_nodes_by_doc("doc-a", tenant_context="default")
        assert len(doc_a_nodes) == 2

        doc_b_nodes = await storage.get_nodes_by_doc("doc-b", tenant_context="default")
        assert len(doc_b_nodes) == 1

    @pytest.mark.asyncio
    async def test_delete_node(self, storage: ApexStorage) -> None:
        """Delete a node by ID."""
        node = ASTNode(
            content="To be deleted",
            node_type=NodeType.PARAGRAPH,
            doc_id="doc-1",
        )
        await storage.save_node(node, tenant_context="default")
        assert await storage.get_node(node.node_id, tenant_context="default") is not None

        deleted = await storage.delete_node(node.node_id, tenant_context="default")
        assert deleted is True
        assert await storage.get_node(node.node_id, tenant_context="default") is None

        # Delete non-existent
        assert await storage.delete_node(str(uuid.uuid4()), tenant_context="default") is False

    @pytest.mark.asyncio
    async def test_temporal_metadata_crud(self, storage: ApexStorage) -> None:
        """Save and retrieve temporal metadata."""
        node = ASTNode(
            content="Node with temporal data",
            node_type=NodeType.PARAGRAPH,
            doc_id="doc-1",
        )
        await storage.save_node(node, tenant_context="default")

        meta = TemporalMetadata(
            node_id=node.node_id,
            freshness_score=0.85,
            decay_rate=0.002,
        )
        await storage.save_temporal_metadata(meta, tenant_context="default")

        retrieved = await storage.get_temporal_metadata(node.node_id)
        assert retrieved is not None
        assert retrieved.freshness_score == 0.85
        assert retrieved.decay_rate == 0.002

    @pytest.mark.asyncio
    async def test_causal_edge_crud(self, storage: ApexStorage) -> None:
        """Save and retrieve causal edges."""
        node_a = ASTNode(content="A", node_type=NodeType.PARAGRAPH, doc_id="doc-1")
        node_b = ASTNode(content="B", node_type=NodeType.PARAGRAPH, doc_id="doc-1")
        await storage.save_nodes([node_a, node_b], tenant_context="default")

        edge = CausalEdge(
            source_node_id=node_a.node_id,
            target_node_id=node_b.node_id,
            edge_type=EdgeType.SUPPORTS,
            strength=0.9,
            evidence="They agree.",
        )
        await storage.save_causal_edge(edge)

        edges = await storage.get_edges_for_node(node_a.node_id)
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.SUPPORTS
        assert edges[0].strength == 0.9
        assert edges[0].evidence == "They agree."

    @pytest.mark.asyncio
    async def test_node_with_children_persistence(self, storage: ApexStorage) -> None:
        """Save a parent node with children references and retrieve correctly."""
        parent = ASTNode(
            content="Parent",
            node_type=NodeType.HEADING,
            doc_id="doc-1",
        )
        child = ASTNode(
            content="Child",
            node_type=NodeType.PARAGRAPH,
            parent_id=parent.node_id,
            doc_id="doc-1",
        )
        parent.children = [child.node_id]

        await storage.save_nodes([parent, child], tenant_context="default")

        retrieved_parent = await storage.get_node(parent.node_id, tenant_context="default")
        assert retrieved_parent is not None
        assert child.node_id in retrieved_parent.children

        retrieved_child = await storage.get_node(child.node_id, tenant_context="default")
        assert retrieved_child is not None
        assert retrieved_child.parent_id == parent.node_id

    @pytest.mark.asyncio
    async def test_storage_count_nodes(self, storage: ApexStorage) -> None:
        """Count nodes works correctly."""
        nodes = [
            ASTNode(content="1", node_type=NodeType.PARAGRAPH, doc_id="d1"),
            ASTNode(content="2", node_type=NodeType.PARAGRAPH, doc_id="d1"),
            ASTNode(content="3", node_type=NodeType.PARAGRAPH, doc_id="d2"),
        ]
        await storage.save_nodes(nodes, tenant_context="default")
        assert await storage.count_nodes() == 3
        assert await storage.count_nodes("d1") == 2
        assert await storage.count_nodes("d2") == 1
        assert await storage.count_nodes("d3") == 0

    @pytest.mark.asyncio
    async def test_get_all_nodes(self, storage: ApexStorage) -> None:
        """Get all nodes across documents."""
        nodes = [
            ASTNode(content="1", node_type=NodeType.PARAGRAPH, doc_id="d1"),
            ASTNode(content="2", node_type=NodeType.PARAGRAPH, doc_id="d2"),
        ]
        await storage.save_nodes(nodes, tenant_context="default")
        all_nodes = await storage.get_all_nodes()
        assert len(all_nodes) == 2

    @pytest.mark.asyncio
    async def test_get_all_edges(self, storage: ApexStorage) -> None:
        """Get all edges across the graph."""
        node_a = ASTNode(content="A", node_type=NodeType.PARAGRAPH, doc_id="d1")
        node_b = ASTNode(content="B", node_type=NodeType.PARAGRAPH, doc_id="d1")
        node_c = ASTNode(content="C", node_type=NodeType.PARAGRAPH, doc_id="d1")
        await storage.save_nodes([node_a, node_b, node_c], tenant_context="default")

        edges = [
            CausalEdge(source_node_id=node_a.node_id, target_node_id=node_b.node_id, edge_type=EdgeType.SUPPORTS),
            CausalEdge(source_node_id=node_b.node_id, target_node_id=node_c.node_id, edge_type=EdgeType.REFINES),
        ]
        for e in edges:
            await storage.save_causal_edge(e)

        all_edges = await storage.get_all_edges()
        assert len(all_edges) == 2


# ═══════════════════════════════════════════════════════════════
# ApexStorage — Page index CRUD tests
# ═══════════════════════════════════════════════════════════════


class TestApexStoragePageIndex:
    """Tests for PageIndexEntry CRUD operations."""

    @pytest.fixture
    async def storage(self) -> ApexStorage:
        s = await ApexStorage.create("sqlite+aiosqlite://", echo=False)
        yield s
        await s.drop_all()
        await s.dispose()

    @pytest.mark.asyncio
    async def test_save_and_retrieve_single_entry(self, storage: ApexStorage) -> None:
        """Save a page index entry and retrieve it."""
        node = ASTNode(content="Test", node_type=NodeType.HEADING, doc_id="doc-1")
        await storage.save_node(node, tenant_context="default")

        entry = {
            "node_id": node.node_id,
            "doc_id": "doc-1",
            "term": "Introduction",
            "page_number": 3,
        }
        await storage.save_page_index_entry(entry)

        entries = await storage.get_page_index_entries("doc-1")
        assert len(entries) == 1
        assert entries[0]["term"] == "Introduction"
        assert entries[0]["page_number"] == 3

    @pytest.mark.asyncio
    async def test_save_and_retrieve_multiple_entries(self, storage: ApexStorage) -> None:
        """Multiple entries are stored and retrieved, sorted by term."""
        node = ASTNode(content="Root", node_type=NodeType.HEADING, doc_id="doc-1")
        await storage.save_node(node, tenant_context="default")

        entries = [
            {"node_id": node.node_id, "doc_id": "doc-1", "term": "Zebra", "page_number": 10},
            {"node_id": node.node_id, "doc_id": "doc-1", "term": "Alpha", "page_number": 1},
            {"node_id": node.node_id, "doc_id": "doc-1", "term": "Beta", "page_number": 5},
        ]
        await storage.save_page_index_entries(entries)

        retrieved = await storage.get_page_index_entries("doc-1")
        assert len(retrieved) == 3
        assert retrieved[0]["term"] == "Alpha"
        assert retrieved[1]["term"] == "Beta"
        assert retrieved[2]["term"] == "Zebra"

    @pytest.mark.asyncio
    async def test_search_page_index(self, storage: ApexStorage) -> None:
        """Search page index by term partial match (case-insensitive)."""
        node = ASTNode(content="Root", node_type=NodeType.HEADING, doc_id="doc-1")
        await storage.save_node(node, tenant_context="default")

        entries = [
            {"node_id": node.node_id, "doc_id": "doc-1", "term": "Revenue Growth", "page_number": 12},
            {"node_id": node.node_id, "doc_id": "doc-1", "term": "Cost Analysis", "page_number": 25},
            {"node_id": node.node_id, "doc_id": "doc-1", "term": "Revenue Forecast", "page_number": 30},
        ]
        await storage.save_page_index_entries(entries)

        results = await storage.search_page_index("doc-1", "revenue", tenant_context="default")
        assert len(results) == 2, f"Expected 2 matches for 'revenue', got {len(results)}"
        assert all("Revenue" in r["term"] for r in results)

    @pytest.mark.asyncio
    async def test_search_page_index_no_match(self, storage: ApexStorage) -> None:
        """Search with no matches returns empty list."""
        node = ASTNode(content="Root", node_type=NodeType.HEADING, doc_id="doc-1")
        await storage.save_node(node, tenant_context="default")

        await storage.save_page_index_entry({
            "node_id": node.node_id, "doc_id": "doc-1", "term": "Only Entry", "page_number": 1
        })

        results = await storage.search_page_index("doc-1", "nonexistent", tenant_context="default")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_page_index_empty_doc(self, storage: ApexStorage) -> None:
        """Document with no entries returns empty list."""
        entries = await storage.get_page_index_entries("empty-doc")
        assert len(entries) == 0


# ═══════════════════════════════════════════════════════════════
# ApexStorage — Semantic cache tests
# ═══════════════════════════════════════════════════════════════


class TestApexStorageCache:
    """Tests for semantic cache CRUD operations."""

    @pytest.fixture
    async def storage(self) -> ApexStorage:
        s = await ApexStorage.create("sqlite+aiosqlite://", echo=False)
        yield s
        await s.drop_all()
        await s.dispose()

    @pytest.mark.asyncio
    async def test_cache_and_retrieve(self, storage: ApexStorage) -> None:
        """Save a cache entry and retrieve it by query_hash."""
        qhash = hashlib.md5(b"test query").hexdigest()
        await storage.cache_query_result(
            query_hash=qhash,
            query_text="test query",
            doc_id="doc-1",
            node_ids=["node-1", "node-2"],
        )

        cached = await storage.get_cached_query(qhash, "doc-1")
        assert cached is not None
        assert cached["query"] == "test query"
        assert cached["doc_id"] == "doc-1"
        assert cached["node_ids"] == ["node-1", "node-2"]

    @pytest.mark.asyncio
    async def test_cache_miss(self, storage: ApexStorage) -> None:
        """Non-existent hash returns None."""
        cached = await storage.get_cached_query("nonexistent-hash", "doc-x")
        assert cached is None

    @pytest.mark.asyncio
    async def test_delete_cache(self, storage: ApexStorage) -> None:
        """Delete a cache entry."""
        qhash = hashlib.md5(b"delete me").hexdigest()
        await storage.cache_query_result(qhash, "delete me", "doc-1", [])
        assert await storage.get_cached_query(qhash, "doc-1") is not None

        deleted = await storage.delete_cached_query(qhash, "doc-1")
        assert deleted is True
        assert await storage.get_cached_query(qhash, "doc-1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_cache(self, storage: ApexStorage) -> None:
        """Delete nonexistent cache returns False."""
        assert await storage.delete_cached_query("no-such-hash", "doc-x") is False


# ═══════════════════════════════════════════════════════════════
# ApexStorage — Global search & document operations
# ═══════════════════════════════════════════════════════════════


class TestApexStorageGlobalSearch:
    """Tests for global search and document-level operations."""

    @pytest.fixture
    async def storage(self) -> ApexStorage:
        s = await ApexStorage.create("sqlite+aiosqlite://", echo=False)
        yield s
        await s.drop_all()
        await s.dispose()

    @pytest.mark.asyncio
    async def test_list_document_ids(self, storage: ApexStorage) -> None:
        """List all document IDs."""
        nodes = [
            ASTNode(content="A", node_type=NodeType.PARAGRAPH, doc_id="doc-1"),
            ASTNode(content="B", node_type=NodeType.PARAGRAPH, doc_id="doc-2"),
        ]
        await storage.save_nodes(nodes, tenant_context="default")

        doc_ids = await storage.list_document_ids(tenant_context="default")
        assert "doc-1" in doc_ids
        assert "doc-2" in doc_ids

    @pytest.mark.asyncio
    async def test_get_document_root_nodes(self, storage: ApexStorage) -> None:
        """Fetch root-level nodes for a document."""
        root = ASTNode(content="Root", node_type=NodeType.HEADING, doc_id="doc-1")
        child = ASTNode(content="Child", node_type=NodeType.PARAGRAPH, parent_id=root.node_id, doc_id="doc-1")
        await storage.save_nodes([root, child], tenant_context="default")

        roots = await storage.get_document_root_nodes("doc-1")
        assert len(roots) == 1
        assert roots[0].node_id == root.node_id

    @pytest.mark.asyncio
    async def test_search_nodes_global(self, storage: ApexStorage) -> None:
        """Search across all documents."""
        nodes = [
            ASTNode(content="Quarterly revenue increased", node_type=NodeType.PARAGRAPH, doc_id="doc-1"),
            ASTNode(content="Operating costs stable", node_type=NodeType.PARAGRAPH, doc_id="doc-1"),
            ASTNode(content="Revenue outlook positive", node_type=NodeType.PARAGRAPH, doc_id="doc-2"),
        ]
        await storage.save_nodes(nodes, tenant_context="default")

        results = await storage.search_nodes_global("revenue", tenant_context="default")
        assert len(results) == 2, f"Expected 2 matches for 'revenue', got {len(results)}"
        assert all("revenue" in n.content.lower() for n in results)

    @pytest.mark.asyncio
    async def test_get_document_stats(self, storage: ApexStorage) -> None:
        """Get aggregate stats for a document."""
        root = ASTNode(content="Root", node_type=NodeType.HEADING, doc_id="doc-1", depth=0)
        leaf1 = ASTNode(content="Leaf 1", node_type=NodeType.PARAGRAPH, doc_id="doc-1", depth=1, parent_id=root.node_id)
        leaf2 = ASTNode(content="Leaf 2", node_type=NodeType.PARAGRAPH, doc_id="doc-1", depth=1, parent_id=root.node_id)
        root.children = [leaf1.node_id, leaf2.node_id]
        await storage.save_nodes([root, leaf1, leaf2], tenant_context="default")

        stats = await storage.get_document_stats("doc-1", tenant_context="default")
        assert stats["doc_id"] == "doc-1"
        assert stats["total_nodes"] == 3
        assert stats["max_depth"] == 1
        assert stats["leaf_count"] == 2

    @pytest.mark.asyncio
    async def test_delete_document(self, storage: ApexStorage) -> None:
        """Delete a document and all its nodes."""
        nodes = [
            ASTNode(content="N1", node_type=NodeType.PARAGRAPH, doc_id="doc-1"),
            ASTNode(content="N2", node_type=NodeType.PARAGRAPH, doc_id="doc-1"),
            ASTNode(content="N3", node_type=NodeType.PARAGRAPH, doc_id="doc-2"),
        ]
        await storage.save_nodes(nodes, tenant_context="default")

        deleted = await storage.delete_document("doc-1", tenant_context="default")
        assert deleted == 2
        assert await storage.count_nodes("doc-1") == 0
        assert await storage.count_nodes("doc-2") == 1
