import pytest

from apex_rag.ingestion.parsers.markdown import MarkdownASTParser


@pytest.mark.asyncio
async def test_markdown_ast_parser_hierarchy():
    md_text = """# Main Title
This is a root paragraph.

## Section 1
Content of section 1.

### Subsection 1.1
Deeply nested content.

## Section 2
Content of section 2.
"""
    parser = MarkdownASTParser()
    root = await parser.parse("dummy.md", raw_text=md_text)

    assert root.node_type == "Document"
    assert len(root.children) == 1
    main_title = root.children[0]
    assert main_title.node_type == "Section"
    assert main_title.content == "Main Title"

    assert len(main_title.children) == 3 # Paragraph, Section 1, Section 2

    assert main_title.children[0].node_type == "Paragraph"
    assert main_title.children[0].content == "This is a root paragraph."

    sec1 = main_title.children[1]
    assert sec1.node_type == "Section"
    assert sec1.content == "Section 1"
    assert len(sec1.children) == 2 # Paragraph, Subsection 1.1

    sec2 = main_title.children[2]
    assert sec2.node_type == "Section"
    assert sec2.content == "Section 2"
    assert len(sec2.children) == 1 # Paragraph
