import pytest

from apex_rag.ingestion.parsers.markdown import MarkdownASTParser
from apex_rag.retrieval.deterministic.keyword import KeywordDeterministicRetriever


@pytest.mark.asyncio
async def test_keyword_retriever():
    md_text = """# Company Report
This is a document about our company.

## Financials
In Q3, revenue was up by 20%.

## Engineering
The engineering team shipped 5 new features.
"""
    parser = MarkdownASTParser()
    root = await parser.parse("dummy.md", raw_text=md_text)

    retriever = KeywordDeterministicRetriever()

    # Query matching the "Financials" section
    results = await retriever.retrieve("What was the Q3 revenue?", root, top_k=2)

    assert len(results) > 0
    top_node = results[0]

    # The paragraph should match best because it contains Q3 and revenue
    assert "Q3" in top_node.content
    assert top_node.node_type == "Paragraph"

    # Query matching the Engineering section heading
    results2 = await retriever.retrieve("engineering", root, top_k=2)
    assert len(results2) > 0

    # Because of the 5.0x heading boost, the Section node "Engineering" should score higher
    # than the paragraph "The engineering team..."
    assert results2[0].node_type == "Section"
    assert results2[0].content == "Engineering"
