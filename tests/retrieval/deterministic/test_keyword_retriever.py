import pytest

from apex_rag.ingestion.apex_parser import ApexParser
from apex_rag.models.unified_models import NodeType
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
    parser = ApexParser()
    nodes = parser.parse_markdown(md_text, doc_id="dummy-doc")

    # We need to link the flat nodes into a tree structure for the retriever
    # or just use the first node (which the parser returns as a root container if possible).
    # Since ApexParser returns a list, let's create a dummy root and attach them
    import uuid

    from apex_rag.models.unified_models import ASTNode

    root = ASTNode(
        node_id=str(uuid.uuid4()),
        doc_id="dummy-doc",
        node_type=NodeType.PARAGRAPH,  # Proxy for document
        content="Root",
        children=nodes,
    )

    retriever = KeywordDeterministicRetriever()

    # Query matching the "Financials" section
    results = await retriever.retrieve("What was the Q3 revenue?", root, top_k=2)

    assert len(results) > 0
    top_node = results[0]

    # The paragraph should match best because it contains Q3 and revenue
    assert "Q3" in top_node.content
    assert top_node.node_type == NodeType.PARAGRAPH

    # Query matching the Engineering section heading
    results2 = await retriever.retrieve("engineering", root, top_k=2)
    assert len(results2) > 0

    # Because of the 5.0x heading boost, the Section node "Engineering" should score higher
    # than the paragraph "The engineering team..."
    assert results2[0].node_type == NodeType.HEADING
    assert "Engineering" in results2[0].content
