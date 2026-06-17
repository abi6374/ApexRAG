import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from apex_rag.providers import AsyncLLM
from apex_rag.retrieval.agentic.navigator import ASTNavigationAgent
from apex_rag.retrieval.deterministic.keyword import KeywordDeterministicRetriever
from apex_rag.retrieval.verification.strict_verifier import StrictLeafVerifier
from apex_rag.ingestion.apex_storage import ApexStorage, ASTNodeRow


@pytest.mark.asyncio
async def test_ast_navigation_agent_flow():
    doc_id = "doc1"
    root_id = str(uuid.uuid4())
    leaf_id = str(uuid.uuid4())

    mock_storage = MagicMock(spec=ApexStorage)
    mock_storage.session.return_value.__aenter__.return_value = AsyncMock()

    # Mock DB rows returned by get_ast_children
    root_row = ASTNodeRow(
        node_id=root_id, doc_id=doc_id, node_type="HEADING", content="Root", parent_id=None
    )
    leaf_row = ASTNodeRow(
        node_id=leaf_id, doc_id=doc_id, node_type="PARAGRAPH", content="Financial data for Q3\n\nThe Q3 revenue is $50M.", parent_id=root_id
    )

    async def mock_get_ast_children(session, parent_id, doc_id):
        if parent_id is None:
            return [root_row]
        if parent_id == root_id:
            return [leaf_row]
        return []

    mock_storage.get_ast_children = AsyncMock(side_effect=mock_get_ast_children)

    # Mock LLM for Navigation
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = f'{{"chosen_id": "{leaf_id}", "fallback_id": null}}'

    # Mock LLM for Verifier
    mock_verifier_llm = AsyncMock(spec=AsyncLLM)
    mock_verifier_llm.generate.return_value = "TRUE"

    retriever = KeywordDeterministicRetriever()
    verifier = StrictLeafVerifier(llm=mock_verifier_llm)

    agent = ASTNavigationAgent(
        storage=mock_storage,
        model=mock_llm,
        retriever=retriever,
        verifier=verifier
    )

    result = await agent.find("What was Q3 revenue?", doc_id)

    assert result is not None
    assert result.verified is True
    assert result.node_id == leaf_id
    assert "Q3 revenue" in result.content
