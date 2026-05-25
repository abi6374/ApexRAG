import uuid
from unittest.mock import AsyncMock

import pytest

from apex_rag.providers import AsyncLLM
from apex_rag.retrieval.agentic.navigator import ASTNavigationAgent
from apex_rag.retrieval.deterministic.keyword import KeywordDeterministicRetriever
from apex_rag.retrieval.verification.strict_verifier import StrictLeafVerifier
from apex_rag.storage import NodeData, SemanticModelData, StorageEngine


@pytest.mark.asyncio
async def test_ast_navigation_agent_flow():
    # In-memory DB
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")

    doc_id = "doc1"
    root_id = str(uuid.uuid4())
    leaf_id = str(uuid.uuid4())

    async with storage.session() as session:
        # Create Root Node
        await storage.insert_ast_node(session, NodeData(
            id=root_id, tenant_id="default", doc_id=doc_id, node_type="Section", content="Root"
        ))

        # Create Leaf Node
        await storage.insert_ast_node(session, NodeData(
            id=leaf_id, tenant_id="default", doc_id=doc_id, parent_id=root_id, node_type="Paragraph", content="The Q3 revenue is $50M."
        ))

        # Create Semantic Map for Leaf
        session.add(SemanticModelData(
            node_id=leaf_id, concise_summary="Financial data for Q3"
        ))
        await session.commit()

    # Mock LLM for Navigation
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = f'{{"chosen_id": "{leaf_id}", "fallback_id": null}}'

    # Mock LLM for Verifier
    mock_verifier_llm = AsyncMock(spec=AsyncLLM)
    mock_verifier_llm.generate.return_value = "TRUE"

    retriever = KeywordDeterministicRetriever()
    verifier = StrictLeafVerifier(llm=mock_verifier_llm)

    agent = ASTNavigationAgent(
        storage=storage,
        model=mock_llm,
        retriever=retriever,
        verifier=verifier
    )

    result = await agent.find("What was Q3 revenue?", doc_id)

    assert result is not None
    assert result.verified is True
    assert result.node_id == leaf_id
    assert "Q3 revenue" in result.content
