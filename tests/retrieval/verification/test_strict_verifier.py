from unittest.mock import AsyncMock

import pytest

from apex_rag.core.ast.models import ASTNode
from apex_rag.providers import AsyncLLM
from apex_rag.retrieval.verification.strict_verifier import StrictLeafVerifier


@pytest.mark.asyncio
async def test_strict_leaf_verifier_true():
    # Mock LLM
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = "TRUE"

    verifier = StrictLeafVerifier(llm=mock_llm)

    node = ASTNode(id="1", node_type="Paragraph", content="The revenue in Q3 was $50M.")
    result = await verifier.verify("What was Q3 revenue?", node)

    assert result is True
    # Ensure system prompt was sent
    call_args = mock_llm.generate.call_args[1]
    assert "strict verification engine" in call_args['prompt']
    assert "temperature" in call_args
    assert call_args['temperature'] == 0.0

@pytest.mark.asyncio
async def test_strict_leaf_verifier_false():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = " FALSE \n"

    verifier = StrictLeafVerifier(llm=mock_llm)

    node = ASTNode(id="1", node_type="Paragraph", content="The CEO is John Doe.")
    result = await verifier.verify("What was Q3 revenue?", node)

    assert result is False
