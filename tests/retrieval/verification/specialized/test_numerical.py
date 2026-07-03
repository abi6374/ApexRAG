from unittest.mock import AsyncMock

import pytest

from apex_rag.core.ast.models import ASTNode
from apex_rag.providers import AsyncLLM
from apex_rag.retrieval.verification.specialized.numerical import NumericalVerifier


@pytest.mark.asyncio
async def test_numerical_verifier_true():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = "TRUE"

    verifier = NumericalVerifier(llm=mock_llm)

    node = ASTNode(id="1", node_type="Paragraph", content="The revenue in Q3 was $50M.")
    result = await verifier.verify("What was Q3 revenue?", node)

    assert result is True
    call_args = mock_llm.generate.call_args[1]
    assert "specialized numerical verification engine" in call_args["prompt"]
    assert call_args["temperature"] == 0.0


@pytest.mark.asyncio
async def test_numerical_verifier_false():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = "FALSE"

    verifier = NumericalVerifier(llm=mock_llm)

    node = ASTNode(id="1", node_type="Paragraph", content="The company had a great quarter.")
    result = await verifier.verify("What was Q3 revenue?", node)

    assert result is False
