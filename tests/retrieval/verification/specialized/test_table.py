from unittest.mock import AsyncMock

import pytest

from apex_rag.core.ast.models import ASTNode
from apex_rag.providers import AsyncLLM
from apex_rag.retrieval.verification.specialized.table import TableVerifier


@pytest.mark.asyncio
async def test_table_verifier_true():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = "TRUE"

    verifier = TableVerifier(llm=mock_llm)

    node = ASTNode(
        id="1", node_type="Table", content="| Quarter | Revenue |\n|---|---|\n| Q3 | $50M |"
    )
    result = await verifier.verify("What was Q3 revenue?", node)

    assert result is True
    call_args = mock_llm.generate.call_args[1]
    assert "specialized table verification engine" in call_args["prompt"]
    assert call_args["temperature"] == 0.0


@pytest.mark.asyncio
async def test_table_verifier_false():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = "FALSE"

    verifier = TableVerifier(llm=mock_llm)

    node = ASTNode(
        id="1", node_type="Table", content="| Employee | Role |\n|---|---|\n| Alice | Engineer |"
    )
    result = await verifier.verify("What was Q3 revenue?", node)

    assert result is False


@pytest.mark.asyncio
async def test_table_verifier_not_a_table():
    mock_llm = AsyncMock(spec=AsyncLLM)
    # The verifier should return False immediately without calling the LLM
    verifier = TableVerifier(llm=mock_llm)

    node = ASTNode(id="1", node_type="Paragraph", content="The revenue in Q3 was $50M.")
    result = await verifier.verify("What was Q3 revenue?", node)

    assert result is False
    mock_llm.generate.assert_not_called()
