from unittest.mock import AsyncMock

import pytest

from apex_rag.agents.critic.agent import EvaluationCriticAgent
from apex_rag.core.ast.models import ASTNode
from apex_rag.providers import AsyncLLM


@pytest.mark.asyncio
async def test_critic_agent_pass():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = '''
    {
      "passes_evaluation": true,
      "reason": "Both Q2 and Q3 revenues are stated."
    }
    '''

    critic = EvaluationCriticAgent(llm=mock_llm)
    nodes = [
        ASTNode(id="1", node_type="Paragraph", content="Q2 revenue was $40M."),
        ASTNode(id="2", node_type="Paragraph", content="Q3 revenue was $50M.")
    ]

    result = await critic.evaluate(["What is Q2 revenue?", "What is Q3 revenue?"], nodes)
    assert result is True

@pytest.mark.asyncio
async def test_critic_agent_fail():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = '''
    {
      "passes_evaluation": false,
      "reason": "Q3 revenue is missing."
    }
    '''

    critic = EvaluationCriticAgent(llm=mock_llm)
    nodes = [
        ASTNode(id="1", node_type="Paragraph", content="Q2 revenue was $40M.")
    ]

    result = await critic.evaluate(["What is Q2 revenue?", "What is Q3 revenue?"], nodes)
    assert result is False
