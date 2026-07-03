from unittest.mock import AsyncMock

import pytest

from apex_rag.agents.planner.agent import QueryPlannerAgent
from apex_rag.providers import AsyncLLM


@pytest.mark.asyncio
async def test_query_planner_multi_hop():
    mock_llm = AsyncMock(spec=AsyncLLM)
    # Mocking a response for a comparison query
    mock_llm.generate.return_value = """
    {
      "sub_queries": [
        "What was the Q2 revenue?",
        "What was the Q3 revenue?"
      ]
    }
    """

    planner = QueryPlannerAgent(llm=mock_llm)
    plan = await planner.plan("Compare the revenue between Q2 and Q3.")

    assert len(plan) == 2
    assert "Q2 revenue" in plan[0]
    assert "Q3 revenue" in plan[1]


@pytest.mark.asyncio
async def test_query_planner_fallback():
    mock_llm = AsyncMock(spec=AsyncLLM)
    mock_llm.generate.return_value = "invalid json"

    planner = QueryPlannerAgent(llm=mock_llm)
    query = "What is the capital of France?"
    plan = await planner.plan(query)

    assert len(plan) == 1
    assert plan[0] == query
