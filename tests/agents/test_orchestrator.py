from unittest.mock import AsyncMock, MagicMock

import pytest

from apex_rag.agents.orchestrator import Orchestrator
from apex_rag.retrieval.agentic.navigator import ASTNavigationResult


@pytest.mark.asyncio
async def test_orchestrator_success():
    # Mocks
    mock_planner = AsyncMock()
    mock_planner.plan.return_value = ["Q2?", "Q3?"]

    mock_navigator = AsyncMock()
    mock_navigator.find.side_effect = [
        ASTNavigationResult(content="Q2 is $40M", node_id="1", path="", title="", trace=[], verified=True, confidence=1.0),
        ASTNavigationResult(content="Q3 is $50M", node_id="2", path="", title="", trace=[], verified=True, confidence=1.0)
    ]

    mock_critic = AsyncMock()
    mock_critic.evaluate.return_value = True

    orchestrator = Orchestrator(
        planner=mock_planner,
        navigator=mock_navigator,
        critic=mock_critic,
        trace=MagicMock()
    )

    result = await orchestrator.execute_query("Compare Q2 and Q3", "doc1")

    assert result is not None
    assert "Q2 is $40M" in result
    assert "Q3 is $50M" in result

    mock_critic.evaluate.assert_called_once()
    assert len(mock_critic.evaluate.call_args[0][1]) == 2 # 2 nodes passed to critic
