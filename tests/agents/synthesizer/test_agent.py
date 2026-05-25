from unittest.mock import AsyncMock

import pytest

from apex_rag.agents.synthesizer.agent import EvidenceSynthesizerAgent
from apex_rag.core.evidence.models import EvidencePacket
from apex_rag.providers import AsyncLLM


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=AsyncLLM)
    llm.generate.return_value = "This is a mocked synthesized answer based on [Node ID: node-1]."
    return llm

@pytest.mark.asyncio
async def test_synthesize_with_evidence(mock_llm):
    agent = EvidenceSynthesizerAgent(llm=mock_llm)

    packet = EvidencePacket(
        node_id="node-1",
        source_document="doc1.txt",
        section_path="Section 1",
        page_number=1,
        paragraph_index=1,
        retrieval_reason="relevant",
        verification_result=True,
        confidence_score=0.9,
        content="This is the content of the document."
    )

    query = "What is the content?"
    response = await agent.synthesize(query, [packet])

    assert response == "This is a mocked synthesized answer based on [Node ID: node-1]."

    mock_llm.generate.assert_called_once()
    called_prompt = mock_llm.generate.call_args[0][0]

    assert "EVIDENCE:" in called_prompt
    assert "[Node ID: node-1]" in called_prompt
    assert "This is the content of the document." in called_prompt
    assert "USER QUERY:\nWhat is the content?" in called_prompt
    assert "ONLY using the provided evidence" in called_prompt
    assert "cite the source" in called_prompt

@pytest.mark.asyncio
async def test_synthesize_empty_packets(mock_llm):
    agent = EvidenceSynthesizerAgent(llm=mock_llm)
    response = await agent.synthesize("test query", [])

    assert "could not find enough evidence" in response
    mock_llm.generate.assert_not_called()

@pytest.mark.asyncio
async def test_synthesize_unverified_packets(mock_llm):
    agent = EvidenceSynthesizerAgent(llm=mock_llm)

    packet = EvidencePacket(
        node_id="node-1",
        source_document="doc1.txt",
        section_path="Section 1",
        retrieval_reason="relevant",
        verification_result=False, # False!
        confidence_score=0.9,
        content="This is the content of the document."
    )

    response = await agent.synthesize("test query", [packet])

    assert "No verified evidence" in response
    mock_llm.generate.assert_not_called()
