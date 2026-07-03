from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apex_rag.navigation import AggregatorAgent, NavigationAgent, NavigationResult
from apex_rag.storage import DocumentNode, StorageEngine


@pytest.mark.asyncio
async def test_groq_provider():
    """Test GroqProvider with proper mocking (injects mock groq module)."""
    mock_groq_module = MagicMock()
    mock_async_client = MagicMock()
    mock_groq_module.AsyncGroq.return_value = mock_async_client
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Groq response"))])
    )

    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        # Re-import to pick up the mock module
        import importlib

        import apex_rag.providers as providers_mod

        importlib.reload(providers_mod)
        from apex_rag.providers import GroqProvider as GP

        provider = GP(api_key="test_key")
        resp = await provider.generate("test prompt")
        assert resp == "Groq response"


@pytest.mark.asyncio
async def test_anthropic_provider():
    """Test AnthropicProvider with proper mocking (injects mock anthropic module)."""
    mock_anthropic_module = MagicMock()
    mock_async_client = MagicMock()
    mock_anthropic_module.AsyncAnthropic.return_value = mock_async_client
    mock_async_client.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text="Anthropic response")])
    )

    with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        import importlib

        import apex_rag.providers as providers_mod

        importlib.reload(providers_mod)
        from apex_rag.providers import AnthropicProvider as AP

        provider = AP(api_key="test_key")
        resp = await provider.generate("test prompt")
        assert resp == "Anthropic response"


@pytest.mark.asyncio
async def test_query_cache_integration():
    """Test query cache: insert, retrieve, and verify hit_count increment."""
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")

    async with storage.session() as session:
        # 1. Setup a node
        node = DocumentNode(
            doc_id="doc1",
            path="1",
            title="Title",
            summary="Summary",
            content="Leaf content",
        )
        await storage.insert_node(session, node)
        node_id = node.id

        # 2. Insert cache entry
        await storage.insert_cache_entry(session, "test query", "doc1", node_id)

    async with storage.session() as session:
        # 3. Retrieve from cache (exact match)
        cache_entry = await storage.get_cached_query(session, "test query", "doc1")
        assert cache_entry is not None
        assert cache_entry.node_id == node_id
        # Default hit_count=1, incremented by get_cached_query → should be 2
        assert cache_entry.hit_count == 2

    async with storage.session() as session:
        # 4. Test substring matching (e.g., similar query)
        cache_entry = await storage.get_cached_query(session, "what is test query about", "doc1")
        assert cache_entry is not None
        assert cache_entry.node_id == node_id


@pytest.mark.asyncio
async def test_aggregator_agent():
    """Test AggregatorAgent synthesizes multiple results."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Synthesized answer")
    aggregator = AggregatorAgent(mock_llm)

    results = [
        NavigationResult(content="Part 1", node_id=1, path="1", title="T1", trace=[]),
        NavigationResult(content="Part 2", node_id=2, path="2", title="T2", trace=[]),
    ]

    answer = await aggregator.synthesize("What are the parts?", results)
    assert answer == "Synthesized answer"
    mock_llm.generate.assert_called_once()
    assert "Part 1" in mock_llm.generate.call_args[1]["prompt"]
    assert "Part 2" in mock_llm.generate.call_args[1]["prompt"]


@pytest.mark.asyncio
async def test_hybrid_search_logic():
    """Test the find_global hybrid search (FTS + agentic doc selection)."""
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")

    # Mock the LLM to return doc2 as the chosen document
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value='{"chosen_doc_ids": ["doc2"], "reason": "Chemistry matches"}'
    )

    agent = NavigationAgent(storage, model=mock_llm)

    async with storage.session() as session:
        node1 = DocumentNode(
            doc_id="doc1",
            path="1",
            title="Physics",
            summary="Science",
            depth=0,
        )
        await storage.insert_node(session, node1)
        node2 = DocumentNode(
            doc_id="doc2",
            path="1",
            title="Chemistry",
            summary="Science",
            depth=0,
        )
        await storage.insert_node(session, node2)

    # Mock the find method to avoid full recursion
    agent.find = AsyncMock(return_value=None)

    # This should trigger the FTS and LLM selection logic
    await agent.find_global("test query")

    # Verify LLM was called with the document summaries
    assert mock_llm.generate.called
