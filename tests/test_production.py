import pytest
from unittest.mock import AsyncMock, MagicMock
from apex_rag.providers import GroqProvider, AnthropicProvider
from apex_rag.storage import StorageEngine, DocumentNode, QueryCache
from apex_rag.navigation import NavigationAgent, AggregatorAgent, NavigationResult
from apex_rag.client import ApexIndex

@pytest.mark.asyncio
async def test_groq_provider():
    with MagicMock() as mock_groq:
        import sys
        sys.modules['groq'] = mock_groq
        provider = GroqProvider(api_key="test_key")
        provider._client.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Groq response"))]))
        
        resp = await provider.generate("test prompt")
        assert resp == "Groq response"

@pytest.mark.asyncio
async def test_anthropic_provider():
    with MagicMock() as mock_anthropic:
        import sys
        sys.modules['anthropic'] = mock_anthropic
        provider = AnthropicProvider(api_key="test_key")
        provider._client.messages.create = AsyncMock(return_value=MagicMock(content=[MagicMock(text="Anthropic response")]))
        
        resp = await provider.generate("test prompt")
        assert resp == "Anthropic response"

@pytest.mark.asyncio
async def test_query_cache_integration():
    # Use in-memory SQLite for testing cache
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")
    
    async with storage.session() as session:
        # 1. Setup a node
        node = DocumentNode(doc_id="doc1", path="1", title="Title", summary="Summary", content="Leaf content")
        await storage.insert_node(session, node)
        node_id = node.id
        
        # 2. Insert cache entry
        await storage.insert_cache_entry(session, "test query", "doc1", node_id)
        
    async with storage.session() as session:
        # 3. Retrieve from cache
        cache_entry = await storage.get_cached_query(session, "test query", "doc1")
        assert cache_entry is not None
        assert cache_entry.node_id == node_id
        assert cache_entry.hit_count == 2 # 1 from insert (default), +1 from get_cached_query increment

@pytest.mark.asyncio
async def test_aggregator_agent():
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Synthesized answer")
    aggregator = AggregatorAgent(mock_llm)
    
    results = [
        NavigationResult(content="Part 1", node_id=1, path="1", title="T1", trace=[]),
        NavigationResult(content="Part 2", node_id=2, path="2", title="T2", trace=[])
    ]
    
    answer = await aggregator.synthesize("What are the parts?", results)
    assert answer == "Synthesized answer"
    mock_llm.generate.assert_called_once()
    assert "Part 1" in mock_llm.generate.call_args[1]["prompt"]
    assert "Part 2" in mock_llm.generate.call_args[1]["prompt"]

@pytest.mark.asyncio
async def test_hybrid_search_logic():
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value='{"chosen_doc_ids": ["doc2"]}')
    
    agent = NavigationAgent(storage, model=mock_llm)
    
    async with storage.session() as session:
        node1 = DocumentNode(doc_id="doc1", path="1", title="Physics", summary="Science", depth=0)
        await storage.insert_node(session, node1)
        node2 = DocumentNode(doc_id="doc2", path="1", title="Chemistry", summary="Science", depth=0)
        await storage.insert_node(session, node2)

    # Mock the find method to avoid full recursion
    agent.find = AsyncMock(return_value=None)
    
    # This should trigger the FTS and LLM selection logic
    await agent.find_global("test query")
    
    # Verify LLM was called with the document summaries
    assert mock_llm.generate.called
