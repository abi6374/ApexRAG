import pytest

from apex_rag.memory.cache import ReasoningMemoryCache


@pytest.mark.asyncio
async def test_store_and_get_path():
    cache = ReasoningMemoryCache()

    query = "What is the capital of France?"
    path = ["Europe", "France", "Paris"]

    # Initially should be None
    assert await cache.get_path(query) is None

    # Store the path
    await cache.store_path(query, path)

    # Retrieve the path
    retrieved_path = await cache.get_path(query)
    assert retrieved_path == path


@pytest.mark.asyncio
async def test_overwrite_path():
    cache = ReasoningMemoryCache()

    query = "Test query"
    path1 = ["Node1", "Node2"]
    path2 = ["Node1", "Node3"]

    await cache.store_path(query, path1)
    assert await cache.get_path(query) == path1

    await cache.store_path(query, path2)
    assert await cache.get_path(query) == path2
