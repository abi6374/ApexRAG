"""
tests/test_caches.py — Tests for the caching infrastructure (Phase 8).
"""

from __future__ import annotations

import pytest

from apex_rag.core.cache import (
    InMemoryCacheBackend,
    NavigationCache,
    QueryCache,
    VerificationCache,
)


class TestInMemoryCacheBackend:
    """Tests for the in-memory cache backend."""

    @pytest.fixture
    def cache(self) -> InMemoryCacheBackend[str]:
        return InMemoryCacheBackend[str](max_size=100)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache: InMemoryCacheBackend[str]) -> None:
        await cache.set("key1", "value1", ttl_seconds=60)
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing(self, cache: InMemoryCacheBackend[str]) -> None:
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_expired(self, cache: InMemoryCacheBackend[str]) -> None:
        await cache.set("key1", "value1", ttl_seconds=0)  # Already expired
        import time

        time.sleep(0.01)  # Tiny wait for expiry
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing(self, cache: InMemoryCacheBackend[str]) -> None:
        await cache.set("key1", "value1", ttl_seconds=60)
        deleted = await cache.delete("key1")
        assert deleted is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_missing(self, cache: InMemoryCacheBackend[str]) -> None:
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear(self, cache: InMemoryCacheBackend[str]) -> None:
        await cache.set("key1", "value1", ttl_seconds=60)
        await cache.set("key2", "value2", ttl_seconds=60)
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_stats(self, cache: InMemoryCacheBackend[str]) -> None:
        await cache.set("key1", "value1", ttl_seconds=60)
        await cache.get("key1")  # hit
        await cache.get("nonexistent")  # miss

        stats = await cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Should evict oldest entries when max_size is reached."""
        cache = InMemoryCacheBackend[str](max_size=3)
        await cache.set("key1", "v1", ttl_seconds=60)
        await cache.set("key2", "v2", ttl_seconds=60)
        await cache.set("key3", "v3", ttl_seconds=60)
        await cache.set("key4", "v4", ttl_seconds=60)  # Should evict key1

        assert await cache.get("key1") is None  # Evicted
        assert await cache.get("key2") == "v2"
        assert await cache.get("key3") == "v3"
        assert await cache.get("key4") == "v4"


class TestNavigationCache:
    """Tests for the NavigationCache."""

    @pytest.fixture
    def nav_cache(self) -> NavigationCache:
        return NavigationCache(ttl_seconds=300)

    @pytest.mark.asyncio
    async def test_set_and_get(self, nav_cache: NavigationCache) -> None:
        decision = {"chosen_id": "node-123", "reason": "best match"}
        await nav_cache.set("What is revenue?", "node-abc", decision)
        cached = await nav_cache.get("What is revenue?", "node-abc")
        assert cached == decision

    @pytest.mark.asyncio
    async def test_get_missing(self, nav_cache: NavigationCache) -> None:
        result = await nav_cache.get("What is revenue?", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, nav_cache: NavigationCache) -> None:
        decision = {"chosen_id": "node-123"}
        await nav_cache.set("query", "node-abc", decision, tenant_id="tenant-a")
        cached_a = await nav_cache.get("query", "node-abc", tenant_id="tenant-a")
        cached_b = await nav_cache.get("query", "node-abc", tenant_id="tenant-b")
        assert cached_a == decision
        assert cached_b is None  # Different tenant -> miss

    @pytest.mark.asyncio
    async def test_clear(self, nav_cache: NavigationCache) -> None:
        await nav_cache.set("query1", "node-1", {"chosen_id": "n1"})
        await nav_cache.set("query2", "node-2", {"chosen_id": "n2"})
        await nav_cache.clear()
        assert await nav_cache.get("query1", "node-1") is None


class TestVerificationCache:
    """Tests for the VerificationCache."""

    @pytest.fixture
    def ver_cache(self) -> VerificationCache:
        return VerificationCache(ttl_seconds=300)

    @pytest.mark.asyncio
    async def test_set_and_get(self, ver_cache: VerificationCache) -> None:
        await ver_cache.set("What is revenue?", "leaf-1", True)
        cached = await ver_cache.get("What is revenue?", "leaf-1")
        assert cached is True

    @pytest.mark.asyncio
    async def test_false_value(self, ver_cache: VerificationCache) -> None:
        await ver_cache.set("What is invalid?", "leaf-2", False)
        cached = await ver_cache.get("What is invalid?", "leaf-2")
        assert cached is False

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, ver_cache: VerificationCache) -> None:
        await ver_cache.set("q", "leaf-1", True, tenant_id="tenant-a")
        cached_a = await ver_cache.get("q", "leaf-1", tenant_id="tenant-a")
        cached_b = await ver_cache.get("q", "leaf-1", tenant_id="tenant-b")
        assert cached_a is True
        assert cached_b is None


class TestQueryCache:
    """Tests for the QueryCache."""

    @pytest.fixture
    def query_cache(self) -> QueryCache:
        return QueryCache(ttl_seconds=600)

    @pytest.mark.asyncio
    async def test_set_and_get(self, query_cache: QueryCache) -> None:
        result = {"answer": "Revenue is $120k", "confidence": 0.95}
        await query_cache.set("What is revenue?", "doc-123", result)
        cached = await query_cache.get("What is revenue?", "doc-123")
        assert cached == result

    @pytest.mark.asyncio
    async def test_different_document(self, query_cache: QueryCache) -> None:
        result = {"answer": "Revenue is $120k"}
        await query_cache.set("What is revenue?", "doc-123", result)
        cached = await query_cache.get("What is revenue?", "doc-456")
        assert cached is None  # Different doc -> miss
