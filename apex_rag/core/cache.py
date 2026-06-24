"""
core/cache.py — Async-safe, tenant-aware, TTL-backed caching for ApexRAG.

Provides three cache types:
  - NavigationCache:  key=(query_hash, node_id) → LLM navigation decisions
  - VerificationCache: key=(query_hash, leaf_id) → verification outcomes
  - QueryCache:        key=(query_hash, document_id) → full query results

All caches support:
  - async-safe operations
  - TTL expiration
  - tenant awareness (tenant-prefixed keys)
  - configurable backend (in-memory default)

Usage:
    cache = QueryCache(ttl_seconds=300)
    await cache.set("query_hash", "doc-123", answer_data)
    cached = await cache.get("query_hash", "doc-123")
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Generic, TypeVar

logger = logging.getLogger("apex_rag.core.cache")

# ── Type Variables ─────────────────────────────────────────────────────────

T = TypeVar("T")


# ── Cache Entry ────────────────────────────────────────────────────────────


class CacheEntry(Generic[T]):
    """A single entry in the cache with TTL tracking."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: T, ttl_seconds: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


# ── Abstract Cache Backend ─────────────────────────────────────────────────


class CacheBackend(ABC, Generic[T]):
    """Abstract cache backend.  Implementations must be async-safe."""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Retrieve a value by key. Returns None if missing or expired."""
        ...

    @abstractmethod
    async def set(self, key: str, value: T, ttl_seconds: float) -> None:
        """Store a value with TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove a key. Returns True if it existed."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries."""
        ...

    @abstractmethod
    async def stats(self) -> dict[str, Any]:
        """Return cache statistics (size, hits, misses)."""
        ...


# ── In-Memory Backend (TTL-aware LRU) ─────────────────────────────────────


class InMemoryCacheBackend(CacheBackend[T]):
    """Thread-safe, TTL-aware, in-memory cache with LRU eviction.

    Uses an OrderedDict for O(1) get/set operations and LRU eviction
    when the max size is reached.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> T | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            self._store.pop(key, None)
            self._misses += 1
            return None
        # LRU: move to end
        self._store.move_to_end(key)
        self._hits += 1
        return entry.value

    async def set(self, key: str, value: T, ttl_seconds: float) -> None:
        # Evict LRU if at capacity
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = CacheEntry(value, ttl_seconds)
        self._store.move_to_end(key)

    async def delete(self, key: str) -> bool:
        if key in self._store:
            self._store.pop(key)
            return True
        return False

    async def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    async def size(self) -> int:
        # Prune expired entries first
        time.monotonic()
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            self._store.pop(k, None)
        return len(self._store)

    async def stats(self) -> dict[str, Any]:
        size = await self.size()
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }


# ── Tenanted Cache Wrapper ─────────────────────────────────────────────────


def _tenant_key(tenant_id: str, *parts: str) -> str:
    """Build a tenant-prefixed cache key.

    Format: ``<tenant_id>:<part1>:<part2>:...``
    """
    return f"{tenant_id}:{':'.join(parts)}"


def _compute_hash(text: str) -> str:
    """Compute a deterministic hash for query strings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── NavigationCache ────────────────────────────────────────────────────────


class NavigationCache:
    """Caches LLM navigation decisions: (query_hash, node_id) → chosen child.

    Reduces LLM calls by reusing previous navigation choices for the same
    query+node combination.
    """

    def __init__(
        self,
        backend: CacheBackend[dict[str, Any]] | None = None,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._backend = backend or InMemoryCacheBackend[dict[str, Any]](max_size=10_000)
        self._default_ttl = ttl_seconds

    async def get(
        self,
        query: str,
        node_id: str,
        tenant_id: str = "default",
    ) -> dict[str, Any] | None:
        """Get cached navigation decision for a query+node combination."""
        key = _tenant_key(tenant_id, _compute_hash(query), node_id, "nav")
        return await self._backend.get(key)

    async def set(
        self,
        query: str,
        node_id: str,
        decision: dict[str, Any],
        tenant_id: str = "default",
        ttl_seconds: float | None = None,
    ) -> None:
        """Cache a navigation decision."""
        key = _tenant_key(tenant_id, _compute_hash(query), node_id, "nav")
        await self._backend.set(key, decision, ttl_seconds or self._default_ttl)

    async def clear(self) -> None:
        await self._backend.clear()

    async def stats(self) -> dict[str, Any]:
        return await self._backend.stats()


# ── VerificationCache ──────────────────────────────────────────────────────


class VerificationCache:
    """Caches leaf verification outcomes: (query_hash, leaf_id) → verified.

    Reduces LLM verification calls by reusing previous verification results
    for the same query+leaf combination.
    """

    def __init__(
        self,
        backend: CacheBackend[bool] | None = None,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._backend = backend or InMemoryCacheBackend[bool](max_size=20_000)
        self._default_ttl = ttl_seconds

    async def get(
        self,
        query: str,
        leaf_id: str,
        tenant_id: str = "default",
    ) -> bool | None:
        """Get cached verification result for a query+leaf combination."""
        key = _tenant_key(tenant_id, _compute_hash(query), leaf_id, "verify")
        return await self._backend.get(key)

    async def set(
        self,
        query: str,
        leaf_id: str,
        verified: bool,
        tenant_id: str = "default",
        ttl_seconds: float | None = None,
    ) -> None:
        """Cache a verification result."""
        key = _tenant_key(tenant_id, _compute_hash(query), leaf_id, "verify")
        await self._backend.set(key, verified, ttl_seconds or self._default_ttl)

    async def clear(self) -> None:
        await self._backend.clear()

    async def stats(self) -> dict[str, Any]:
        return await self._backend.stats()


# ── QueryCache ─────────────────────────────────────────────────────────────


class QueryCache:
    """Caches full query results: (query_hash, document_id) → answer.

    Provides instant replay of identical queries without re-running
    the entire retrieval pipeline.
    """

    def __init__(
        self,
        backend: CacheBackend[dict[str, Any]] | None = None,
        ttl_seconds: float = 600.0,
    ) -> None:
        self._backend = backend or InMemoryCacheBackend[dict[str, Any]](max_size=5_000)
        self._default_ttl = ttl_seconds

    async def get(
        self,
        query: str,
        document_id: str,
        tenant_id: str = "default",
    ) -> dict[str, Any] | None:
        """Get cached query result for a query+document combination."""
        key = _tenant_key(tenant_id, _compute_hash(query), document_id, "query")
        return await self._backend.get(key)

    async def set(
        self,
        query: str,
        document_id: str,
        result: dict[str, Any],
        tenant_id: str = "default",
        ttl_seconds: float | None = None,
    ) -> None:
        """Cache a query result."""
        key = _tenant_key(tenant_id, _compute_hash(query), document_id, "query")
        await self._backend.set(key, result, ttl_seconds or self._default_ttl)

    async def clear(self) -> None:
        await self._backend.clear()

    async def stats(self) -> dict[str, Any]:
        return await self._backend.stats()
