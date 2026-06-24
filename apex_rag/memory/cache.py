from typing import Any

from apex_rag.models.unified_models import CausalEdge, EvidencePacket


class TraversalCache:
    """Cache for successful AST navigation routes, mapped by normalized query terms."""

    def __init__(self) -> None:
        self._cache: dict[str, list[str]] = {}

    def get(self, query: str) -> list[str] | None:
        return self._cache.get(query.lower().strip())

    def set(self, query: str, path: list[str]) -> None:
        self._cache[query.lower().strip()] = path


class ReasoningCache:
    """Cache for planner effectiveness plans, mapped by normalized queries."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def get(self, query: str) -> dict[str, Any] | None:
        return self._cache.get(query.lower().strip())

    def set(self, query: str, plan_data: dict[str, Any]) -> None:
        self._cache[query.lower().strip()] = plan_data


class EvidenceCache:
    """Cache for verified EvidencePackets retrieved for specific queries."""

    def __init__(self) -> None:
        self._cache: dict[str, list[EvidencePacket]] = {}

    def get(self, query: str) -> list[EvidencePacket] | None:
        return self._cache.get(query.lower().strip())

    def set(self, query: str, packets: list[EvidencePacket]) -> None:
        self._cache[query.lower().strip()] = packets


class VerificationCache:
    """Cache for leaf node verification results (node_id + query -> verification status)."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def _get_key(self, query: str, node_id: str) -> str:
        return f"{node_id}:{query.lower().strip()}"

    def get(self, query: str, node_id: str) -> dict[str, Any] | None:
        return self._cache.get(self._get_key(query, node_id))

    def set(self, query: str, node_id: str, result: dict[str, Any]) -> None:
        self._cache[self._get_key(query, node_id)] = result


class GraphPathCache:
    """Cache for evaluated Causal Reasoning Paths between nodes."""

    def __init__(self) -> None:
        self._cache: dict[str, list[CausalEdge]] = {}

    def _get_key(self, source_id: str, target_id: str) -> str:
        return f"{source_id}->{target_id}"

    def get(self, source_id: str, target_id: str) -> list[CausalEdge] | None:
        return self._cache.get(self._get_key(source_id, target_id))

    def set(self, source_id: str, target_id: str, path: list[CausalEdge]) -> None:
        self._cache[self._get_key(source_id, target_id)] = path


class ReasoningMemoryCache:
    """Compatibility cache layer mapping to TraversalCache."""

    def __init__(self) -> None:
        self._cache = TraversalCache()

    async def get_path(self, query: str) -> list[str] | None:
        return self._cache.get(query)

    async def store_path(self, query: str, path: list[str]) -> None:
        self._cache.set(query, path)
