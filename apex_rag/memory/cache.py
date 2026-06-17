from typing import Any, Dict, List, Optional
from apex_rag.models.unified_models import ASTNode, CausalEdge, EvidencePacket


class TraversalCache:
    """Cache for successful AST navigation routes, mapped by normalized query terms."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[str]] = {}

    def get(self, query: str) -> Optional[List[str]]:
        return self._cache.get(query.lower().strip())

    def set(self, query: str, path: List[str]) -> None:
        self._cache[query.lower().strip()] = path


class ReasoningCache:
    """Cache for planner effectiveness plans, mapped by normalized queries."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(query.lower().strip())

    def set(self, query: str, plan_data: Dict[str, Any]) -> None:
        self._cache[query.lower().strip()] = plan_data


class EvidenceCache:
    """Cache for verified EvidencePackets retrieved for specific queries."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[EvidencePacket]] = {}

    def get(self, query: str) -> Optional[List[EvidencePacket]]:
        return self._cache.get(query.lower().strip())

    def set(self, query: str, packets: List[EvidencePacket]) -> None:
        self._cache[query.lower().strip()] = packets


class VerificationCache:
    """Cache for leaf node verification results (node_id + query -> verification status)."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_key(self, query: str, node_id: str) -> str:
        return f"{node_id}:{query.lower().strip()}"

    def get(self, query: str, node_id: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(self._get_key(query, node_id))

    def set(self, query: str, node_id: str, result: Dict[str, Any]) -> None:
        self._cache[self._get_key(query, node_id)] = result


class GraphPathCache:
    """Cache for evaluated Causal Reasoning Paths between nodes."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[CausalEdge]] = {}

    def _get_key(self, source_id: str, target_id: str) -> str:
        return f"{source_id}->{target_id}"

    def get(self, source_id: str, target_id: str) -> Optional[List[CausalEdge]]:
        return self._cache.get(self._get_key(source_id, target_id))

    def set(self, source_id: str, target_id: str, path: List[CausalEdge]) -> None:
        self._cache[self._get_key(source_id, target_id)] = path


class ReasoningMemoryCache:
    """Compatibility cache layer mapping to TraversalCache."""

    def __init__(self) -> None:
        self._cache = TraversalCache()

    async def get_path(self, query: str) -> Optional[List[str]]:
        return self._cache.get(query)

    async def store_path(self, query: str, path: List[str]) -> None:
        self._cache.set(query, path)

