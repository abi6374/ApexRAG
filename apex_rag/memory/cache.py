

class ReasoningMemoryCache:
    """An in-memory cache for storing and retrieving structural reasoning paths."""
    def __init__(self) -> None:
        self._cache: dict[str, list[str]] = {}

    async def store_path(self, query: str, path: list[str]) -> None:
        """Stores a structural path for a given query."""
        self._cache[query] = path

    async def get_path(self, query: str) -> list[str] | None:
        """Retrieves a structural path for a given query if it exists."""
        return self._cache.get(query)
