"""
ApexRAG — Production-grade, local-first Agentic RAG Library.

Install::

    pip install apex-rag

Quick start::

    import asyncio
    from apex_rag import ApexIndex

    async def main():
        async with await ApexIndex.create() as index:
            doc_id = await index.ingest("report.pdf")
            result = await index.query("What is the Q3 revenue?", doc_id)
            if result:
                print(result.content)
                print(result.path, result.verified, result.confidence)

    asyncio.run(main())
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("apex-rag")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

# ── Public API ────────────────────────────────────────────────────────────
# These are the classes and functions users interact with directly.
# Internal implementation details (StorageEngine, DocumentNode, etc.) are
# available from their respective submodules but not re-exported here.

from apex_rag.client import ApexIndex
from apex_rag.config import settings

# ── Error hierarchy ───────────────────────────────────────────────────────
from apex_rag.exceptions import (  # noqa: F401
    ApexRAGError,
    AuthenticationError,
    ConfigurationError,
    DatabaseConnectionError,
    DocumentExistsError,
    DocumentNotFoundError,
    FileValidationError,
    IngestionError,
    InvalidProviderError,
    ProviderError,
    QueryError,
    RateLimitError,
    StorageError,
)

# ── Internal Re-exports (available but not in __all__) ────────────────────
# These can be imported directly:  from apex_rag import NavigationAgent
# But they won't appear in `from apex_rag import *` — keeping the
# user-facing surface area small and discoverable.
from apex_rag.ingestion.legacy import IngestionEngine, ParsedSection, Summariser  # noqa: F401
from apex_rag.navigation import AggregatorAgent, NavigationAgent, NavigationResult  # noqa: F401
from apex_rag.providers import (
    AnthropicProvider,
    AsyncLLM,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
)

# ── Hybrid Search Engine ──────────────────────────────────────────────────
from apex_rag.search import EmbeddingsEngine, HybridSearch  # noqa: F401
from apex_rag.storage import DocumentNode, PageIndexEntry, QueryCache, StorageEngine  # noqa: F401

# ── Telemetry ─────────────────────────────────────────────────────────────
from apex_rag.telemetry import QueryMetricsCollector, query_metrics, setup_telemetry  # noqa: F401
from apex_rag.utils import (
    ReasoningTrace,  # noqa: F401
    logger,
    set_log_level,
)

__all__ = [
    # Configuration
    "settings",
    # Primary API — the one class users need
    "ApexIndex",
    # New Multi-Agent / AST Architecture
    "ASTNode",
    "ASTNodeMetadata",
    "SemanticModel",
    "ASTNavigationAgent",
    "ASTNavigationResult",
    "Orchestrator",
    "QueryPlannerAgent",
    "EvaluationCriticAgent",
    "TenantContext",
    # Query result type
    "NavigationResult",
    # Provider protocol & implementations
    "AsyncLLM",
    "OllamaProvider",
    "OpenAIProvider",
    "GroqProvider",
    "AnthropicProvider",
    # Error hierarchy
    "ApexRAGError",
    "ConfigurationError",
    "InvalidProviderError",
    "DocumentNotFoundError",
    "DocumentExistsError",
    "IngestionError",
    "QueryError",
    "ProviderError",
    "StorageError",
    "DatabaseConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "FileValidationError",
    # Observability
    "logger",
    "set_log_level",
    "query_metrics",
    "QueryMetricsCollector",
    "setup_telemetry",
    # Hybrid Search
    "HybridSearch",
    "EmbeddingsEngine",
    # Package version
    "__version__",
]
