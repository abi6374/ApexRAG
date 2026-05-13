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

from apex_rag.client import ApexIndex
from apex_rag.ingestion import IngestionEngine, ParsedSection, Summariser
from apex_rag.navigation import NavigationAgent, NavigationResult
from apex_rag.providers import AsyncLLM, OllamaProvider, OpenAIProvider
from apex_rag.storage import DocumentNode, PageIndexEntry, StorageEngine
from apex_rag.utils import ReasoningTrace, logger, set_log_level

__all__ = [
    # Primary API
    "ApexIndex",
    # Navigation
    "NavigationAgent",
    "NavigationResult",
    # Ingestion
    "IngestionEngine",
    "Summariser",
    "ParsedSection",
    # Providers
    "AsyncLLM",
    "OllamaProvider",
    "OpenAIProvider",
    # Storage
    "StorageEngine",
    "DocumentNode",
    "PageIndexEntry",
    # Observability
    "ReasoningTrace",
    "logger",
    "set_log_level",
    "__version__",
]
