"""
ApexRAG — High-accuracy Structural AI Retrieval Infrastructure.

ApexRAG preserves document hierarchy using Abstract Syntax Trees (AST) instead
of naive chunking, enabling zero-hallucination agentic navigation.

Basic usage:
    >>> from apex_rag import ApexIndex, OpenAIProvider
    >>> async with await ApexIndex.create(model=OpenAIProvider()) as index:
    >>>     doc_id = await index.ingest("report.pdf")
    >>>     answer = await index.orchestrate_query("What is Q3 revenue?", doc_id)
"""
# ruff: noqa: E402
import logging
from importlib.metadata import PackageNotFoundError, version

# Initialize Logging
logger = logging.getLogger("apex_rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    __version__ = version("apex-rag")
except PackageNotFoundError:
    __version__ = "1.0.2"

# ── Primary Library Exports ───────────────────────────────────────────────

from apex_rag.client import ApexIndex
from apex_rag.core.ast.models import ASTNode, ASTNodeMetadata
from apex_rag.core.evidence.models import EvidencePacket
from apex_rag.enterprise.auth.models import TenantContext

# ── Error Hierarchy ───────────────────────────────────────────────────────
from apex_rag.exceptions import (
    ApexRAGError,
    AuthenticationError,
    ConfigurationError,
    DocumentNotFoundError,
    FileValidationError,
    StorageError,
)

# ── Integrations ──────────────────────────────────────────────────────────
from apex_rag.integrations.langchain import ApexRAGRetriever
from apex_rag.providers import (
    AnthropicProvider,
    GroqProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
)
from apex_rag.retrieval.agentic.navigator import ASTNavigationResult

# ── Vision / Multi-modal (Part 8) ────────────────────────────────────────
from apex_rag.retrieval.vision import ImageParser, VisionAdapter

__all__ = [
    "ApexIndex",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GroqProvider",
    "OllamaProvider",
    "ASTNode",
    "ASTNodeMetadata",
    "EvidencePacket",
    "VisionAdapter",
    "ImageParser",
    "TenantContext",
    "ASTNavigationResult",
    "ApexRAGRetriever",
    "ApexRAGError",
    "AuthenticationError",
    "ConfigurationError",
    "DocumentNotFoundError",
    "FileValidationError",
    "StorageError",
    "__version__",
]
