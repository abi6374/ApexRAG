"""
ApexRAG — High-accuracy Structural AI Retrieval Infrastructure.

ApexRAG preserves document hierarchy using Abstract Syntax Trees (AST) instead
of naive chunking, enabling zero-hallucination agentic navigation.

Basic usage:
    >>> from apex_rag import ApexIndex
    >>> from apex_rag.providers import OpenAIProvider
    >>> async with await ApexIndex.create(model=OpenAIProvider()) as index:
    >>>     doc_id = await index.ingest("report.pdf")
    >>>     answer = await index.query("What is Q3 revenue?", doc_id)
"""

# ruff: noqa: E402
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

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
    __version__ = "1.0.5"

# ── Primary Library Exports ───────────────────────────────────────────────

from apex_rag.client import ApexIndex
from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.exceptions import (
    ApexRAGError,
    AuthenticationError,
    ConfigurationError,
    DocumentNotFoundError,
    FileValidationError,
    StorageError,
)
from apex_rag.models.unified_models import ApexAnswer, EvidencePacket
from apex_rag.providers import LLMProvider

__all__ = [
    "ApexIndex",
    "LLMProvider",
    "TenantContext",
    "ApexAnswer",
    "EvidencePacket",
    "ApexRAGError",
    "AuthenticationError",
    "ConfigurationError",
    "DocumentNotFoundError",
    "FileValidationError",
    "StorageError",
    "__version__",
]


# ── Deprecation shims ──────────────────────────────────────────────────────
# These provide clear guidance when users try to import symbols that have been
# moved to subpackages as part of the API stabilization (v1.0).


def __getattr__(name: str) -> Any:
    """Provide helpful error messages for removed/moved exports."""
    _moved: dict[str, str] = {
        "OpenAIProvider": "from apex_rag.providers import OpenAIProvider",
        "AnthropicProvider": "from apex_rag.providers import AnthropicProvider",
        "GroqProvider": "from apex_rag.providers import GroqProvider",
        "OllamaProvider": "from apex_rag.providers import OllamaProvider",
        "ASTNode": "from apex_rag.core.ast.models import ASTNode",
        "ASTNodeMetadata": "from apex_rag.core.ast.models import ASTNodeMetadata",
        "ASTNavigationResult": "from apex_rag.retrieval.agentic.navigator import ASTNavigationResult",
        "ApexRAGRetriever": "from apex_rag.integrations.langchain import ApexRAGRetriever",
        "VisionAdapter": "from apex_rag.retrieval.vision import VisionAdapter",
        "ImageParser": "from apex_rag.retrieval.vision import ImageParser",
        "EnterpriseClient": "Use index.enterprise (ApexIndex.enterprise property)",
    }
    if name in _moved:
        raise ImportError(
            f"'{name}' is no longer exported from 'apex_rag'. "
            f"Use: {_moved[name]}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
