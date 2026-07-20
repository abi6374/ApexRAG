"""
config.py — Centralised configuration for ApexRAG.

All settings are loaded from environment variables with sensible defaults.
Uses ``os.environ.get()`` — no external dependency required.

Usage:
    from apex_rag.config import settings
    # settings.db_url, settings.ollama_host, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

_VALID_PARSER_BACKENDS: frozenset[str] = frozenset({"markitdown", "docling", "plaintext"})
_VALID_LOG_FORMATS: frozenset[str] = frozenset({"rich", "json"})


class ApexSettings:
    """
    Application settings loaded from environment variables.

    All values are read from environment variables at import time.
    No external dependencies required (pure os.environ.get()).

    ``parser_backend`` and ``log_format`` are validated against
    allowed :class:`Literal` values at access time via properties.
    """

    # ── Database ──────────────────────────────────────────────────────────
    db_url: str = os.getenv("APEX_DB_URL", "sqlite+aiosqlite:///apex.db")
    db_echo: bool = os.getenv("APEX_DB_ECHO", "false").lower() == "true"
    db_pool_size: int = int(os.getenv("APEX_DB_POOL_SIZE", "10"))
    db_max_overflow: int = int(os.getenv("APEX_DB_MAX_OVERFLOW", "20"))

    # ── Ollama / LLM ──────────────────────────────────────────────────────
    ollama_host: str = os.getenv("APEX_OLLAMA_HOST", "http://localhost:11434")
    ollama_timeout: float = float(os.getenv("APEX_OLLAMA_TIMEOUT", "120"))
    model: str = os.getenv("APEX_MODEL", "llama3.1")
    summariser_model: str | None = os.getenv("APEX_SUMMARISER_MODEL") or None
    verifier_model: str | None = os.getenv("APEX_VERIFIER_MODEL") or None
    aggregator_model: str | None = os.getenv("APEX_AGGREGATOR_MODEL") or None

    # ── Ingestion ─────────────────────────────────────────────────────────
    _parser_backend_raw: str = os.getenv("APEX_PARSER_BACKEND", "markitdown")
    max_concurrent_summaries: int = int(os.getenv("APEX_MAX_CONCURRENT_SUMMARIES", "10"))
    verify_leaves: bool = os.getenv("APEX_VERIFY", "true").lower() == "true"

    @property
    def parser_backend(self) -> Literal["markitdown", "docling", "plaintext"]:
        """Return the validated parser backend, falling back to ``markitdown``."""
        if self._parser_backend_raw in _VALID_PARSER_BACKENDS:
            return self._parser_backend_raw  # type: ignore[return-value]
        return "markitdown"

    # ── API Server ────────────────────────────────────────────────────────
    cors_origins: list[str] = [
        o.strip() for o in os.getenv("APEX_CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    api_key: str | None = os.getenv("APEX_API_KEY") or None
    rate_limit: str = os.getenv("APEX_RATE_LIMIT", "60/minute")
    max_upload_size_mb: int = int(os.getenv("APEX_MAX_UPLOAD_MB", "50"))

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = os.getenv("APEX_LOG_LEVEL", "INFO").upper()
    _log_format_raw: str = os.getenv("APEX_LOG_FORMAT", "rich")
    trace_enabled: bool = os.getenv("APEX_TRACE_ENABLED", "true").lower() == "true"

    @property
    def log_format(self) -> Literal["rich", "json"]:
        """Return the validated log format, falling back to ``rich``."""
        if self._log_format_raw in _VALID_LOG_FORMATS:
            return self._log_format_raw  # type: ignore[return-value]
        return "rich"

    # ── Graph / DAG Construction ────────────────────────────────────────────
    graph_construction_mode: str = os.getenv(
        "APEX_GRAPH_MODE", "adaptive"
    ).lower()

    # ── DAG Router Backend ────────────────────────────────────────────────
    router_backend: str = os.getenv(
        "APEX_ROUTER_BACKEND", "heuristic"
    ).lower()
    """
    Which classifier backend the DAGRouter should use:

    - ``"heuristic"`` (default): Regex/keyword-based classifier (always
      available, <10\u00b5s, no dependencies).
    - ``"ml"``: Fine-tuned LogisticRegression on MiniLM sentence embeddings
      (requires ``apex-rag[ml]`` extras installed, ~5ms inference).
      Falls back to ``heuristic`` if the model file is not found.
    """
    """
    Controls how Knowledge DAGs are built during ingestion:

    - ``"adaptive"`` (default): DocumentDAG built eagerly; EntityDAG,
      CitationDAG, and PolicyDAG built lazily on first query that needs
      them; FactDAG and ReasoningDAG are already deferred/query-time.
    - ``"eager"``: All DAGs built synchronously at ingest time (old
      behavior, for backward compatibility and benchmarks).
    - ``"minimal"``: Only DocumentDAG is built — no entity, citation,
      or policy edges at all.  For pure hybrid-search use cases.
    """

    # ── File paths ────────────────────────────────────────────────────────
    data_dir: Path = Path(os.getenv("APEX_DATA_DIR", "."))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def __repr__(self) -> str:
        return (
            f"ApexSettings(db_url={self.db_url!r}, model={self.model!r}, "
            f"parser_backend={self.parser_backend!r}, log_format={self.log_format!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# Singleton — import once, use everywhere
settings = ApexSettings()
