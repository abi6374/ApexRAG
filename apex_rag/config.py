"""
config.py — Centralised configuration for ApexRAG.

All settings are loaded from environment variables with sensible defaults.
Uses pydantic-settings for validation and type coercion.

Usage:
    from apex_rag.config import settings
    # settings.db_url, settings.ollama_host, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Lightweight settings — no pydantic dependency required for core library
# ---------------------------------------------------------------------------


class ApexSettings:
    """
    Application settings loaded from environment variables.

    All values are read from environment variables at import time.
    No external dependencies required (pure os.environ.get()).
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
    parser_backend: Literal["markitdown", "docling", "plaintext"] = os.getenv(
        "APEX_PARSER_BACKEND", "markitdown"
    )  # type: ignore[assignment]
    max_concurrent_summaries: int = int(os.getenv("APEX_MAX_CONCURRENT_SUMMARIES", "10"))
    verify_leaves: bool = os.getenv("APEX_VERIFY", "true").lower() == "true"

    # ── API Server ────────────────────────────────────────────────────────
    cors_origins: list[str] = [
        o.strip() for o in os.getenv("APEX_CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    api_key: str | None = os.getenv("APEX_API_KEY") or None
    rate_limit: str = os.getenv("APEX_RATE_LIMIT", "60/minute")
    max_upload_size_mb: int = int(os.getenv("APEX_MAX_UPLOAD_MB", "50"))

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = os.getenv("APEX_LOG_LEVEL", "INFO").upper()
    log_format: Literal["rich", "json"] = os.getenv("APEX_LOG_FORMAT", "rich")  # type: ignore[assignment]
    trace_enabled: bool = os.getenv("APEX_TRACE_ENABLED", "true").lower() == "true"

    # ── File paths ────────────────────────────────────────────────────────
    data_dir: Path = Path(os.getenv("APEX_DATA_DIR", "."))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


# Singleton — import once, use everywhere
settings = ApexSettings()
