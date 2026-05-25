"""
exceptions.py — Structured error hierarchy for ApexRAG.

Every public API call raises typed exceptions with:
  - An error code (for programmatic handling)
  - A human-readable message
  - A resolution hint (guides users toward the fix)

This makes debugging trivial and enables SDK-style error handling.

Usage::

    from apex_rag.exceptions import DocumentNotFoundError

    try:
        result = await index.query("question", doc_id)
    except DocumentNotFoundError as e:
        print(f"[{e.code}] {e.message}")
        print(f"  → Hint: {e.hint}")
"""

from __future__ import annotations


class ApexRAGError(Exception):
    """Base exception for all ApexRAG errors."""

    code: str = "APEX_000"
    message: str = "An unexpected ApexRAG error occurred."
    hint: str = "Check the logs for details. If the issue persists, open a GitHub issue."
    status_code: int = 500

    def __init__(
        self,
        message: str | None = None,
        hint: str | None = None,
        *,
        _cause: BaseException | None = None,
    ) -> None:
        if message:
            self.message = message
        if hint:
            self.hint = hint
        super().__init__(self._format())

    def _format(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }


# ── Configuration Errors ─────────────────────────────────────────────────


class ConfigurationError(ApexRAGError):
    """Raised when the library is misconfigured."""

    code = "APEX_001"
    message = "Configuration error."
    hint = "Check your environment variables and settings."


class InvalidProviderError(ConfigurationError):
    """Raised when an unknown/unavailable LLM provider is specified."""

    code = "APEX_002"
    message = "Invalid or unavailable LLM provider."
    hint = "Install the required package (e.g., 'pip install openai') and check the provider name."


# ── Document Errors ──────────────────────────────────────────────────────


class DocumentNotFoundError(ApexRAGError):
    """Raised when a document ID does not exist in the index."""

    code = "APEX_100"
    message = "Document not found."
    hint = "Use index.list_documents() to see available documents. Did you call ingest() first?"
    status_code = 404


class DocumentExistsError(ApexRAGError):
    """Raised when trying to ingest a document with an existing doc_id."""

    code = "APEX_101"
    message = "Document already exists."
    hint = "Use a different doc_id or delete the existing document first."
    status_code = 409


class IngestionError(ApexRAGError):
    """Raised when document ingestion fails."""

    code = "APEX_102"
    message = "Document ingestion failed."
    hint = "Check that the file format is supported (PDF, DOCX, MD, TXT, HTML). Try converting to Markdown first."
    status_code = 422


# ── Query Errors ─────────────────────────────────────────────────────────


class QueryError(ApexRAGError):
    """Raised when a query fails to execute."""

    code = "APEX_200"
    message = "Query execution failed."
    hint = "Check that the LLM provider is running (e.g., 'ollama serve')."
    status_code = 503


class ProviderError(ApexRAGError):
    """Raised when the LLM provider returns an error."""

    code = "APEX_201"
    message = "LLM provider error."
    hint = "Check that Ollama/OpenAI/Anthropic is running and accessible."


class VerificationError(ApexRAGError):
    """Raised when leaf verification fails unexpectedly."""

    code = "APEX_202"
    message = "Leaf verification failed."
    hint = "This is usually a transient LLM error. Try the query again."


# ── Storage Errors ───────────────────────────────────────────────────────


class StorageError(ApexRAGError):
    """Raised when a database operation fails."""

    code = "APEX_300"
    message = "Database operation failed."
    hint = "Check your database connection and schema."


class DatabaseConnectionError(StorageError):
    """Raised when the database cannot be reached."""

    code = "APEX_301"
    message = "Cannot connect to the database."
    hint = "Verify APEX_DB_URL is correct and the database server is running."
    status_code = 503


# ── API Errors ───────────────────────────────────────────────────────────


class AuthenticationError(ApexRAGError):
    """Raised when API key authentication fails."""

    code = "APEX_400"
    message = "Authentication failed."
    hint = "Provide a valid API key via the X-API-Key header."
    status_code = 401


class RateLimitError(ApexRAGError):
    """Raised when the rate limit is exceeded."""

    code = "APEX_401"
    message = "Rate limit exceeded."
    hint = "Slow down your requests or increase APEX_RATE_LIMIT."
    status_code = 429


class FileValidationError(ApexRAGError):
    """Raised when a file upload fails validation."""

    code = "APEX_402"
    message = "File validation failed."
    hint = "Check the file size (max 50MB) and format (PDF, DOCX, MD, TXT, HTML)."
    status_code = 415
