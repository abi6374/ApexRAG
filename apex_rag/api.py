"""
api.py — Production-grade FastAPI REST API for ApexRAG.

Supports:
    - API key authentication via X-API-Key header
    - Configurable CORS origins
    - Health check endpoints (liveness + readiness)
    - In-memory rate limiting
    - File upload validation (size + MIME type)
    - HTML template-based UI (no inline HTML)
    - SSE streaming for real-time agent traces
    - Startup connectivity check

Endpoints:
    GET  /health                   -> Liveness check
    GET  /health/ready             -> Readiness check (DB + Ollama)
    GET  /                          -> Dashboard (HTML)
    POST /documents/ingest/file     -> Ingest a file (multipart upload)
    POST /documents/ingest/text     -> Ingest raw text/markdown
    GET  /documents                 -> List all doc_ids
    GET  /documents/{doc_id}/stats  -> Document statistics
    GET  /documents/{doc_id}/tree   -> Full node tree (JSON)
    GET  /documents/{doc_id}/index  -> Book-style page index (JSON)
    GET  /documents/{doc_id}/index/page -> Visual index page for a document
    POST /documents/{doc_id}/search -> Search the page index
    DELETE /documents/{doc_id}      -> Delete document
    POST /query                     -> Query a document
    POST /query/stream              -> Streamed query (SSE)
    POST /query/global              -> Query across all documents
    POST /query/global/stream       -> Streamed global query (SSE)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text as sa_text

from apex_rag.client import ApexIndex
from apex_rag.config import settings
from apex_rag.exceptions import (
    ApexRAGError,
    DocumentNotFoundError,
    FileValidationError,
)
from apex_rag.navigation import NavigationResult
from apex_rag.utils import logger

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

# Allowed MIME types for file uploads
_ALLOWED_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def validate_file_upload(filename: str, file_size: int, _content_type: str | None) -> None:
    """
    Validate file upload size and extension/MIME type.
    Raises FileValidationError on failure (caught by @app.exception_handler).
    """
    # Size check
    max_bytes = settings.max_upload_bytes
    if file_size > max_bytes:
        size_mb = file_size / (1024 * 1024)
        raise FileValidationError(
            message=f"File too large ({size_mb:.1f} MB). Max: {settings.max_upload_size_mb} MB",
            hint="Reduce file size or increase APEX_MAX_UPLOAD_MB.",
        )

    # Extension check
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_MIME_TYPES:
        raise FileValidationError(
            message=f"Unsupported file type '{suffix}'.",
            hint=f"Allowed types: {', '.join(_ALLOWED_MIME_TYPES)}",
        )


# ---------------------------------------------------------------------------
# Rate Limiter (in-memory, sliding window)
# ---------------------------------------------------------------------------


class InMemoryRateLimiter:
    """
    Simple sliding-window rate limiter.
    Not distributed — use Redis in multi-worker deployments.
    """

    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, max_requests: int = 60, window_seconds: float = 60.0) -> bool:
        now = time.monotonic()
        # Prune expired entries
        self._requests[key] = [t for t in self._requests[key] if now - t < window_seconds]
        if len(self._requests[key]) >= max_requests:
            return False
        self._requests[key].append(now)
        return True


_rate_limiter = InMemoryRateLimiter()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


class AppState:
    """Holds the ApexIndex singleton and startup status."""

    index: ApexIndex | None = None
    started: bool = False
    ollama_reachable: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown."""
    global state

    # -- Startup ------------------------------------------------------------
    try:
        state.index = await ApexIndex.create(
            db_url=settings.db_url,
            ollama_host=settings.ollama_host,
            model=settings.model,
            trace_enabled=settings.trace_enabled,
            verify_leaves=settings.verify_leaves,
        )
        state.started = True

        # Quick connectivity check
        try:
            await state.index.list_documents()
            state.ollama_reachable = True
            logger.info("Ollama connectivity check: OK")
        except Exception as exc:
            state.ollama_reachable = False
            logger.warning("Ollama connectivity check: FAILED — %s", exc)

        logger.info(
            "ApexRAG API started | db=%s | model=%s | auth=%s",
            settings.db_url.split("?")[0],
            settings.model,
            "enabled" if settings.api_key else "disabled",
        )

    except Exception as exc:
        logger.error("ApexRAG API startup FAILED: %s", exc)
        # Don't crash — health endpoint will report unhealthy

    yield

    # -- Shutdown -----------------------------------------------------------
    if state.index:
        await state.index.close()
        logger.info("ApexRAG API shut down.")


def get_index() -> ApexIndex:
    """Dependency: get the ApexIndex singleton."""
    if state.index is None:
        raise HTTPException(503, "ApexIndex not initialised")
    return state.index


# ---------------------------------------------------------------------------
# Middleware: API Key Authentication
# ---------------------------------------------------------------------------


async def api_key_middleware(request: Request, call_next: Any) -> Response:
    """If APEX_API_KEY is set, require it in X-API-Key header."""
    if settings.api_key:
        path = request.url.path
        if path in ("/health", "/health/ready", "/docs", "/redoc", "/openapi.json"):
            return cast(Response, await call_next(request))

        api_key = request.headers.get("X-API-Key", "")
        if api_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "code": "APEX_400",
                    "message": "Missing or invalid API key.",
                    "hint": "Provide via X-API-Key header. Check your APEX_API_KEY environment variable.",
                },
            )

    return cast(Response, await call_next(request))


async def rate_limit_middleware(request: Request, call_next: Any) -> Response:
    """Apply rate limiting based on client IP."""
    path = request.url.path
    # Skip rate limiting for health checks and static docs
    if path in ("/health", "/health/ready"):
        return cast(Response, await call_next(request))

    # Parse rate from settings (e.g., "60/minute")
    try:
        max_r = int(settings.rate_limit.split("/")[0])
    except (ValueError, IndexError):
        max_r = 60

    client_ip: str = request.client.host if request.client else "unknown"
    if not _rate_limiter.check(client_ip, max_requests=max_r):
        return JSONResponse(
            status_code=429,
            content={
                "code": "APEX_401",
                "message": "Rate limit exceeded.",
                "hint": f"Max {max_r} requests per minute. Increase APEX_RATE_LIMIT or slow down.",
            },
        )

    return cast(Response, await call_next(request))


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ApexRAG API",
    description="Local-first Agentic RAG — structural document navigation powered by Ollama.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — configurable via APEX_CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware
app.middleware("http")(api_key_middleware)

# Rate limiting middleware
app.middleware("http")(rate_limit_middleware)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ApexRAGError)
async def apexrag_exception_handler(_request: Request, exc: ApexRAGError) -> JSONResponse:
    """Convert typed ApexRAG errors to JSON responses with status codes."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Convert unhandled exceptions to JSON responses.

    FastAPI-native HTTPException is handled inline; all other errors
    are logged and return a generic 500 response.
    """
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": "APEX_500",
            "message": "Internal server error.",
            "hint": "Check the server logs for details.",
        },
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
async def health_liveness() -> dict[str, Any]:
    """Liveness probe — always returns OK if the app is running."""
    return {"status": "healthy", "app": "apex-rag", "started": state.started}


@app.get("/health/ready", tags=["System"])
async def health_readiness() -> JSONResponse:
    """Readiness probe — checks DB connectivity and Ollama status."""
    issues = []
    if not state.started:
        issues.append("App not fully started")
    if not state.ollama_reachable and state.started:
        issues.append("Ollama not reachable (queries may fail)")

    db_ok = False
    try:
        if state.index:
            async with state.index._storage.session() as session:
                await session.execute(sa_text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        issues.append(f"Database: {exc}")

    status_code = 503 if issues else 200
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "unhealthy" if issues else "healthy",
            "db": db_ok,
            "ollama": state.ollama_reachable,
            "issues": issues,
        },
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IngestTextRequest(BaseModel):
    doc_id: str
    text: str
    synthesize_summaries: bool = True


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    root_node_id: int | None = None
    verify_leaves: bool = True


class GlobalQueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    found: bool
    content: str | None
    node_id: int | None
    path: str | None
    title: str | None
    verified: bool
    confidence: float
    trace: list[list[Any]]  # [[node_id, title], ...]


# ---------------------------------------------------------------------------
# Routes — Documents
# ---------------------------------------------------------------------------


@app.post("/documents/ingest/text", tags=["Documents"])
async def ingest_text(req: IngestTextRequest) -> dict[str, Any]:
    """Ingest raw Markdown/plain text and build a decision tree."""
    idx = get_index()
    doc_id = await idx.ingest_text(
        req.text,
        doc_id=req.doc_id,
        synthesize_summaries=req.synthesize_summaries,
    )
    stats = await idx.get_stats(doc_id)
    return {"ok": True, "doc_id": doc_id, "stats": stats}


@app.post("/documents/ingest/file", tags=["Documents"])
async def ingest_file(
    file: UploadFile = File(...),  # noqa: B008
    doc_id: str | None = Form(default=None),
    synthesize_summaries: bool = Form(default=True),
) -> dict[str, Any]:
    """
    Upload and ingest a document file (PDF, DOCX, MD, TXT, HTML).

    File is validated for:
      - Size (configurable via APEX_MAX_UPLOAD_MB, default 50 MB)
      - Extension / MIME type (PDF, DOCX, MD, TXT, HTML, PPTX, XLSX)
    """
    idx = get_index()

    # Read file content (with size check)
    content = await file.read()
    filename = file.filename or "document"
    content_type = file.content_type

    # Validate
    validate_file_upload(filename, len(content), content_type)

    suffix = Path(filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        ingested_id = await idx.ingest(
            tmp_path,
            doc_id=doc_id,
            synthesize_summaries=synthesize_summaries,
        )
        stats = await idx.get_stats(ingested_id)
        return {"ok": True, "doc_id": ingested_id, "stats": stats}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/documents", tags=["Documents"])
async def list_documents() -> dict[str, Any]:
    """List all indexed document IDs."""
    idx = get_index()
    docs = await idx.list_documents()
    return {"documents": docs, "count": len(docs)}


@app.get("/documents/{doc_id}/stats", tags=["Documents"])
async def get_stats(doc_id: str) -> dict[str, Any]:
    idx = get_index()
    return await idx.get_stats(doc_id)


@app.get("/documents/{doc_id}/tree", tags=["Documents"])
async def get_tree(doc_id: str) -> dict[str, Any]:
    """Return the full decision tree as a flat list ordered by LTree path."""
    idx = get_index()
    nodes = await idx.get_tree(doc_id)
    if not nodes:
        raise DocumentNotFoundError(message=f"Document '{doc_id}' not found or empty.")
    return {"doc_id": doc_id, "node_count": len(nodes), "nodes": nodes}


@app.get("/documents/{doc_id}/export", tags=["Documents"])
async def export_tree(doc_id: str) -> list[dict[str, Any]]:
    """Export the full document tree in the nested PageIndex JSON format."""
    idx = get_index()
    nested_roots = await idx.export_tree(doc_id)
    if not nested_roots:
        raise DocumentNotFoundError(message=f"Document '{doc_id}' not found or empty.")
    return nested_roots


@app.get("/documents/{doc_id}/index", tags=["Documents"])
async def get_page_index(doc_id: str) -> dict[str, Any]:
    """Return the alphabetical page index for a document (JSON)."""
    idx = get_index()
    entries = await idx.get_page_index(doc_id)
    return {"doc_id": doc_id, "entry_count": len(entries), "entries": entries}


@app.post("/documents/{doc_id}/search", tags=["Documents"])
async def search_index(
    doc_id: str,
    q: Annotated[str, Query(description="Search term")] = "",
) -> dict[str, Any]:
    """Search the page index by term (case-insensitive partial match)."""
    idx = get_index()
    results = await idx.search_index(doc_id, q)
    return {"doc_id": doc_id, "query": q, "results": results}


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str) -> dict[str, Any]:
    idx = get_index()
    count = await idx.delete(doc_id)
    return {"ok": True, "doc_id": doc_id, "nodes_deleted": count}


# ---------------------------------------------------------------------------
# Shared SSE Streaming Helper
# ---------------------------------------------------------------------------


async def _stream_query_to_sse(
    task_coro: asyncio.Task[Any],
    event_queue: asyncio.Queue[Any],
) -> AsyncGenerator[str, None]:
    """
    Shared SSE generator: polls an event queue and a background task,
    yielding SSE-formatted events until the task completes.

    Args:
        task_coro:   An asyncio.Task that produces events.
        event_queue: Queue fed by the background task.
    """
    try:
        while True:
            done, _ = await asyncio.wait(
                [asyncio.create_task(event_queue.get()), task_coro],
                return_when=asyncio.FIRST_COMPLETED,
            )
            while not event_queue.empty():
                event = await event_queue.get()
                yield f"data: {json.dumps(event)}\n\n"
            if task_coro.done():
                result = await task_coro
                if isinstance(result, str):
                    final_data = {
                        "event": "result",
                        "found": True,
                        "content": result,
                        "title": "Synthesized Answer",
                        "verified": True,
                    }
                elif result:
                    final_data = {
                        "event": "result",
                        "found": True,
                        "content": result.content,
                        "node_id": result.node_id,
                        "path": result.path,
                        "title": result.title,
                        "verified": result.verified,
                        "confidence": result.confidence,
                    }
                else:
                    final_data = {"event": "result", "found": False}
                yield f"data: {json.dumps(final_data)}\n\n"
                break
    except asyncio.CancelledError:
        task_coro.cancel()
        raise


# ---------------------------------------------------------------------------
# Routes — Query
# ---------------------------------------------------------------------------


def _to_query_response(result: NavigationResult | str | None) -> QueryResponse:
    """Convert a NavigationResult (or str/None) into a QueryResponse."""
    if result is None:
        return QueryResponse(
            found=False,
            content=None,
            node_id=None,
            path=None,
            title=None,
            verified=False,
            confidence=0.0,
            trace=[],
        )
    if isinstance(result, str):
        return QueryResponse(
            found=True,
            content=result,
            node_id=0,
            path="global",
            title="Synthesized Global Answer",
            verified=True,
            confidence=1.0,
            trace=[],
        )
    return QueryResponse(
        found=True,
        content=result.content,
        node_id=result.node_id,
        path=result.path,
        title=result.title,
        verified=result.verified,
        confidence=result.confidence,
        trace=[[nid, t] for nid, t in result.trace],
    )


@app.post("/query", tags=["Query"])
async def query_document(req: QueryRequest) -> QueryResponse:
    """
    Navigate the document tree to answer a natural-language question.

    The agent uses structural navigation + LLM verification to achieve
    high-precision retrieval without vector similarity.
    """
    idx = get_index()
    result = await idx.query(
        req.question,
        req.doc_id,
        root_node_id=req.root_node_id,
    )
    return _to_query_response(result)


@app.post("/query/stream", tags=["Query"])
async def query_document_stream(req: QueryRequest) -> StreamingResponse:
    """
    Stream the document navigation process via Server-Sent Events (SSE).

    Clients will receive real-time updates as the agent enters nodes,
    explores children, makes choices, and verifies leaves.
    """
    idx = get_index()
    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    query_task = asyncio.create_task(
        idx.query(
            req.question,
            req.doc_id,
            root_node_id=req.root_node_id,
            event_queue=event_queue,
        )
    )
    return StreamingResponse(
        _stream_query_to_sse(query_task, event_queue),
        media_type="text/event-stream",
    )


@app.post("/query/global", tags=["Query"])
async def query_global(req: GlobalQueryRequest) -> QueryResponse:
    """Query across all indexed documents."""
    idx = get_index()
    result = await idx.query_global(req.question, synthesize=True)
    return _to_query_response(result)


@app.post("/query/global/stream", tags=["Query"])
async def query_global_stream(req: GlobalQueryRequest) -> StreamingResponse:
    """Stream a global query across all documents via SSE."""
    idx = get_index()
    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    query_task = asyncio.create_task(
        idx.query_global(req.question, event_queue=event_queue, synthesize=True)
    )
    return StreamingResponse(
        _stream_query_to_sse(query_task, event_queue),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Visual UI — HTML Pages
# ---------------------------------------------------------------------------


def _load_template(name: str) -> str:
    """Load an HTML template from the templates directory."""
    template_path = Path(__file__).parent / "templates" / name
    return template_path.read_text(encoding="utf-8")


def _common_styles() -> str:
    return """<style>
:root {
  --bg: #0f0f13;
  --card: #16161e;
  --border: #2a2a3a;
  --accent: #6366f1;
  --text: #e2e4f0;
  --muted: #6b6e8a;
  --leaf: #34d399;
  --grad: linear-gradient(135deg, #6366f1, #a855f7, #06b6d4);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; min-height: 100vh; }
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap" rel="stylesheet">"""


@app.get("/documents/{doc_id}/index/page", response_class=HTMLResponse, tags=["UI"])
async def visual_index_page(doc_id: str) -> HTMLResponse:
    """Render a beautiful visual index page for a specific document."""
    idx = get_index()
    nodes = await idx.get_tree(doc_id)
    entries = await idx.get_page_index(doc_id)
    stats = await idx.get_stats(doc_id)
    if not nodes:
        raise DocumentNotFoundError(message=f"Document '{doc_id}' not found.")

    html = _render_doc_index_page(doc_id, nodes, entries, stats)
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root_index_page() -> HTMLResponse:
    """ApexRAG dashboard — lists all indexed documents."""
    idx = get_index()
    docs = await idx.list_documents()
    doc_stats = []
    for doc_id in docs:
        try:
            s = await idx.get_stats(doc_id)
            doc_stats.append(s)
        except Exception:
            doc_stats.append({"doc_id": doc_id, "total_nodes": 0, "leaf_count": 0, "max_depth": 0})
    html = _render_root_page(doc_stats)
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# HTML renderers — use template-based approach
# ---------------------------------------------------------------------------


def _render_root_page(doc_stats: list[dict[str, Any]]) -> str:
    """Render the root dashboard using the HTML template."""
    template = _load_template("dashboard.html")

    cards_html = ""
    if not doc_stats:
        cards_html = '<div class="empty">No documents indexed yet. Use POST /documents/ingest/file to get started.</div>'
    else:
        for s in doc_stats:
            doc_id = s["doc_id"]
            cards_html += f"""
            <a class="doc-card" href="/documents/{doc_id}/index/page">
                <div class="doc-icon">&#x1F4C4;</div>
                <div class="doc-info">
                    <div class="doc-id">{doc_id}</div>
                    <div class="doc-meta">
                        <span>{s.get("total_nodes", 0)} nodes</span>
                        <span>{s.get("leaf_count", 0)} leaves</span>
                        <span>depth {s.get("max_depth", 0)}</span>
                    </div>
                </div>
                <div class="doc-arrow">&rarr;</div>
            </a>"""

    html = template.replace("{{ common_styles }}", _common_styles())
    html = html.replace("{{ doc_count }}", str(len(doc_stats)))
    html = html.replace("{{ cards_html }}", cards_html)
    return html


def _render_doc_index_page(
    doc_id: str,
    nodes: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    """Render the document index page using the HTML template."""
    template = _load_template("document_view.html")

    # Build tree HTML
    tree_html = _build_tree_html(nodes)

    # Build alphabetical index HTML
    alpha_html = _build_alpha_index_html(entries)

    html = template.replace("{{ common_styles }}", _common_styles())
    html = html.replace("{{ doc_id }}", doc_id)
    html = html.replace("{{ total_nodes }}", str(stats.get("total_nodes", 0)))
    html = html.replace("{{ leaf_count }}", str(stats.get("leaf_count", 0)))
    html = html.replace("{{ max_depth }}", str(stats.get("max_depth", 0)))
    html = html.replace("{{ entry_count }}", str(len(entries)))
    html = html.replace("{{ node_count }}", str(len(nodes)))
    html = html.replace("{{ tree_html }}", tree_html)
    html = html.replace("{{ alpha_html }}", alpha_html)
    return html


def _build_tree_html(nodes: list[dict[str, Any]]) -> str:
    """Render an interactive collapsible tree from the flat node list."""
    # Group children by parent_id
    by_parent: dict[int | None, list[dict[str, Any]]] = {}
    for n in nodes:
        pid = n["parent_id"]
        by_parent.setdefault(pid, []).append(n)

    def render_nodes(parent_id: int | None, depth: int = 0) -> str:
        children = by_parent.get(parent_id, [])
        if not children:
            return ""
        html = ""
        for node in children:
            nid = node["id"]
            has_children = nid in by_parent
            indent_px = depth * 20
            icon = (
                "&#x1F4C4;" if node["is_leaf"] else ("&#x1F4C2;" if has_children else "&#x1F4C1;")
            )
            title_cls = "leaf" if node["is_leaf"] else ""
            toggle = "&#x25BC;" if has_children else " "
            page = (
                f'<span class="node-badge">{node["page_range"]}</span>'
                if node.get("page_range")
                else ""
            )
            path_badge = f'<span class="node-path">{node["path"]}</span>'
            html += f"""
<div class="tree-node" data-id="{nid}" data-has-children="{str(has_children).lower()}" style="padding-left:{indent_px}px">
  <span class="node-toggle">{toggle}</span>
  <div class="node-body">
    <span class="node-icon">{icon}</span>
    <span class="node-title {title_cls}">{node["title"]}</span>
    {page}{path_badge}
  </div>
</div>"""
            if has_children:
                html += f'<div class="children-wrap" id="children-{nid}">'
                html += render_nodes(nid, depth + 1)
                html += "</div>"
        return html

    return render_nodes(None)


def _build_alpha_index_html(entries: list[dict[str, Any]]) -> str:
    """Build alphabetical grouped index HTML."""
    if not entries:
        return '<div style="color:var(--muted);font-size:.85rem">No index entries.</div>'

    by_letter: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        letter = (e["term"] or "?")[0].upper()
        if not letter.isalpha():
            letter = "#"
        by_letter.setdefault(letter, []).append(e)

    html = ""
    for letter in sorted(by_letter.keys()):
        html += f'<div class="alpha-letter">{letter}</div>'
        for e in by_letter[letter]:
            page = e.get("page_start", 0)
            page_end = e.get("page_end", 0)
            if page and page_end and page != page_end:
                page_str = f"p.{page}\u2013{page_end}"
            elif page:
                page_str = f"p.{page}"
            else:
                page_str = ""
            path = e.get("path", "")
            html += f"""<div class="alpha-entry">
  <span class="alpha-term">{e["term"]}</span>
  {f'<span class="alpha-page">{page_str}</span>' if page_str else ""}
  <span class="alpha-path">{path}</span>
</div>"""
    return html
