"""
api.py — FastAPI REST API for ApexRAG with a visual index page.

Endpoints:
    GET  /                          → Visual index page (HTML)
    POST /documents/ingest/file     → Ingest a file (multipart upload)
    POST /documents/ingest/text     → Ingest raw text/markdown
    GET  /documents                 → List all doc_ids
    GET  /documents/{doc_id}/stats  → Document statistics
    GET  /documents/{doc_id}/tree   → Full node tree (JSON)
    GET  /documents/{doc_id}/index  → Book-style page index (JSON)
    GET  /documents/{doc_id}/index/page → Visual index page for a document
    POST /documents/{doc_id}/search → Search the page index
    DELETE /documents/{doc_id}      → Delete document
    POST /query                     → Query a document
"""

from __future__ import annotations

import json
import asyncio
import tempfile
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from apex_rag.client import ApexIndex
from apex_rag.utils import logger

# ... (rest of imports and setup)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ApexRAG API",
    description="Local-first Agentic RAG — structural document navigation powered by Ollama.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Singleton index — initialised on startup
_index: ApexIndex | None = None


@app.on_event("startup")
async def startup() -> None:
    global _index
    import os
    _index = await ApexIndex.create(
        db_url=os.getenv("APEX_DB_URL", "sqlite+aiosqlite:///apex.db"),
        ollama_host=os.getenv("APEX_OLLAMA_HOST", "http://localhost:11434"),
        model=os.getenv("APEX_MODEL", "llama3.1"),
        trace_enabled=True,
        verify_leaves=os.getenv("APEX_VERIFY", "true").lower() == "true",
    )
    logger.info("ApexRAG API started.")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _index:
        await _index.close()


def get_index() -> ApexIndex:
    if _index is None:
        raise HTTPException(503, "ApexIndex not initialised")
    return _index


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
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    synthesize_summaries: bool = Form(default=True),
) -> dict[str, Any]:
    """Upload and ingest a document file (PDF, DOCX, MD, TXT)."""
    idx = get_index()
    suffix = Path(file.filename or "doc").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
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
        raise HTTPException(404, f"Document '{doc_id}' not found or empty.")
    return {"doc_id": doc_id, "node_count": len(nodes), "nodes": nodes}


@app.get("/documents/{doc_id}/export", tags=["Documents"])
async def export_tree(doc_id: str) -> list[dict[str, Any]]:
    """Export the full document tree in the nested PageIndex JSON format."""
    idx = get_index()
    nested_roots = await idx.export_tree(doc_id)
    if not nested_roots:
        raise HTTPException(404, f"Document '{doc_id}' not found or empty.")
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
# Routes — Query
# ---------------------------------------------------------------------------

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


@app.post("/query/stream", tags=["Query"])
async def query_document_stream(req: QueryRequest) -> StreamingResponse:
    """
    Stream the document navigation process via Server-Sent Events (SSE).

    Clients will receive real-time updates as the agent enters nodes,
    explores children, makes choices, and verifies leaves.
    """
    idx = get_index()
    event_queue: asyncio.Queue = asyncio.Queue()

    async def event_generator() -> AsyncGenerator[str, None]:
        # Start the query in a background task
        query_task = asyncio.create_task(
            idx.query(
                req.question,
                req.doc_id,
                root_node_id=req.root_node_id,
                event_queue=event_queue,
            )
        )

        try:
            while True:
                # Wait for an event or the task to finish
                done, _ = await asyncio.wait(
                    [asyncio.create_task(event_queue.get()), query_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process all items in the queue
                while not event_queue.empty():
                    event = await event_queue.get()
                    yield f"data: {json.dumps(event)}\n\n"

                if query_task.done():
                    result = await query_task
                    if result:
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
            query_task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/query/global", tags=["Query"])
async def query_global(req: GlobalQueryRequest) -> QueryResponse:
    """Query across all indexed documents."""
    idx = get_index()
    result = await idx.query_global(req.question, synthesize=True)

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


@app.post("/query/global/stream", tags=["Query"])
async def query_global_stream(req: GlobalQueryRequest) -> StreamingResponse:
    """Stream a global query across all documents via SSE."""
    idx = get_index()
    event_queue: asyncio.Queue = asyncio.Queue()

    async def event_generator() -> AsyncGenerator[str, None]:
        query_task = asyncio.create_task(
            idx.query_global(req.question, event_queue=event_queue, synthesize=True)
        )

        try:
            while True:
                done, _ = await asyncio.wait(
                    [asyncio.create_task(event_queue.get()), query_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                while not event_queue.empty():
                    event = await event_queue.get()
                    yield f"data: {json.dumps(event)}\n\n"

                if query_task.done():
                    result = await query_task
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
            query_task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Visual Index Page — HTML UI
# ---------------------------------------------------------------------------

@app.get("/documents/{doc_id}/index/page", response_class=HTMLResponse, tags=["UI"])
async def visual_index_page(doc_id: str) -> HTMLResponse:
    """Render a beautiful visual index page for a specific document."""
    idx = get_index()
    nodes = await idx.get_tree(doc_id)
    entries = await idx.get_page_index(doc_id)
    stats = await idx.get_stats(doc_id)
    if not nodes:
        raise HTTPException(404, f"Document '{doc_id}' not found.")
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
# HTML renderers
# ---------------------------------------------------------------------------

def _render_root_page(doc_stats: list[dict[str, Any]]) -> str:
    cards_html = ""
    if not doc_stats:
        cards_html = '<div class="empty">No documents indexed yet. Use POST /documents/ingest/file to get started.</div>'
    else:
        for s in doc_stats:
            doc_id = s["doc_id"]
            cards_html += f"""
            <a class="doc-card" href="/documents/{doc_id}/index/page">
                <div class="doc-icon">📄</div>
                <div class="doc-info">
                    <div class="doc-id">{doc_id}</div>
                    <div class="doc-meta">
                        <span>{s.get('total_nodes', 0)} nodes</span>
                        <span>{s.get('leaf_count', 0)} leaves</span>
                        <span>depth {s.get('max_depth', 0)}</span>
                    </div>
                </div>
                <div class="doc-arrow">→</div>
            </a>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ApexRAG — Dashboard</title>
{_common_styles()}
<style>
.hero {{ text-align:center; padding: 40px 20px 20px; }}
.hero-logo {{ font-size: 3rem; margin-bottom: 8px; }}
.hero h1 {{ font-size: 2.2rem; font-weight: 800; background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }}
.hero p {{ color: var(--muted); margin-top: 10px; font-size: 1.05rem; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 0 40px 60px; }}
.section-title {{ font-size: 1.2rem; font-weight: 700; margin: 40px 0 20px; display: flex; align-items: center; gap: 12px; }}
.section-title span {{ background: var(--accent); color: white; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }}
.docs-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
.doc-card {{ display: flex; align-items: center; gap: 16px; background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; text-decoration: none; color: inherit; transition: all .2s; cursor: pointer; }}
.doc-card:hover {{ border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(99,102,241,.15); }}
.doc-icon {{ font-size: 2rem; }}
.doc-info {{ flex: 1; }}
.doc-id {{ font-weight: 700; font-size: 1rem; color: var(--text); word-break: break-all; }}
.doc-meta {{ display: flex; gap: 12px; margin-top: 6px; font-size: 0.8rem; color: var(--muted); }}
.doc-arrow {{ color: var(--accent); font-size: 1.3rem; }}
.empty {{ text-align: center; color: var(--muted); padding: 40px 20px; font-size: 1rem; border: 1px dashed var(--border); border-radius: 12px; }}
.api-links {{ display: flex; justify-content: center; gap: 12px; padding: 0 0 20px; flex-wrap: wrap; }}
.api-btn {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 18px; color: var(--accent); text-decoration: none; font-size: 0.85rem; transition: all .15s; }}
.api-btn:hover {{ border-color: var(--accent); background: rgba(99,102,241,.08); }}

/* Global Query */
.global-query-wrap {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 32px; margin-bottom: 40px; }}
.query-box {{ display: flex; gap: 12px; margin-bottom: 24px; }}
.query-input {{ flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; color: var(--text); font-size: 1rem; outline: none; }}
.query-input:focus {{ border-color: var(--accent); }}
.query-btn {{ background: var(--accent); color: white; border: none; border-radius: 8px; padding: 0 32px; font-weight: 700; cursor: pointer; }}
.query-btn:disabled {{ opacity: 0.5; }}
.trace-container {{ background: #000; border-radius: 12px; padding: 20px; font-family: monospace; font-size: 0.85rem; height: 300px; overflow-y: auto; border: 1px solid var(--border); }}
.trace-line {{ margin-bottom: 6px; border-left: 2px solid #333; padding-left: 12px; }}
.trace-line.global_start {{ border-left-color: #a855f7; font-weight: bold; }}
.trace-line.global_chosen {{ border-left-color: #06b6d4; }}
.trace-line.enter {{ border-left-color: var(--accent); }}
.trace-line.result {{ border-left-color: var(--leaf); background: rgba(52, 211, 153, 0.05); padding: 10px; }}
</style>
</head>
<body>
<div class="hero">
  <div class="hero-logo">⚡</div>
  <h1>ApexRAG Dashboard</h1>
  <p>Local-first Agentic RAG — structural document navigation</p>
</div>
<div class="api-links">
  <a class="api-btn" href="/docs">Swagger UI</a>
  <a class="api-btn" href="/redoc">ReDoc</a>
</div>

<div class="container">
  <div class="section-title">🔍 Global Agentic Search <span>New</span></div>
  <div class="global-query-wrap">
    <div class="query-box">
      <input type="text" id="globalQueryInput" class="query-input" placeholder="Search across all indexed documents..." onkeypress="if(event.key==='Enter') runGlobalQuery()">
      <button id="globalQueryBtn" class="query-btn" onclick="runGlobalQuery()">Search All</button>
    </div>
    <div id="globalTraceLog" class="trace-container">
      <div style="color:var(--muted)">Enter a query to see the cross-document navigation agent in action.</div>
    </div>
  </div>

  <div class="section-title">📤 Quick Ingest <span>Drag & Drop</span></div>
  <div class="global-query-wrap" id="dropzone" style="border: 2px dashed var(--border); text-align: center; padding: 40px; cursor: pointer;">
    <div style="font-size: 2rem; margin-bottom: 12px;">📁</div>
    <div style="font-weight: 700; margin-bottom: 4px;">Drag & Drop files here</div>
    <div style="color: var(--muted); font-size: 0.85rem;">PDF, DOCX, Markdown, or Plain Text</div>
    <input type="file" id="fileInput" style="display: none;" onchange="handleFileSelect(this.files[0])">
  </div>

  <div class="section-title">📄 Indexed Documents <span>{len(doc_stats)}</span></div>
  <div class="docs-grid">{cards_html}</div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');

dropzone.onclick = () => fileInput.click();
dropzone.ondragover = (e) => {{ e.preventDefault(); dropzone.style.borderColor = 'var(--accent)'; }};
dropzone.ondragleave = () => {{ dropzone.style.borderColor = 'var(--border)'; }};
dropzone.ondrop = (e) => {{
  e.preventDefault();
  dropzone.style.borderColor = 'var(--border)';
  if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
}};

async function handleFileSelect(file) {{
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  
  dropzone.innerHTML = `<div class="hero-logo" style="font-size: 1.5rem">⏳</div><div style="font-weight: 700">Ingesting ${{file.name}}...</div><div style="color: var(--muted); font-size: 0.85rem">Building decision tree and synthesizing summaries.</div>`;
  
  try {{
    const resp = await fetch('/documents/ingest/file', {{ method: 'POST', body: formData }});
    const data = await resp.json();
    if (data.ok) {{
      location.reload();
    }} else {{
      alert('Ingestion failed: ' + data.error);
    }}
  } catch (err) {{
    alert('Error: ' + err.message);
  }}
}}

async function runGlobalQuery() {{
  const input = document.getElementById('globalQueryInput');
  const btn = document.getElementById('globalQueryBtn');
  const log = document.getElementById('globalTraceLog');
  const q = input.value.trim();
  
  if (!q) return;
  
  input.disabled = true;
  btn.disabled = true;
  log.innerHTML = '';
  
  function addLine(text, type='info') {{
    const line = document.createElement('div');
    line.className = 'trace-line ' + type;
    line.innerHTML = text;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }}

  addLine(`<b>Initializing global search...</b>`, 'global_start');

  try {{
    const response = await fetch('/query/global/stream', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ question: q }})
    }});

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {{
      const {{ value, done }} = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, {{ stream: true }});
      const lines = buffer.split('\\n\\n');
      buffer = lines.pop();

      for (const line of lines) {{
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.substring(6));
        
        switch(data.event) {{
          case 'global_start':
            addLine(`Agent is identifying relevant documents...`, 'global_start');
            break;
          case 'global_chosen':
            addLine(`Selected candidates: <b>${{data.doc_ids.join(', ')}}</b>`, 'global_chosen');
            break;
          case 'enter':
            addLine(`Navigating [${{data.doc_id || ''}}] <b>${{data.title}}</b>`, 'enter');
            break;
          case 'cache_hit':
            addLine(`⚡ <b>Cache Hit!</b> Fast-path retrieval successful.`, 'global_chosen');
            break;
          case 'synthesize_start':
            addLine(`🖋️ <b>Synthesizing final answer...</b>`, 'global_start');
            break;
          case 'result':
            if (data.found) {{
              addLine(`<b>Success!</b> Found in <i>${{data.title || 'Multiple Documents'}}</i>`, 'result');
              const resDiv = document.createElement('div');
              resDiv.style.marginTop = '8px';
              resDiv.style.padding = '12px';
              resDiv.style.background = '#111';
              resDiv.style.borderRadius = '8px';
              resDiv.style.color = 'var(--leaf)';
              resDiv.textContent = data.content;
              log.appendChild(resDiv);
            }} else {{
              addLine(`<b>No results found</b> across all documents.`, 'result');
            }}
            break;
        }
      }}
    }}
  } catch (err) {{
    addLine(`Error: ${{err.message}}`, 'backtrack');
  }} finally {{
    input.disabled = false;
    btn.disabled = false;
    log.scrollTop = log.scrollHeight;
  }}
}}
</script>
</body></html>"""



def _render_doc_index_page(
    doc_id: str,
    nodes: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    # Build tree HTML
    tree_html = _build_tree_html(nodes)

    # Build alphabetical index HTML
    alpha_html = _build_alpha_index_html(entries)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ApexRAG Index — {doc_id}</title>
{_common_styles()}
<style>
.page-header {{ display:flex; align-items:center; gap:16px; padding: 28px 40px 0; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
.back-btn {{ color: var(--accent); text-decoration:none; font-size:.9rem; }}
.back-btn:hover {{ text-decoration:underline; }}
.page-title {{ flex:1; }}
.page-title h1 {{ font-size:1.5rem; font-weight:800; margin:0; }}
.page-title .doc-id-badge {{ display:inline-block; background:rgba(99,102,241,.12); color:var(--accent); border-radius:6px; padding:2px 10px; font-size:.8rem; margin-top:4px; font-family:monospace; }}
.stats-bar {{ display:flex; gap:24px; padding:16px 40px; background: var(--card); border-bottom: 1px solid var(--border); }}
.stat {{ display:flex; flex-direction:column; }}
.stat-val {{ font-size:1.4rem; font-weight:800; color:var(--accent); }}
.stat-label {{ font-size:.75rem; color:var(--muted); margin-top:2px; }}
.layout {{ display:grid; grid-template-columns:1fr 320px; gap:0; height: calc(100vh - 180px); overflow:hidden; }}
.tree-panel {{ overflow-y:auto; padding:24px 32px; border-right:1px solid var(--border); position: relative; }}
.index-panel {{ overflow-y:auto; padding:24px 24px; }}
.panel-title {{ font-size:1rem; font-weight:700; color:var(--text); margin-bottom:16px; display:flex; align-items:center; gap:8px; }}
.panel-title span {{ background:var(--accent); color:#fff; border-radius:4px; padding:1px 8px; font-size:.75rem; }}

/* Tabs */
.tabs {{ display: flex; gap: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--border); }}
.tab {{ padding: 8px 4px; color: var(--muted); font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all 0.2s; border-bottom: 2px solid transparent; }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* Tree */
.tree-node {{ display:flex; align-items:flex-start; gap:0; margin:2px 0; cursor:pointer; }}
.tree-node .indent {{ flex-shrink:0; }}
.node-toggle {{ color:var(--muted); font-size:.75rem; width:18px; text-align:center; flex-shrink:0; user-select:none; }}
.node-body {{ flex:1; display:flex; align-items:baseline; gap:8px; padding:4px 8px; border-radius:6px; transition:background .15s; }}
.tree-node:hover .node-body {{ background: rgba(99,102,241,.08); }}
.node-icon {{ font-size:.85rem; flex-shrink:0; }}
.node-title {{ font-size:.875rem; color:var(--text); font-weight:500; flex:1; }}
.node-title.leaf {{ color: var(--leaf); }}
.node-badge {{ font-size:.7rem; color:var(--muted); background:rgba(255,255,255,.05); border:1px solid var(--border); border-radius:4px; padding:1px 6px; white-space:nowrap; }}
.node-path {{ font-size:.7rem; color:var(--muted); font-family:monospace; }}
.children-wrap.collapsed {{ display:none; }}

/* Query Panel */
.query-box {{ margin-bottom: 24px; }}
.query-input-wrap {{ position: relative; display: flex; gap: 8px; }}
.query-input {{ flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; color: var(--text); font-size: 0.95rem; outline: none; }}
.query-input:focus {{ border-color: var(--accent); }}
.query-btn {{ background: var(--accent); color: white; border: none; border-radius: 8px; padding: 0 24px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }}
.query-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

.trace-container {{ background: #000; border-radius: 12px; padding: 20px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85rem; line-height: 1.5; height: 400px; overflow-y: auto; border: 1px solid var(--border); }}
.trace-line {{ margin-bottom: 4px; border-left: 2px solid #222; padding-left: 12px; }}
.trace-line.enter {{ border-left-color: var(--accent); }}
.trace-line.explore {{ border-left-color: #f59e0b; }}
.trace-line.choice {{ border-left-color: #10b981; }}
.trace-line.leaf {{ border-left-color: var(--leaf); font-weight: bold; }}
.trace-line.verify {{ border-left-color: #6366f1; }}
.trace-line.backtrack {{ border-left-color: #ef4444; color: #ef4444; }}
.trace-line.result {{ border-left-color: var(--grad); margin-top: 12px; padding: 12px; background: rgba(99,102,241,0.1); border-radius: 4px; }}
.trace-timestamp {{ color: var(--muted); font-size: 0.7rem; margin-right: 8px; }}

/* Alpha index */
.alpha-letter {{ font-size:1.1rem; font-weight:800; color:var(--accent); margin:16px 0 6px; padding-bottom:4px; border-bottom:1px solid var(--border); }}
.alpha-entry {{ display:flex; align-items:baseline; gap:8px; padding:5px 0; border-bottom:1px solid rgba(255,255,255,.04); }}
.alpha-entry:last-child {{ border-bottom:none; }}
.alpha-term {{ flex:1; font-size:.83rem; color:var(--text); }}
.alpha-page {{ font-size:.75rem; color:var(--accent); font-weight:600; white-space:nowrap; }}
.alpha-path {{ font-size:.68rem; color:var(--muted); font-family:monospace; }}
/* Search */
.search-wrap {{ margin-bottom:16px; }}
.search-input {{ width:100%; background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:8px 12px; color:var(--text); font-size:.85rem; outline:none; box-sizing:border-box; }}
.search-input:focus {{ border-color:var(--accent); }}
</style>
</head>
<body>
<div class="page-header">
  <a class="back-btn" href="/">← All Documents</a>
  <div class="page-title">
    <h1>Document Index</h1>
    <div class="doc-id-badge">{doc_id}</div>
  </div>
</div>
<div class="stats-bar">
  <div class="stat"><div class="stat-val">{stats.get('total_nodes',0)}</div><div class="stat-label">Total Nodes</div></div>
  <div class="stat"><div class="stat-val">{stats.get('leaf_count',0)}</div><div class="stat-label">Content Leaves</div></div>
  <div class="stat"><div class="stat-val">{stats.get('max_depth',0)}</div><div class="stat-label">Max Depth</div></div>
  <div class="stat"><div class="stat-val">{len(entries)}</div><div class="stat-label">Index Entries</div></div>
</div>
<div class="layout">
  <div class="tree-panel">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('tree')">Decision Tree</div>
      <div class="tab" onclick="switchTab('query')">Agentic Query</div>
    </div>
    
    <div id="tree-content" class="tab-content active">
      <div class="panel-title">🌲 Structural Tree <span>{len(nodes)} nodes</span></div>
      {tree_html}
    </div>

    <div id="query-content" class="tab-content">
      <div class="panel-title">🤖 Navigation Agent</div>
      <div class="query-box">
        <div class="query-input-wrap">
          <input type="text" id="queryInput" class="query-input" placeholder="Ask a question about this document..." onkeypress="if(event.key==='Enter') runQuery()">
          <button id="queryBtn" class="query-btn" onclick="runQuery()">Ask</button>
        </div>
      </div>
      <div id="traceLog" class="trace-container">
        <div class="trace-line" style="color:var(--muted)">The agent's reasoning trace will appear here in real-time.</div>
      </div>
    </div>
  </div>

  <div class="index-panel">
    <div class="panel-title">📖 Page Index <span>{len(entries)}</span></div>
    <div class="search-wrap">
      <input class="search-input" id="indexSearch" placeholder="Filter index…" oninput="filterIndex(this.value)">
    </div>
    <div id="alphaIndex">{alpha_html}</div>
  </div>
</div>
<script>
const DOC_ID = "{doc_id}";

function switchTab(tab) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  
  if (tab === 'tree') {{
    document.querySelector('.tab:nth-child(1)').classList.add('active');
    document.getElementById('tree-content').classList.add('active');
  }} else {{
    document.querySelector('.tab:nth-child(2)').classList.add('active');
    document.getElementById('query-content').classList.add('active');
  }}
}}

// Tree toggle
document.querySelectorAll('.tree-node[data-has-children="true"]').forEach(n => {{
  n.addEventListener('click', e => {{
    e.stopPropagation();
    const wrap = document.getElementById('children-' + n.dataset.id);
    const tog = n.querySelector('.node-toggle');
    if (wrap) {{
      const collapsed = wrap.classList.toggle('collapsed');
      tog.textContent = collapsed ? '▶' : '▼';
    }}
  }});
}});

// Index filter
function filterIndex(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('.alpha-entry').forEach(el => {{
    const term = el.querySelector('.alpha-term').textContent.toLowerCase();
    el.style.display = term.includes(q) ? '' : 'none';
  }});
  document.querySelectorAll('.alpha-letter').forEach(letter => {{
    const entries = letter.nextElementSibling ? letter.nextElementSibling.querySelectorAll('.alpha-entry') : [];
    const visible = [...entries].some(e => e.style.display !== 'none');
    letter.style.display = visible ? '' : 'none';
  }});
}}

async function runQuery() {{
  const input = document.getElementById('queryInput');
  const btn = document.getElementById('queryBtn');
  const log = document.getElementById('traceLog');
  const q = input.value.trim();
  
  if (!q) return;
  
  input.disabled = true;
  btn.disabled = true;
  log.innerHTML = '';
  
  function addLine(text, type='info', data=null) {{
    const line = document.createElement('div');
    line.className = 'trace-line ' + type;
    const ts = new Date().toLocaleTimeString([], {{hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'}});
    line.innerHTML = `<span class="trace-timestamp">${{ts}}</span> ${{text}}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }}

  addLine(`Starting query: "${{q}}"`, 'start');

  try {{
    const response = await fetch('/query/stream', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ doc_id: DOC_ID, question: q }})
    }});

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {{
      const {{ value, done }} = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, {{ stream: true }});
      const lines = buffer.split('\\n\\n');
      buffer = lines.pop();

      for (const line of lines) {{
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.substring(6));
        
        switch(data.event) {{
          case 'enter':
            addLine(`Entering <b>${{data.title}}</b> (path=${{data.path}})`, 'enter');
            break;
          case 'explore':
            addLine(`Evaluating ${{data.child_count}} sub-sections...`, 'explore');
            break;
          case 'choice':
            addLine(`Agent chose <b>node ${{data.chosen_id}}</b>. Reason: <i>${{data.reason}}</i>`, 'choice');
            break;
          case 'leaf':
            addLine(`Leaf reached. Preview: "${{data.content_preview}}..."`, 'leaf');
            break;
          case 'verify':
            addLine(`Verification: <b>${{data.verified ? 'SUCCESS' : 'FAILED'}}</b> (confidence=${{data.confidence.toFixed(2)}})`, 'verify');
            break;
          case 'backtrack':
            addLine(`Backtracking to parent...`, 'backtrack');
            break;
          case 'result':
            if (data.found) {{
              addLine(`<b>Success!</b> Answer found in section: ${{data.title}}`, 'result');
              const resDiv = document.createElement('div');
              resDiv.style.marginTop = '10px';
              resDiv.style.padding = '12px';
              resDiv.style.background = '#111';
              resDiv.style.borderRadius = '8px';
              resDiv.style.whiteSpace = 'pre-wrap';
              resDiv.style.color = 'var(--leaf)';
              resDiv.textContent = data.content;
              log.appendChild(resDiv);
            }} else {{
              addLine(`<b>Failed.</b> No relevant section found for this query.`, 'result');
            }}
            break;
        }
      }}
    }}
  } catch (err) {{
    addLine(`Error: ${{err.message}}`, 'backtrack');
  }} finally {{
    input.disabled = false;
    btn.disabled = false;
    log.scrollTop = log.scrollHeight;
  }}
}}
</script>
</body></html>"""



def _build_tree_html(nodes: list[dict[str, Any]]) -> str:
    """Render an interactive collapsible tree from the flat node list."""
    # Group children by parent_id
    by_parent: dict[int | None, list[dict]] = {}
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
            icon = "📄" if node["is_leaf"] else ("📂" if has_children else "📁")
            title_cls = "leaf" if node["is_leaf"] else ""
            toggle = "▼" if has_children else " "
            page = f'<span class="node-badge">{node["page_range"]}</span>' if node.get("page_range") else ""
            path_badge = f'<span class="node-path">{node["path"]}</span>'
            html += f"""
<div class="tree-node" data-id="{nid}" data-has-children="{str(has_children).lower()}" style="padding-left:{indent_px}px">
  <span class="node-toggle">{toggle}</span>
  <div class="node-body">
    <span class="node-icon">{icon}</span>
    <span class="node-title {title_cls}">{node['title']}</span>
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

    by_letter: dict[str, list[dict]] = {}
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
                page_str = f"p.{page}–{page_end}"
            elif page:
                page_str = f"p.{page}"
            else:
                page_str = ""
            path = e.get("path", "")
            html += f"""<div class="alpha-entry">
  <span class="alpha-term">{e['term']}</span>
  {f'<span class="alpha-page">{page_str}</span>' if page_str else ''}
  <span class="alpha-path">{path}</span>
</div>"""
    return html


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
