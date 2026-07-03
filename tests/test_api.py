"""
test_api.py — FastAPI endpoint tests for the ApexRAG API.

Tests all public endpoints including health checks, document management,
query, LLM streaming, and orchestrated streaming routes.
Uses httpx AsyncClient and patches app state.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from apex_rag.api import app, state
from apex_rag.navigation import NavigationResult


async def _mock_stream_query(
    question: str, doc_id: str, *, tenant_id: str = "default"
) -> AsyncGenerator[str, None]:
    """Mock async generator for ApexIndex.stream_query()."""
    tokens = ["Here", " is", " the", " synthesized", " answer."]
    for t in tokens:
        yield t


async def _mock_llm_stream(
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 150,
    images: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Mock async generator for LLMProvider.stream_generate()."""
    tokens = ["Mock", "LLM", " response"]
    for t in tokens:
        yield t


@pytest.fixture(autouse=True)
def setup_app_state():
    """Ensure the FastAPI app has a valid state for testing."""
    mock_index = MagicMock()
    mock_index.list_documents = AsyncMock(return_value=["doc1", "doc2"])
    mock_index.get_stats = AsyncMock(
        return_value={"doc_id": "doc1", "total_nodes": 10, "leaf_count": 5, "max_depth": 3}
    )
    mock_index.get_tree = AsyncMock(
        return_value=[
            {
                "id": 1,
                "doc_id": "doc1",
                "parent_id": None,
                "path": "1",
                "title": "Root",
                "summary": "Root",
                "depth": 0,
                "position": 1,
                "page_start": 0,
                "page_end": 0,
                "page_range": "",
                "is_leaf": False,
                "has_content": True,
                "meta": {},
            }
        ]
    )
    mock_index.get_page_index = AsyncMock(
        return_value=[
            {
                "id": 1,
                "doc_id": "doc1",
                "node_id": 1,
                "term": "Root",
                "page_start": 1,
                "page_end": 5,
                "path": "1",
            }
        ]
    )
    mock_index.query = AsyncMock(
        return_value=NavigationResult(
            content="Test answer",
            node_id=1,
            path="1.2.3",
            title="Test Section",
            trace=[(1, "Root"), (2, "Test Section")],
            verified=True,
            confidence=0.95,
        )
    )
    mock_index.ingest_text = AsyncMock(return_value="test-doc")
    mock_index.ingest = AsyncMock(return_value="test-doc")
    mock_index.delete = AsyncMock(return_value=5)
    mock_index.export_tree = AsyncMock(return_value=[])
    mock_index.search_index = AsyncMock(return_value=[])
    mock_index.query_global = AsyncMock(
        return_value=NavigationResult(
            content="Global answer",
            node_id=0,
            path="global",
            title="Global",
            trace=[],
            verified=True,
            confidence=1.0,
        )
    )

    # Stream query mock
    mock_index.stream_query = _mock_stream_query

    # Mock the underlying LLM model on the agent
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Mock LLM generated response")
    mock_llm.stream_generate = _mock_llm_stream
    mock_index._agent = MagicMock()
    mock_index._agent._model = mock_llm

    state.index = mock_index
    state.started = True
    state.ollama_reachable = True
    yield
    state.index = None
    state.started = False
    state.ollama_reachable = False


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_health_liveness() -> None:
    """GET /health should return healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["app"] == "apex-rag"


@pytest.mark.asyncio
async def test_health_readiness() -> None:
    """GET /health/ready should return healthy when all systems are up."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_page_returns_html() -> None:
    """GET / should return HTML dashboard."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "ApexRAG" in resp.text


@pytest.mark.asyncio
async def test_docs_available() -> None:
    """Swagger UI and ReDoc should be accessible."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower()

        resp = await client.get("/redoc")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_openapi_spec() -> None:
    """OpenAPI spec should be valid JSON."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert "paths" in spec
        assert "/health" in spec["paths"]
        assert "/query" in spec["paths"]


@pytest.mark.asyncio
async def test_list_documents() -> None:
    """GET /documents should return list of documents."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert "doc1" in data["documents"]


@pytest.mark.asyncio
async def test_api_key_auth_blocks_requests() -> None:
    """When API key is set, requests without it should be rejected."""
    with patch("apex_rag.api.settings.api_key", "test-secret-key"):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/documents")
            assert resp.status_code == 401
            assert "API key" in resp.text

            # With correct key
            resp = await client.get(
                "/documents",
                headers={"X-API-Key": "test-secret-key"},
            )
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limiting_skips_health() -> None:
    """Rate limiter should not block health checks."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_ingest_text_endpoint() -> None:
    """POST /documents/ingest/text should work when index is ready."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/documents/ingest/text",
            json={
                "doc_id": "test-api-doc",
                "text": "# Test\nContent here.",
                "synthesize_summaries": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True


@pytest.mark.asyncio
async def test_file_upload_validation() -> None:
    """File upload should reject unsupported file types."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/documents/ingest/file",
            files={"file": ("test.exe", b"fake content", "application/x-msdownload")},
            data={"synthesize_summaries": "false"},
        )
        assert resp.status_code == 415


@pytest.mark.asyncio
async def test_static_routes_exist() -> None:
    """All documented routes should exist in the OpenAPI spec."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/openapi.json")
        spec = resp.json()
        paths = spec["paths"]

        expected_routes = [
            "/health",
            "/health/ready",
            "/documents/ingest/text",
            "/documents/ingest/file",
            "/documents",
            "/documents/{doc_id}/stats",
            "/documents/{doc_id}/tree",
            "/documents/{doc_id}/export",
            "/documents/{doc_id}/index",
            "/documents/{doc_id}/search",
            "/documents/{doc_id}",
            "/query",
            "/query/stream",
            "/query/global",
            "/query/global/stream",
            "/query/orchestrate/stream",
            "/llm/stream",
            "/llm/generate",
        ]

        for route in expected_routes:
            assert route in paths, f"Route {route} not found in OpenAPI spec"


@pytest.mark.asyncio
async def test_query_endpoint() -> None:
    """POST /query should return a NavigationResult."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/query",
            json={
                "doc_id": "doc1",
                "question": "What is the answer?",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert "Test answer" in data["content"]


@pytest.mark.asyncio
async def test_get_tree() -> None:
    """GET /documents/{doc_id}/tree should return tree nodes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/documents/doc1/tree")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] > 0


@pytest.mark.asyncio
async def test_get_stats() -> None:
    """GET /documents/{doc_id}/stats should return stats."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/documents/doc1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 10


@pytest.mark.asyncio
async def test_visual_index_page() -> None:
    """GET /documents/{doc_id}/index/page should render HTML."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/documents/doc1/index/page")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_query_orchestrate_stream_returns_sse() -> None:
    """POST /query/orchestrate/stream should stream tokens as SSE."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/query/orchestrate/stream",
            json={
                "doc_id": "doc1",
                "question": "What is the answer?",
                "tenant_id": "default",
                "mode": "factual",
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Collect SSE events
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(line[6:])

        # Should contain at least a 'done' event
        done_events = [e for e in events if '"done"' in e or '"event": "done"' in e]
        assert len(done_events) >= 1
        # Should have at least one token event
        token_events = [e for e in events if '"token"' in e or '"event": "token"' in e]
        assert len(token_events) >= 1


@pytest.mark.asyncio
async def test_llm_stream_returns_sse() -> None:
    """POST /llm/stream should stream LLM tokens as SSE."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/llm/stream",
            json={
                "prompt": "Hello world",
                "temperature": 0.5,
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Check we got streamed tokens
        assert '{"event": "token"' in resp.text or '"token"' in resp.text
        assert '{"event": "done"' in resp.text


@pytest.mark.asyncio
async def test_llm_generate_returns_text() -> None:
    """POST /llm/generate should return generated text."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/llm/generate",
            json={
                "prompt": "Generate something",
                "temperature": 0.0,
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert data["content"] == "Mock LLM generated response"


@pytest.mark.asyncio
async def test_llm_generate_with_images() -> None:
    """POST /llm/generate should accept images parameter."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/llm/generate",
            json={
                "prompt": "Describe this image",
                "images": [
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data


@pytest.mark.asyncio
async def test_query_orchestrate_stream_missing_doc() -> None:
    """Orchestrated stream should handle missing documents gracefully."""

    # Patch stream_query to raise an error for missing doc
    async def _mock_stream_error(
        question: str, doc_id: str, *, tenant_id: str = "default"
    ) -> AsyncGenerator[str, None]:
        raise Exception("Document not found")
        yield  # pragma: no cover

    mock_index = state.index
    original = mock_index.stream_query
    mock_index.stream_query = _mock_stream_error

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/query/orchestrate/stream",
            json={
                "doc_id": "nonexistent",
                "question": "Where is it?",
            },
        )
        assert resp.status_code == 200
        assert "error" in resp.text
        assert "Document not found" in resp.text

    mock_index.stream_query = original
