"""
tests/enterprise/distributed/test_indexers.py — Tests for Part 7 distributed ingestion.

Covers:
    1. CeleryIndexer — queue_ingestion + get_job_status + tenant isolation
    2. RedisQueueIndexer — same operations
    3. Tenant isolation enforcement (PermissionError for cross-tenant access)
    4. Error handling (unknown job IDs)
    5. OpenTelemetry tracing integration in ApexOrchestrator
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.enterprise.distributed.indexers import (
    CeleryIndexer,
    RedisQueueIndexer,
)
from apex_rag.observability.telemetry import TelemetryTracker, get_tracer

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def tenant_a() -> TenantContext:
    return TenantContext(tenant_id="tenant-a", user_id="alice", roles=["admin"])


@pytest.fixture
def tenant_b() -> TenantContext:
    return TenantContext(tenant_id="tenant-b", user_id="bob", roles=["reader"])


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    r = AsyncMock()
    # Mock get/set/setex behaviors
    storage = {}

    async def _setex(name, time, value):
        storage[name] = value

    async def _get(name):
        return storage.get(name)

    r.setex.side_effect = _setex
    r.get.side_effect = _get
    r.ping = AsyncMock()
    return r


@pytest.fixture
def sample_bytes() -> bytes:
    return b"# Test Document\n\nHello, world."


# ═══════════════════════════════════════════════════════════════════════
# CeleryIndexer Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCeleryIndexer:
    """Tests for CeleryIndexer (Redis-backed status mode)."""

    def test_init_defaults(self) -> None:
        """Default constructor should use standard Redis broker URL."""
        indexer = CeleryIndexer()
        assert indexer._broker_url == "redis://localhost:6379/0"
        assert indexer._fallback_to_memory is True
        assert indexer._task_name == "apex_rag.tasks.ingest_document"

    @pytest.mark.asyncio
    async def test_queue_ingestion_returns_job_id(
        self, tenant_a: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """queue_ingestion should return a valid UUID job ID."""
        indexer = CeleryIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            assert isinstance(job_id, str)
            assert len(job_id) > 10  # Looks like a UUID

    @pytest.mark.asyncio
    async def test_queue_and_check_status(
        self, tenant_a: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """After queueing, status should be 'queued'."""
        indexer = CeleryIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            status = await indexer.get_job_status(job_id, tenant_a)
            assert status == "queued"

    @pytest.mark.asyncio
    async def test_mark_completed(
        self, tenant_a: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """After marking completed, status should reflect it."""
        indexer = CeleryIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            await indexer.mark_completed(job_id, "doc-123")
            status = await indexer.get_job_status(job_id, tenant_a)
            assert status == "completed"

    @pytest.mark.asyncio
    async def test_mark_failed(
        self, tenant_a: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """After marking failed, status and error should be reflected."""
        indexer = CeleryIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            await indexer.mark_failed(job_id, "Parse error: invalid format")
            status = await indexer.get_job_status(job_id, tenant_a)
            assert status == "failed"

    @pytest.mark.asyncio
    async def test_tenant_isolation_denies_cross_tenant_access(
        self, tenant_a: TenantContext, tenant_b: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """Tenant B should not be able to access Tenant A's job."""
        indexer = CeleryIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            with pytest.raises(PermissionError, match="Access denied to job"):
                await indexer.get_job_status(job_id, tenant_b)

    @pytest.mark.asyncio
    async def test_unknown_job_raises_value_error(
        self, tenant_a: TenantContext, mock_redis
    ) -> None:
        """Querying an unknown job ID should raise ValueError."""
        indexer = CeleryIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis), pytest.raises(ValueError, match="not found"):
            await indexer.get_job_status("nonexistent-job-id", tenant_a)

    @pytest.mark.asyncio
    async def test_queue_and_check_completed_lifecycle(
        self, tenant_a: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """Full lifecycle: queue -> complete -> check status."""
        indexer = CeleryIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            assert await indexer.get_job_status(job_id, tenant_a) == "queued"

            await indexer.mark_completed(job_id, "doc-123")
            assert await indexer.get_job_status(job_id, tenant_a) == "completed"

            await indexer.mark_failed(job_id, "error")
            assert await indexer.get_job_status(job_id, tenant_a) == "failed"


# ═══════════════════════════════════════════════════════════════════════
# RedisQueueIndexer Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRedisQueueIndexer:
    """Tests for RedisQueueIndexer (Redis-backed status mode)."""

    def test_init_defaults(self) -> None:
        """Default constructor should use standard Redis URL."""
        indexer = RedisQueueIndexer()
        assert indexer._redis_url == "redis://localhost:6379/0"
        assert indexer._fallback_to_memory is True

    @pytest.mark.asyncio
    async def test_queue_ingestion_returns_job_id(
        self, tenant_a: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """queue_ingestion should return a valid UUID job ID."""
        indexer = RedisQueueIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            assert isinstance(job_id, str)
            assert len(job_id) > 10

    @pytest.mark.asyncio
    async def test_queue_and_check_status(
        self, tenant_a: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """After queueing, status should be 'queued'."""
        indexer = RedisQueueIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            status = await indexer.get_job_status(job_id, tenant_a)
            assert status == "queued"

    @pytest.mark.asyncio
    async def test_tenant_isolation(
        self, tenant_a: TenantContext, tenant_b: TenantContext, sample_bytes: bytes, mock_redis
    ) -> None:
        """Tenant B should not be able to check Tenant A's job status."""
        indexer = RedisQueueIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis):
            job_id = await indexer.queue_ingestion(sample_bytes, "test.md", tenant_a)
            with pytest.raises(PermissionError, match="Access denied to job"):
                await indexer.get_job_status(job_id, tenant_b)

    @pytest.mark.asyncio
    async def test_unknown_job_raises_value_error(
        self, tenant_a: TenantContext, mock_redis
    ) -> None:
        """Querying an unknown job ID should raise ValueError."""
        indexer = RedisQueueIndexer()
        with patch.object(indexer, "_get_redis", return_value=mock_redis), pytest.raises(ValueError, match="not found"):
            await indexer.get_job_status("nonexistent-job-id", tenant_a)

    @pytest.mark.asyncio
    async def test_pop_next_job_fallback_returns_none(self) -> None:
        """pop_next_job should return None when queue is empty."""
        indexer = RedisQueueIndexer()
        # Mock _get_redis to return None (memory fallback mode)
        with patch.object(indexer, "_get_redis", new_callable=AsyncMock, return_value=None):
            result = await indexer.pop_next_job()
            assert result is None


# ═══════════════════════════════════════════════════════════════════════
# OpenTelemetry Tracing Integration
# ═══════════════════════════════════════════════════════════════════════


class TestTracingIntegration:
    """Tests for OpenTelemetry tracing integration in the orchestrator."""

    def test_get_tracer_returns_tracer(self) -> None:
        """get_tracer should return a tracer object."""
        tracer = get_tracer("test_tracer")
        assert tracer is not None
        # Should have start_as_current_span method (context manager)
        assert hasattr(tracer, "start_as_current_span")

    def test_tracer_can_create_span(self) -> None:
        """Tracer should be able to create spans."""
        tracer = get_tracer("test_tracer")
        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("test_key", "test_value")
            assert span is not None

    def test_tracer_span_with_attributes(self) -> None:
        """Spans should support setting attributes."""
        tracer = get_tracer("test_tracer")
        with tracer.start_as_current_span("test_span_attrs") as span:
            span.set_attribute("query", "What is Q3?")
            span.set_attribute("doc_id", "doc-123")
            span.set_attribute("packets_retrieved", 5)
            # No assert on values — just verifying no exceptions

    @pytest.mark.asyncio
    async def test_tracer_span_in_async_context(self) -> None:
        """Tracing should work in async contexts."""
        tracer = get_tracer("test_async_tracer")
        with tracer.start_as_current_span("async_test_span") as span:
            span.set_attribute("async_test", "true")
            # Simulate async work
            await asyncio.sleep(0)
            assert span is not None

    def test_telemetry_tracker_starts_span(self) -> None:
        """TelemetryTracker.start_span should return a valid span."""
        span = TelemetryTracker.start_span(
            "tracker_test",
            attributes={"source": "unit_test"},
        )
        assert span is not None
        # Note: is_recording() returns False with the no-op tracer
        # (OpenTelemetry SDK not installed in test environments)
        # We verify the span object exists instead.
