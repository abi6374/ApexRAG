"""
tests/temporal/test_fact_pipeline.py — Tests for ingestion/fact_pipeline.py.

Covers:
  - FactJobRow model defaults
  - Enqueue document (creates PENDING job)
  - Idempotency (same dedup_key returns existing completed job)
  - process_pending_jobs with actual extraction
  - Retry on failure
  - Dead letter queue
  - get_job_status
  - retry_dead_letter_jobs
  - clean_completed_jobs
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from apex_rag.ingestion.apex_storage import ApexBase, ApexStorage
from apex_rag.ingestion.fact_pipeline import FactJobRow, FactPipeline
from apex_rag.models.unified_models import ASTNode, NodeType
from apex_rag.temporal.fact_extractor import FactExtractor
from apex_rag.temporal.fact_store import FactStore


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage() -> AsyncGenerator[ApexStorage, None]:
    """Create a fresh ApexStorage per test using a temp file.

    Uses ``ApexStorage.create()`` which has production-grade schema
    creation that gracefully handles SQLite's lack of INDEX IF NOT EXISTS.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    storage = await ApexStorage.create(f"sqlite+aiosqlite:///{tmp.name}")
    yield storage


@pytest_asyncio.fixture
async def pipeline(storage: ApexStorage) -> FactPipeline:
    return FactPipeline(
        storage,
        fact_store=FactStore(storage),
        extractor=FactExtractor(),
    )


def make_node(content: str, doc_id: str = "test-doc") -> ASTNode:
    return ASTNode(
        node_id=str(uuid.uuid4()),
        content=content,
        node_type=NodeType.PARAGRAPH,
        doc_id=doc_id,
    )


# ── FactJobRow Model ────────────────────────────────────────────────────


class TestFactJobRowModel:
    """FactJobRow default values."""

    def test_default_fields(self) -> None:
        row = FactJobRow(
            job_id=str(uuid.uuid4()),
            doc_id="doc-1",
            dedup_key="test-key",
            status="PENDING",
            retry_count=0,
            max_retries=3,
            facts_extracted=0,
            node_count=0,
            tenant_id="default",
        )
        assert row.status == "PENDING"
        assert row.retry_count == 0
        assert row.max_retries == 3
        assert row.facts_extracted == 0
        assert row.node_count == 0
        assert row.tenant_id == "default"

    def test_custom_values(self) -> None:
        row = FactJobRow(
            job_id="custom-id",
            doc_id="doc-1",
            tenant_id="tenant-b",
            dedup_key="key-1",
            status="COMPLETED",
            retry_count=2,
            facts_extracted=5,
            node_count=10,
        )
        assert row.job_id == "custom-id"
        assert row.status == "COMPLETED"
        assert row.retry_count == 2
        assert row.facts_extracted == 5


# ── Enqueue ─────────────────────────────────────────────────────────────


class TestEnqueue:
    """Enqueue creates PENDING jobs."""

    @pytest.mark.asyncio
    async def test_enqueue_creates_pending_job(self, pipeline: FactPipeline, storage: ApexStorage) -> None:
        nodes = [make_node("Revenue was $40M.")]
        job_id = await pipeline.enqueue_document("doc-1", nodes, tenant_id="tenant-a")
        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID4 length

    @pytest.mark.asyncio
    async def test_enqueue_returns_different_job_ids_for_different_docs(
        self, pipeline: FactPipeline,
    ) -> None:
        nodes_a = [make_node("Revenue was $40M.", doc_id="doc-a")]
        nodes_b = [make_node("Expenses were $20M.", doc_id="doc-b")]
        job_a = await pipeline.enqueue_document("doc-a", nodes_a, tenant_id="tenant-a")
        job_b = await pipeline.enqueue_document("doc-b", nodes_b, tenant_id="tenant-a")
        assert job_a != job_b


class TestIdempotency:
    """Same dedup_key returns existing completed job (Principle 20)."""

    @pytest.mark.asyncio
    async def test_enqueue_idempotent_for_same_content(
        self, pipeline: FactPipeline,
    ) -> None:
        nodes = [make_node("Revenue was $40M.")]
        job1 = await pipeline.enqueue_document("doc-id", nodes, tenant_id="tenant-a")
        job2 = await pipeline.enqueue_document("doc-id", nodes, tenant_id="tenant-a")
        # Same content → same dedup_key → same job_id
        assert job1 == job2

    @pytest.mark.asyncio
    async def test_different_content_different_job(
        self, pipeline: FactPipeline,
    ) -> None:
        nodes1 = [make_node("Revenue was $40M.")]
        nodes2 = [make_node("Revenue was $60M.")]
        job1 = await pipeline.enqueue_document("doc-id", nodes1, tenant_id="tenant-a")
        job2 = await pipeline.enqueue_document("doc-id", nodes2, tenant_id="tenant-a")
        # Different content → different dedup_key → different job_id
        assert job1 != job2


# ── Job Status ──────────────────────────────────────────────────────────


class TestJobStatus:
    """get_job_status returns correct state."""

    @pytest.mark.asyncio
    async def test_get_job_status_pending(self, pipeline: FactPipeline) -> None:
        nodes = [make_node("Revenue was $40M.")]
        job_id = await pipeline.enqueue_document("doc-status", nodes)
        status = await pipeline.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "PENDING"
        assert status["doc_id"] == "doc-status"

    @pytest.mark.asyncio
    async def test_get_job_status_nonexistent(self, pipeline: FactPipeline) -> None:
        status = await pipeline.get_job_status("nonexistent")
        assert status is None


# ── Process Pending Jobs ────────────────────────────────────────────────


class TestProcessPending:
    """Processing pending jobs extracts facts."""

    @pytest.mark.asyncio
    async def test_process_no_pending_jobs(self, pipeline: FactPipeline) -> None:
        results = await pipeline.process_pending_jobs()
        assert results == []

    @pytest.mark.asyncio
    async def test_process_single_job(self, pipeline: FactPipeline, storage: ApexStorage) -> None:
        # First save a node so the pipeline can fetch it
        node = make_node("Revenue was $40M. Acme Corp reported growth.")
        await storage.save_node(node, tenant_context="tenant-a")

        # Enqueue
        job_id = await pipeline.enqueue_document("test-doc", [node], tenant_id="tenant-a")

        # Process
        results = await pipeline.process_pending_jobs()
        assert len(results) >= 1
        assert results[0]["status"] == "COMPLETED"

        # Check the job status
        status = await pipeline.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_process_job_with_metrics_content(
        self, pipeline: FactPipeline, storage: ApexStorage,
    ) -> None:
        """Extract metrics from document content."""
        node = make_node("Revenue was $50M. Profit was $10M. Headcount is 500.")
        await storage.save_node(node, tenant_context="tenant-a")

        await pipeline.enqueue_document("test-doc", [node], tenant_id="tenant-a")
        results = await pipeline.process_pending_jobs()

        assert len(results) >= 1
        assert results[0]["status"] == "COMPLETED"
        # Should have extracted at least 3 metric facts
        assert results[0]["facts_extracted"] >= 3


# ── Retry & Dead Letter ─────────────────────────────────────────────────


class TestRetryAndDeadLetter:
    """Jobs that fail are retried, then moved to dead letter."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, pipeline: FactPipeline, storage: ApexStorage) -> None:
        node = make_node("Test content")
        await storage.save_node(node, tenant_context="tenant-a")

        # Enqueue
        await pipeline.enqueue_document("test-doc", [node], tenant_id="tenant-a")

        # Make extractor raise by patching it
        with patch.object(pipeline._extractor, "extract_from_node", side_effect=ValueError("Extraction failed")):
            results = await pipeline.process_pending_jobs()
            assert len(results) >= 1
            assert results[0]["status"] == "FAILED"

    @pytest.mark.asyncio
    async def test_dead_letter_after_max_retries(self, pipeline: FactPipeline, storage: ApexStorage) -> None:
        node = make_node("Test content")
        await storage.save_node(node, tenant_context="tenant-a")

        job_id = await pipeline.enqueue_document("test-doc", [node], tenant_id="tenant-a")

        with patch.object(pipeline._extractor, "extract_from_node", side_effect=ValueError("Extraction failed")):
            for _ in range(pipeline.max_retries):  # Hit max retries
                await pipeline.process_pending_jobs()

        status = await pipeline.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "DEAD_LETTER"

    @pytest.mark.asyncio
    async def test_retry_dead_letter_jobs(self, pipeline: FactPipeline, storage: ApexStorage) -> None:
        """retry_dead_letter_jobs resets DEAD_LETTER jobs to PENDING."""
        node = make_node("Test content")
        await storage.save_node(node, tenant_context="tenant-a")

        await pipeline.enqueue_document("test-doc", [node], tenant_id="tenant-a")

        with patch.object(pipeline._extractor, "extract_from_node", side_effect=ValueError("Extraction failed")):
            for _ in range(pipeline.max_retries):
                await pipeline.process_pending_jobs()

        count = await pipeline.retry_dead_letter_jobs()
        assert count >= 1


class TestCleanCompleted:
    """Clean old completed jobs."""

    @pytest.mark.asyncio
    async def test_clean_no_old_jobs(self, pipeline: FactPipeline) -> None:
        """Cleaning with no jobs should return 0."""
        count = await pipeline.clean_completed_jobs(older_than_days=1)
        assert count == 0
