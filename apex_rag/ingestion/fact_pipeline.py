"""
ingestion/fact_pipeline.py — Async Fact Extraction Pipeline with Idempotency.

PRINCIPLE 2 — Non-Blocking Fact Extraction.
  Ingestion saves the document and enqueues a fact extraction job.
  The job runs asynchronously and never blocks the ingestion response.

PRINCIPLE 20 — Idempotent Async Jobs.
  Every job has a deduplication key (doc_id + content_hash) so that
  re-running the same job is safe.  Jobs track their processing status
  in a dedicated table and are skipped if already completed.

Architecture:
    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
    │ Ingest   │───▶│ FactPipeline │───▶│ FactExtractor│
    │ Document │    │ (enqueue)    │    │ (process)    │
    └──────────┘    └──────────────┘    └──────┬───────┘
                                               ▼
                                        ┌──────────────┐
                                        │  FactStore    │
                                        │  (persist)    │
                                        └──────────────┘

Usage:
    pipeline = FactPipeline(storage, fact_store, extractor)
    job_id = await pipeline.enqueue_document("doc-123", nodes)
    # ... later, in a background worker:
    await pipeline.process_pending_jobs(max_jobs=10)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from apex_rag.graph.dags.fact_dag import FactDagBuilder
from apex_rag.ingestion.apex_storage import ApexBase, ApexStorage
from apex_rag.models.unified_models import ASTNode
from apex_rag.temporal.fact_extractor import FactExtractor
from apex_rag.temporal.fact_store import FactStore, TemporalFact

logger = logging.getLogger("apex_rag.ingestion.fact_pipeline")

# ──────────────────────────────────────────────────────────────────────
# FactJobRow — Job Tracking Table
# ──────────────────────────────────────────────────────────────────────


class FactJobRow(ApexBase):
    """SQL row tracking fact extraction job status.

    PRINCIPLE 20 — Idempotent Async Jobs.
    Each job has a deduplication key (``dedup_key`` derived from
    doc_id + content_hash) so that re-running is safe.
    """

    __tablename__ = "fact_extraction_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, default="default"
    )
    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    facts_extracted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_fj_dedup", "dedup_key"),
        Index("ix_fj_status", "status"),
    )


# ──────────────────────────────────────────────────────────────────────
# FactPipeline
# ──────────────────────────────────────────────────────────────────────


class FactPipeline:
    """Async background fact extraction pipeline with idempotency.

    Enqueues fact extraction jobs during ingestion and processes them
    asynchronously.  Supports retry, dead-letter queuing, and
    deduplication.

    Attributes:
        storage:       The :class:`ApexStorage` instance.
        fact_store:    The :class:`FactStore` for persisting extracted facts.
        extractor:     The :class:`FactExtractor` instance.
        max_retries:   Default max retry attempts per job.
        batch_size:    Number of jobs to process per batch.
    """

    def __init__(
        self,
        storage: ApexStorage,
        fact_store: FactStore | None = None,
        extractor: FactExtractor | None = None,
        *,
        max_retries: int = 3,
        batch_size: int = 10,
    ) -> None:
        self._storage = storage
        self._fact_store = fact_store or FactStore(storage)
        self._extractor = extractor or FactExtractor()
        self.max_retries = max_retries
        self.batch_size = batch_size

    # ── Enqueue ────────────────────────────────────────────────────────────

    async def enqueue_document(
        self,
        doc_id: str,
        nodes: list[ASTNode],
        *,
        tenant_id: str = "default",
    ) -> str:
        """Enqueue a document for fact extraction.

        PRINCIPLE 2 — Non-Blocking.
        This method returns immediately after creating the job record.
        Actual extraction runs asynchronously via ``process_pending_jobs()``.

        PRINCIPLE 20 — Idempotency.
        If a job with the same dedup_key already exists and is COMPLETED,
        this method returns the existing job_id without re-enqueueing.

        Args:
            doc_id:    The document ID to extract facts from.
            nodes:     The AST nodes for this document.
            tenant_id: Tenant isolation boundary.

        Returns:
            The job_id of the created (or existing) job.
        """
        # Build dedup_key from doc_id + content hash
        content_str = "".join(n.content or "" for n in nodes)
        content_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:16]
        dedup_key = f"{doc_id}:{content_hash}"

        async with self._storage.session() as session:
            # Check if already completed
            existing = await session.execute(
                select(FactJobRow).where(FactJobRow.dedup_key == dedup_key)
            )
            existing_job = existing.scalars().first()
            if existing_job:
                # Return existing job_id regardless of status (idempotency)
                logger.debug(
                    "Job %s already exists with status %s, skipping.",
                    existing_job.job_id,
                    existing_job.status,
                )
                return existing_job.job_id

            # Create new job
            job_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            job = FactJobRow(
                job_id=job_id,
                doc_id=doc_id,
                tenant_id=tenant_id,
                dedup_key=dedup_key,
                status="PENDING",
                retry_count=0,
                max_retries=self.max_retries,
                created_at=now,
                updated_at=now,
                node_count=len(nodes),
            )
            session.add(job)
            # Store the nodes in a JSON column for the worker to process
            # We use a simple approach: store node IDs; the worker fetches them
            # from the storage layer.

            logger.info(
                "Enqueued fact extraction job %s for doc %s (%d nodes)",
                job_id,
                doc_id,
                len(nodes),
            )
            return job_id

    # ── Process Pending Jobs ───────────────────────────────────────────────

    async def process_pending_jobs(
        self,
        max_jobs: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Process pending fact extraction jobs.

        PRINCIPLE 2 — Non-Blocking.
        Designed to run as a background worker.  Processes up to
        ``batch_size`` jobs per call.

        PRINCIPLE 20 — Idempotent.
        Jobs already COMPLETED are skipped.  Failed jobs are retried
        up to ``max_retries``, then moved to DEAD_LETTER.

        Args:
            max_jobs:   Max jobs to process (defaults to ``batch_size``).
            tenant_id:  Optional tenant filter.

        Returns:
            List of result dicts with keys: job_id, status, facts_extracted, error.
        """
        limit = max_jobs or self.batch_size
        results: list[dict[str, Any]] = []

        async with self._storage.session() as session:
            # Fetch pending or failed (with retries remaining) jobs
            filters = [
                FactJobRow.status.in_(["PENDING", "FAILED"]),
                FactJobRow.retry_count < FactJobRow.max_retries,
            ]
            if tenant_id:
                filters.append(FactJobRow.tenant_id == tenant_id)

            stmt = (
                select(FactJobRow)
                .where(*filters)
                .order_by(FactJobRow.created_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            jobs = list(result.scalars().all())

        if not jobs:
            logger.debug("No pending fact extraction jobs.")
            return results

        logger.info("Processing %d fact extraction jobs...", len(jobs))

        for job in jobs:
            job_result = await self._process_single_job(job)
            results.append(job_result)

        return results

    async def _process_single_job(
        self,
        job: FactJobRow,
    ) -> dict[str, Any]:
        """Process a single fact extraction job.

        Args:
            job: The job row to process.

        Returns:
            Dict with job_id, status, facts_extracted, error.
        """
        job_id = job.job_id
        doc_id = job.doc_id
        tenant_id = job.tenant_id

        try:
            # Mark as PROCESSING
            async with self._storage.session() as session:
                db_job = await session.get(FactJobRow, job_id)
                if db_job is None:
                    return {"job_id": job_id, "status": "FAILED", "error": "Job not found"}
                db_job.status = "PROCESSING"
                db_job.updated_at = datetime.now(timezone.utc)

            # Fetch document nodes
            nodes = await self._storage.get_nodes_by_doc(
                doc_id,
                tenant_context=tenant_id,
            )
            if not nodes:
                logger.warning("No nodes found for doc %s in tenant %s", doc_id, tenant_id)
                async with self._storage.session() as session:
                    db_job = await session.get(FactJobRow, job_id)
                    if db_job:
                        db_job.status = "COMPLETED"
                        db_job.facts_extracted = 0
                        db_job.completed_at = datetime.now(timezone.utc)
                        db_job.updated_at = datetime.now(timezone.utc)
                return {"job_id": job_id, "status": "COMPLETED", "facts_extracted": 0}

            # Extract facts from all nodes
            all_facts: list[TemporalFact] = []
            for node in nodes:
                node_facts = await self._extractor.extract_from_node(
                    node,
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                )
                all_facts.extend(node_facts)

            # Save facts via FactStore
            if all_facts:
                await self._fact_store.save_facts(
                    all_facts,
                    tenant_context=tenant_id,
                )

                # Build FactDAG edges from extracted facts
                fact_dag_builder = FactDagBuilder(self._storage)
                fact_edges = await fact_dag_builder.build(
                    all_facts,
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                )
                if fact_edges:
                    for edge in fact_edges:
                        await self._storage.save_knowledge_edge(edge)
                    logger.info(
                        "FactDAG: %d edges from job %s (%d facts)",
                        len(fact_edges),
                        job_id,
                        len(all_facts),
                    )

            # Mark as COMPLETED
            async with self._storage.session() as session:
                db_job = await session.get(FactJobRow, job_id)
                if db_job:
                    db_job.status = "COMPLETED"
                    db_job.facts_extracted = len(all_facts)
                    db_job.completed_at = datetime.now(timezone.utc)
                    db_job.updated_at = datetime.now(timezone.utc)

            logger.info(
                "Fact extraction job %s completed: %d facts from %d nodes",
                job_id,
                len(all_facts),
                len(nodes),
            )
            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "facts_extracted": len(all_facts),
            }

        except Exception as exc:
            logger.error(
                "Fact extraction job %s failed: %s",
                job_id,
                exc,
                exc_info=True,
            )
            async with self._storage.session() as session:
                db_job = await session.get(FactJobRow, job_id)
                if db_job:
                    db_job.retry_count += 1
                    db_job.error_message = str(exc)[:500]
                    db_job.updated_at = datetime.now(timezone.utc)
                    if db_job.retry_count >= db_job.max_retries:
                        db_job.status = "DEAD_LETTER"
                        logger.warning(
                            "Job %s moved to DEAD_LETTER after %d retries",
                            job_id,
                            db_job.retry_count,
                        )
                    else:
                        db_job.status = "FAILED"

            return {
                "job_id": job_id,
                "status": "FAILED",
                "error": str(exc),
                "retry_count": job.retry_count + 1,
            }

    # ── Queue Management ─────────────────────────────────────────────────

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the current status of a fact extraction job.

        Args:
            job_id: The job UUID.

        Returns:
            Dict with job status info, or None.
        """
        async with self._storage.session() as session:
            job = await session.get(FactJobRow, job_id)
            if job is None:
                return None
            return {
                "job_id": job.job_id,
                "doc_id": job.doc_id,
                "status": job.status,
                "retry_count": job.retry_count,
                "max_retries": job.max_retries,
                "facts_extracted": job.facts_extracted,
                "node_count": job.node_count,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }

    async def retry_dead_letter_jobs(
        self,
        *,
        tenant_id: str | None = None,
    ) -> int:
        """Re-enqueue DEAD_LETTER jobs for a fresh retry attempt.

        Args:
            tenant_id: Optional tenant filter.

        Returns:
            Number of jobs re-enqueued.
        """
        async with self._storage.session() as session:
            filters = [FactJobRow.status == "DEAD_LETTER"]
            if tenant_id:
                filters.append(FactJobRow.tenant_id == tenant_id)

            stmt = select(FactJobRow).where(*filters)
            result = await session.execute(stmt)
            dead_jobs = list(result.scalars().all())

            count = 0
            for job in dead_jobs:
                job.status = "PENDING"
                job.retry_count = 0
                job.error_message = None
                job.updated_at = datetime.now(timezone.utc)
                count += 1

            if count:
                logger.info("Re-enqueued %d DEAD_LETTER jobs", count)
            return count

    async def clean_completed_jobs(
        self,
        older_than_days: int = 7,
        *,
        tenant_id: str | None = None,
    ) -> int:
        """Remove completed jobs older than the specified threshold.

        Args:
            older_than_days: Age threshold in days.
            tenant_id:       Optional tenant filter.

        Returns:
            Number of jobs removed.
        """
        cutoff = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        from datetime import timedelta

        cutoff = cutoff - timedelta(days=older_than_days)

        async with self._storage.session() as session:
            filters = [
                FactJobRow.status == "COMPLETED",
                FactJobRow.completed_at <= cutoff,
            ]
            if tenant_id:
                filters.append(FactJobRow.tenant_id == tenant_id)

            stmt = select(FactJobRow).where(*filters)
            result = await session.execute(stmt)
            old_jobs = list(result.scalars().all())

            for job in old_jobs:
                await session.delete(job)

            return len(old_jobs)
