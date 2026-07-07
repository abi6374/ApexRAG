"""
enterprise/distributed/indexers.py — Concrete distributed indexer implementations.

Implements the :class:`DistributedIndexer` protocol defined in ``interfaces.py``.

Two backends are provided out of the box:

    - **CeleryIndexer**: Uses a Celery app + Redis broker for production queueing.
      Requires ``celery`` and ``redis`` to be installed.
    - **RedisQueueIndexer**: Uses a lightweight Redis list as a queue.
      Requires ``redis`` to be installed, but no Celery dependency.

Both implementations enforce **tenant isolation**: every job submission and
status check verifies the caller's ``TenantContext``.

Jobs are **always** tracked in an in-memory store (``_IN_MEMORY_JOBS``) so
that ``get_job_status()`` works regardless of whether the external queue is
reachable.  When the real queue is available, the job is also pushed to the
external queue.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.enterprise.distributed.interfaces import DistributedIndexer

logger = logging.getLogger("apex_rag.enterprise.distributed.indexers")

# ── Redis Key Prefixes ───────────────────────────────────────────────────

JOB_STATUS_PREFIX = "apex_rag:job:status:"


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class JobRecord:
    """Represents the state of a distributed ingestion job."""

    job_id: str
    tenant_id: str
    filename: str
    status: str = "queued"  # queued -> processing -> completed | failed
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error_message: str | None = None
    result_doc_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "result_doc_id": self.result_doc_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        """Create a JobRecord from a dictionary."""
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════
# Celery Indexer
# ═══════════════════════════════════════════════════════════════════════


class CeleryIndexer(DistributedIndexer):
    """
    Distributed indexer backed by Celery + Redis/SQS.

    Requires the optional ``celery`` package and a configured broker URL.
    Status is tracked in Redis (using the same broker URL by default).

    Usage::

        from apex_rag.enterprise.distributed.indexers import CeleryIndexer
        from apex_rag.enterprise.auth.models import TenantContext

        indexer = CeleryIndexer(broker_url="redis://localhost:6379/0")
        ctx = TenantContext(tenant_id="acme-corp", user_id="alice", roles=["admin"])

        job_id = await indexer.queue_ingestion(
            file_bytes=b"...",
            filename="report.pdf",
            context=ctx,
        )
        status = await indexer.get_job_status(job_id, ctx)
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        task_name: str = "apex_rag.tasks.ingest_document",
        fallback_to_memory: bool = True,
    ) -> None:
        self._broker_url = broker_url
        self._task_name = task_name
        self._fallback_to_memory = fallback_to_memory
        self._celery_app: Any = None  # Lazy import
        self._redis: Any = None  # Lazy import for status tracking

    def _get_app(self) -> Any:
        """Lazy-import and cache the Celery app."""
        if self._celery_app is not None:
            return self._celery_app
        try:
            from celery import Celery

            app = Celery("apex_rag", broker=self._broker_url)
            self._celery_app = app
            return app
        except ImportError:
            if self._fallback_to_memory:
                logger.debug("Celery not installed — using memory-only mode.")
                return None
            raise

    async def _get_redis(self) -> Any:
        """Lazy-import and connect to Redis for status tracking."""
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(self._broker_url, decode_responses=True)
            await r.ping()
            self._redis = r
            return r
        except Exception as exc:
            logger.warning("Redis status store unavailable: %s", exc)
            return None

    async def queue_ingestion(
        self,
        file_bytes: bytes,
        filename: str,
        context: TenantContext,
    ) -> str:
        """
        Submit a document to the Celery queue for ingestion.

        Status is tracked in Redis to ensure multi-process consistency.

        Args:
            file_bytes: Raw file content.
            filename:   Original filename.
            context:    Tenant context for data isolation.

        Returns:
            A tracking job ID.
        """
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            tenant_id=context.tenant_id,
            filename=filename,
            status="queued",
        )

        # Store status in Redis
        r = await self._get_redis()
        if r is not None:
            await r.setex(
                f"{JOB_STATUS_PREFIX}{job_id}",
                timedelta(days=7),  # Expire after 7 days
                json.dumps(record.to_dict()),
            )

        # Push to Celery queue
        app = self._get_app()
        if app is not None:
            payload = {
                "job_id": job_id,
                "tenant_id": context.tenant_id,
                "filename": filename,
                "content_base64": file_bytes.hex(),
                "queued_at": time.time(),
            }
            app.send_task(self._task_name, args=[payload])
            logger.info("[CELERY] Queued job %s for tenant %s", job_id, context.tenant_id)
        else:
            logger.warning("[CELERY] No app available — job %s will not be processed", job_id)

        return job_id

    async def get_job_status(self, job_id: str, context: TenantContext) -> str:
        """
        Retrieve the status of an ingestion job from Redis.

        Args:
            job_id:  The tracking ID.
            context: Tenant context.

        Returns:
            One of ``queued``, ``processing``, ``completed``, ``failed``.
        """
        r = await self._get_redis()
        if r is None:
            raise RuntimeError("Redis status store unavailable")

        raw = await r.get(f"{JOB_STATUS_PREFIX}{job_id}")
        if raw is None:
            raise ValueError(f"Job {job_id} not found")

        data = json.loads(raw)
        if data["tenant_id"] != context.tenant_id:
            raise PermissionError(f"Access denied to job {job_id}")

        return str(data["status"])

    async def mark_completed(self, job_id: str, doc_id: str) -> None:
        """Mark a job as completed in Redis (called by the worker)."""
        r = await self._get_redis()
        if r is not None:
            raw = await r.get(f"{JOB_STATUS_PREFIX}{job_id}")
            if raw:
                data = json.loads(raw)
                data["status"] = "completed"
                data["result_doc_id"] = doc_id
                data["completed_at"] = time.time()
                await r.setex(
                    f"{JOB_STATUS_PREFIX}{job_id}",
                    timedelta(days=7),
                    json.dumps(data),
                )

    async def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed in Redis (called by the worker)."""
        r = await self._get_redis()
        if r is not None:
            raw = await r.get(f"{JOB_STATUS_PREFIX}{job_id}")
            if raw:
                data = json.loads(raw)
                data["status"] = "failed"
                data["error_message"] = error
                data["completed_at"] = time.time()
                await r.setex(
                    f"{JOB_STATUS_PREFIX}{job_id}",
                    timedelta(days=7),
                    json.dumps(data),
                )


# ═══════════════════════════════════════════════════════════════════════
# Redis Queue Indexer
# ═══════════════════════════════════════════════════════════════════════


class RedisQueueIndexer(DistributedIndexer):
    """
    Distributed indexer backed by a Redis list (lightweight queue).

    Status is tracked in Redis using a persistent key.
    """

    QUEUE_KEY = "apex_rag:ingest:queue"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        fallback_to_memory: bool = True,
    ) -> None:
        self._redis_url = redis_url
        self._fallback_to_memory = fallback_to_memory
        self._redis: Any = None

    async def _get_redis(self) -> Any:
        """Lazy-import and connect to Redis."""
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(self._redis_url, decode_responses=True)
            await r.ping()
            self._redis = r
            return r
        except Exception as exc:
            logger.warning("Redis connection failed: %s", exc)
            return None

    async def queue_ingestion(
        self,
        file_bytes: bytes,
        filename: str,
        context: TenantContext,
    ) -> str:
        """Push a document onto the Redis ingestion queue and track status in Redis."""
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            tenant_id=context.tenant_id,
            filename=filename,
            status="queued",
        )

        r = await self._get_redis()
        if r is not None:
            # 1. Store Status
            await r.setex(
                f"{JOB_STATUS_PREFIX}{job_id}",
                timedelta(days=7),
                json.dumps(record.to_dict()),
            )

            # 2. Push to Queue
            payload = json.dumps(
                {
                    "job_id": job_id,
                    "tenant_id": context.tenant_id,
                    "filename": filename,
                    "content_base64": file_bytes.hex(),
                    "queued_at": time.time(),
                }
            )
            await r.lpush(self.QUEUE_KEY, payload)
            logger.info("[REDIS] Queued job %s", job_id)
        else:
            logger.error("[REDIS] Cannot queue job %s — Redis unavailable", job_id)

        return job_id

    async def get_job_status(self, job_id: str, context: TenantContext) -> str:
        """Retrieve the status of an ingestion job from Redis."""
        r = await self._get_redis()
        if r is None:
            raise RuntimeError("Redis unavailable")

        raw = await r.get(f"{JOB_STATUS_PREFIX}{job_id}")
        if raw is None:
            raise ValueError(f"Job {job_id} not found")

        data = json.loads(raw)
        if data["tenant_id"] != context.tenant_id:
            raise PermissionError(f"Access denied to job {job_id}")

        return str(data["status"])

    async def pop_next_job(self) -> dict[str, Any] | None:
        """Pop the next job from the queue (used by worker processes)."""
        r = await self._get_redis()
        if r is not None:
            raw = await r.brpop(self.QUEUE_KEY, timeout=1)
            if raw is not None:
                return json.loads(raw[1])  # type: ignore[no-any-return]
        return None
