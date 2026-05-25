from typing import Protocol

from apex_rag.enterprise.auth.models import TenantContext


class DistributedIndexer(Protocol):
    """
    Protocol for Distributed Ingestion Queues.
    Implementations (like Celery or Redis) must accept document payloads and TenantContext
    to ensure isolated processing across worker nodes.
    """
    async def queue_ingestion(self, file_bytes: bytes, filename: str, context: TenantContext) -> str:
        """
        Submits a document to the distributed queue for parsing, AST generation, and summarization.
        Returns a tracking Job ID.
        """
        ...

    async def get_job_status(self, job_id: str, context: TenantContext) -> str:
        """
        Retrieves the status of an ingestion job. Must verify the tenant_id matches.
        """
        ...
