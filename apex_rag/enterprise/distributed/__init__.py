"""
apex_rag.enterprise.distributed — Distributed ingestion infrastructure.

Provides the :class:`DistributedIndexer` protocol and two concrete
implementations for horizontally scaled document parsing:

    - :class:`CeleryIndexer` — Backed by Celery + Redis/SQS broker.
    - :class:`RedisQueueIndexer` — Lightweight Redis list queue.

Both enforce **tenant isolation** via ``TenantContext`` on every operation.
"""

from apex_rag.enterprise.distributed.indexers import (
    CeleryIndexer,
    JobRecord,
    RedisQueueIndexer,
)
from apex_rag.enterprise.distributed.interfaces import DistributedIndexer

__all__ = [
    "DistributedIndexer",
    "CeleryIndexer",
    "RedisQueueIndexer",
    "JobRecord",
]
