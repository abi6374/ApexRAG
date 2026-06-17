from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from apex_rag.models.unified_models import ASTNode


class DocumentVersion(BaseModel):
    """Represents a specific version of an ingested document."""

    doc_id: str
    version: str
    superseded_by: Optional[str] = None
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    is_authoritative: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentLineageEngine:
    """
    Evolved v3 Document Lineage Engine to track version lineage,
    policy supersessions, expiration dates, and obsolete node suppression.
    """

    def __init__(self) -> None:
        self.versions: Dict[str, DocumentVersion] = {}
        self.supersession_map: Dict[str, str] = {}  # old_doc_id -> new_doc_id

    def register_version(
        self,
        doc_id: str,
        version: str,
        superseded_by: Optional[str] = None,
        effective_from: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        is_authoritative: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Registers a document version in the lineage tracker."""
        effective = effective_from or datetime.now(timezone.utc)
        doc_ver = DocumentVersion(
            doc_id=doc_id,
            version=version,
            superseded_by=superseded_by,
            effective_from=effective,
            expires_at=expires_at,
            is_authoritative=is_authoritative,
            metadata=metadata or {},
        )
        self.versions[doc_id] = doc_ver

        if superseded_by:
            self.supersession_map[doc_id] = superseded_by
            # Update the old version record
            doc_ver.superseded_by = superseded_by
            doc_ver.is_authoritative = False

    def is_superseded(self, doc_id: str) -> bool:
        """Checks if a document ID has been superseded by a newer one."""
        return doc_id in self.supersession_map

    def resolve_latest_version(self, doc_id: str) -> str:
        """Recursively resolves the latest active version ID in the lineage chain."""
        visited = {doc_id}
        current = doc_id
        while current in self.supersession_map:
            next_doc = self.supersession_map[current]
            if next_doc in visited:
                break  # Cycle guard
            visited.add(next_doc)
            current = next_doc
        return current

    def filter_obsolete_nodes(self, nodes: Sequence[ASTNode]) -> list[ASTNode]:
        """
        Suppresses nodes belonging to expired or superseded document versions.
        Prioritizes authoritative revisions.
        """
        active_nodes = []
        now = datetime.now(timezone.utc)

        for node in nodes:
            doc_id = node.doc_id

            # 1. Supersession filter
            if self.is_superseded(doc_id):
                latest_doc_id = self.resolve_latest_version(doc_id)
                # If the latest doc is in our candidate list or available, we suppress the old doc
                logger_msg = f"Suppressing node {node.node_id} because doc {doc_id} is superseded by {latest_doc_id}"
                continue

            # 2. Expiration filter
            version_info = self.versions.get(doc_id)
            if version_info:
                # Ensure current time is within effective ranges
                if version_info.effective_from and now < version_info.effective_from:
                    continue
                if version_info.expires_at and now > version_info.expires_at:
                    continue
                if not version_info.is_authoritative:
                    continue

            active_nodes.append(node)

        return active_nodes

    def get_amendment_chain(self, doc_id: str) -> list[str]:
        """Traces the amendment lineage chain from the oldest to the latest version."""
        chain = [doc_id]
        current = doc_id
        while current in self.supersession_map:
            current = self.supersession_map[current]
            chain.append(current)
        return chain
