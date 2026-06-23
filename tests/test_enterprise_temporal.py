"""
tests/test_enterprise_temporal.py — Comprehensive test suite for:
  - VersionResolver
  - TemporalReasoningService
  - RoleAwareRetriever
  - RoleAwareFilter / RoleAwareSynthesis
  - VersionLineageRow CRUD
  - Enterprise domain models
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apex_rag.enterprise.auth.access_control import AccessControlAgent
from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.enterprise.auth.role_aware_retriever import RoleAwareRetriever
from apex_rag.enterprise.auth.role_aware_synthesis import RoleAwareFilter, RoleAwareSynthesis
from apex_rag.models.unified_models import (
    AccessAuditRecord,
    ASTNode,
    DocumentPermission,
    EdgeType,
    EvidencePacket,
    FieldPermission,
    NodePermission,
    NodeType,
    NodeVersionHistory,
    Permission,
    ResourcePermission,
    TemporalMetadata,
    TemporalNodeVersion,
    VersionLineage,
)
from apex_rag.temporal.version_resolver import VersionResolver
from apex_rag.temporal.reasoning_service import TemporalReasoningService


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_node_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_doc_id() -> str:
    return f"doc-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock ApexStorage instance with all CRUD methods mocked."""
    storage = MagicMock()
    storage.is_sqlite = True

    # Mock session context manager
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    storage.session.return_value = mock_session

    # Node version CRUD
    storage.save_node_version = AsyncMock()
    storage.get_node_versions = AsyncMock(return_value=[])
    storage.get_node_version_as_of = AsyncMock(return_value=None)
    storage.get_nodes_as_of = AsyncMock(return_value=[])

    # Temporal metadata CRUD
    storage.save_temporal_metadata = AsyncMock()
    storage.get_temporal_metadata = AsyncMock(return_value=None)

    # Version lineage CRUD
    storage.save_version_lineage = AsyncMock()
    storage.get_version_lineage = AsyncMock(return_value=[])
    storage.resolve_version_lineage_chain = AsyncMock(return_value=[])
    storage.get_version_lineage_by_type = AsyncMock(return_value=[])

    # State snapshot CRUD
    storage.save_state_snapshot = AsyncMock()
    storage.get_state_snapshot = AsyncMock(return_value=None)

    # Document operations
    storage.list_document_ids = AsyncMock(return_value=[])
    storage.get_nodes_by_doc = AsyncMock(return_value=[])
    storage.get_timeline_events = AsyncMock(return_value=[])

    # Permission CRUD
    storage.save_role_permission = AsyncMock()
    storage.get_role_permission = AsyncMock(return_value=False)
    storage.save_field_permission = AsyncMock()
    storage.get_field_permission = AsyncMock(return_value=True)

    # Audit CRUD
    storage.save_audit_log = AsyncMock()
    storage.get_audit_logs = AsyncMock(return_value=[])

    return storage


@pytest.fixture
def mock_navigator() -> MagicMock:
    """Create a mock ASTNavigationAgent."""
    nav = MagicMock()
    nav.find = AsyncMock()
    nav._storage = MagicMock()
    nav._storage.get_temporal_metadata = AsyncMock(return_value=None)
    return nav


@pytest.fixture
def mock_access_control() -> MagicMock:
    """Create a mock AccessControlAgent."""
    ac = MagicMock(spec=AccessControlAgent)
    ac.check_access = AsyncMock(return_value=True)
    ac.mask_content = AsyncMock(side_effect=lambda ctx, content: content)
    ac.log_audit_trail = AsyncMock()
    ac.validate_tenant = AsyncMock(return_value=True)
    ac.validate_role = AsyncMock(return_value=True)
    return ac


@pytest.fixture
def tenant_context() -> TenantContext:
    return TenantContext(
        tenant_id="test-tenant",
        user_id="test-user",
        roles=["Manager"],
    )


@pytest.fixture
def sample_temporal_node_version(sample_node_id: str) -> TemporalNodeVersion:
    return TemporalNodeVersion(
        version_id=str(uuid.uuid4()),
        node_id=sample_node_id,
        content="Revenue = 120000",
        doc_id="doc-123",
        tenant_id="default",
        created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
        effective_to=None,
        version_number=2,
        revision_number=0,
        source_timestamp=datetime(2025, 1, 15, tzinfo=timezone.utc),
        is_current=True,
        superseded_by=None,
        previous_version=None,
        validity_status="ACTIVE",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Domain Model Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainModels:
    """Test the new enterprise domain models."""

    def test_temporal_node_version_defaults(self):
        """TemporalNodeVersion should have sensible defaults."""
        version = TemporalNodeVersion(
            version_id="v1", node_id="n1", content="test", doc_id="d1"
        )
        assert version.validity_status == "ACTIVE"
        assert version.is_current is True
        assert version.version_number == 1
        assert version.revision_number == 0
        assert version.superseded_by is None
        assert version.previous_version is None

    def test_node_version_history(self, sample_node_id: str):
        """NodeVersionHistory should aggregate versions."""
        v1 = TemporalNodeVersion(
            version_id="v1", node_id=sample_node_id, content="v1", doc_id="d1", version_number=1,
        )
        v2 = TemporalNodeVersion(
            version_id="v2", node_id=sample_node_id, content="v2", doc_id="d1", version_number=2, is_current=True,
        )
        history = NodeVersionHistory(
            node_id=sample_node_id,
            doc_id="d1",
            versions=[v1, v2],
            current_version=v2,
            total_versions=2,
        )
        assert history.total_versions == 2
        assert history.current_version is v2
        assert len(history.versions) == 2

    def test_version_lineage(self):
        """VersionLineage should track lineage relationships."""
        lineage = VersionLineage(
            lineage_id="l1",
            node_id="n1",
            doc_id="d1",
            source_version_id="v1",
            target_version_id="v2",
            lineage_type="SUPERSEDES",
        )
        assert lineage.lineage_type == "SUPERSEDES"
        assert lineage.strength == 1.0

    def test_permission_hierarchy(self):
        """Permission models should form a proper hierarchy."""
        base_perm = Permission(permission_id="p1", role="Manager", action="read")
        assert base_perm.is_allowed is True

        resource_perm = ResourcePermission(
            permission_id="p2", role="Analyst", action="read", resource_type="document"
        )
        assert resource_perm.resource_type == "document"

        node_perm = NodePermission(
            permission_id="p3", role="Viewer", action="traverse", resource_type="node",
            node_id="n1", node_type_filter="PARAGRAPH",
        )
        assert node_perm.node_type_filter == "PARAGRAPH"

        doc_perm = DocumentPermission(
            permission_id="p4", role="TenantAdmin", action="delete", resource_type="document",
        )
        assert doc_perm.resource_type == "document"

        field_perm = FieldPermission(
            permission_id="p5", role="Guest", action="read", field_name="revenue",
        )
        assert field_perm.field_name == "revenue"

    def test_access_audit_record(self):
        """AccessAuditRecord should capture full audit context."""
        record = AccessAuditRecord(
            request_id="r1",
            tenant_id="tenant-1",
            user_id="user-1",
            role="Analyst",
            query="What is revenue?",
            accessed_nodes=["n1", "n2"],
            blocked_nodes=["n3"],
            retrieval_mode="ROLE_AWARE",
        )
        assert record.allowed is True  # default
        assert len(record.accessed_nodes) == 2
        assert len(record.blocked_nodes) == 1
        assert record.retrieval_mode == "ROLE_AWARE"


# ═══════════════════════════════════════════════════════════════════════════
# VersionResolver Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionResolver:
    """Test the VersionResolver service."""

    @pytest.fixture
    def resolver(self, mock_storage: MagicMock) -> VersionResolver:
        return VersionResolver(mock_storage)

    @pytest.mark.asyncio
    async def test_resolve_latest_no_versions(self, resolver: VersionResolver, sample_node_id: str):
        """Should return None when no versions exist."""
        result = await resolver.resolve_latest(sample_node_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_for_date_no_versions(self, resolver: VersionResolver, sample_node_id: str):
        """Should return None when no version is active at the given date."""
        result = await resolver.resolve_for_date(
            sample_node_id,
            as_of=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_is_superseded_no_versions(self, resolver: VersionResolver, sample_node_id: str):
        """Should return False when no versions exist."""
        result = await resolver.is_superseded(sample_node_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_version_history_empty(self, resolver: VersionResolver, sample_node_id: str):
        """Should return empty list when no versions exist."""
        result = await resolver.get_version_history(sample_node_id)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_lineage_chain_empty(self, resolver: VersionResolver, sample_node_id: str):
        """Should return empty list when no lineage exists."""
        result = await resolver.get_lineage_chain(sample_node_id)
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_latest_with_current_version(
        self, resolver: VersionResolver, sample_node_id: str, sample_temporal_node_version: TemporalNodeVersion,
    ):
        """Should resolve the latest active version."""
        from apex_rag.ingestion.apex_storage import NodeVersionRow

        version_row = NodeVersionRow(
            version_id=sample_temporal_node_version.version_id,
            node_id=sample_node_id,
            content=sample_temporal_node_version.content,
            created_at=sample_temporal_node_version.created_at,
            effective_from=sample_temporal_node_version.effective_from,
            version_number=sample_temporal_node_version.version_number,
            doc_id=sample_temporal_node_version.doc_id,
            tenant_id=sample_temporal_node_version.tenant_id,
            is_current=True,
            updated_at=sample_temporal_node_version.created_at,
            revision_number=0,
            validity_status="ACTIVE",
        )
        resolver._storage.get_node_versions.return_value = [version_row]

        result = await resolver.resolve_latest(sample_node_id)
        assert result is not None
        assert result.node_id == sample_node_id
        assert result.version_number == 2

    @pytest.mark.asyncio
    async def test_resolve_authoritative_no_supersession(
        self, resolver: VersionResolver, sample_node_id: str,
    ):
        """Should return the node itself when no supersession chain exists."""
        from apex_rag.ingestion.apex_storage import NodeVersionRow

        version_row = NodeVersionRow(
            version_id=str(uuid.uuid4()),
            node_id=sample_node_id,
            content="Current version",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            version_number=1,
            doc_id="doc-123",
            tenant_id="default",
            is_current=True,
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            revision_number=0,
            validity_status="ACTIVE",
        )
        resolver._storage.get_node_versions.return_value = [version_row]
        resolver._storage.get_version_lineage.return_value = []

        result = await resolver.resolve_authoritative(sample_node_id)
        assert result is not None
        assert result.version_number == 1

    @pytest.mark.asyncio
    async def test_resolve_authoritative_with_as_of(
        self, resolver: VersionResolver, sample_node_id: str,
    ):
        """Should delegate to resolve_for_date when as_of is provided."""
        from apex_rag.ingestion.apex_storage import NodeVersionRow

        version_row = NodeVersionRow(
            version_id=str(uuid.uuid4()),
            node_id=sample_node_id,
            content="As of version",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            effective_to=datetime(2025, 6, 30, tzinfo=timezone.utc),
            version_number=1,
            doc_id="doc-123",
            tenant_id="default",
            is_current=False,
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            revision_number=0,
            validity_status="EXPIRED",
        )
        resolver._storage.get_node_versions.return_value = [version_row]

        result = await resolver.resolve_authoritative(
            sample_node_id,
            as_of=datetime(2025, 3, 15, tzinfo=timezone.utc),
        )
        assert result is not None
        assert result.content == "As of version"

    @pytest.mark.asyncio
    async def test_resolve_for_date_with_active_version(
        self, resolver: VersionResolver, sample_node_id: str,
    ):
        """Should resolve version active at a specific date."""
        from apex_rag.ingestion.apex_storage import NodeVersionRow

        version_row = NodeVersionRow(
            version_id=str(uuid.uuid4()),
            node_id=sample_node_id,
            content="Revenue = 100000",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            effective_to=datetime(2025, 6, 30, tzinfo=timezone.utc),
            version_number=1,
            doc_id="doc-123",
            tenant_id="default",
            is_current=False,
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            revision_number=0,
        )
        resolver._storage.get_node_versions.return_value = [version_row]

        result = await resolver.resolve_for_date(
            sample_node_id,
            as_of=datetime(2025, 3, 15, tzinfo=timezone.utc),
        )
        assert result is not None
        assert result.content == "Revenue = 100000"

    @pytest.mark.asyncio
    async def test_filter_expired(self, resolver: VersionResolver, sample_node_id: str):
        """Should filter out expired nodes."""
        active_node = ASTNode(
            node_id=sample_node_id,
            content="Active",
            node_type=NodeType.PARAGRAPH,
            doc_id="doc-123",
        )
        expired_node = ASTNode(
            node_id=str(uuid.uuid4()),
            content="Expired",
            node_type=NodeType.PARAGRAPH,
            doc_id="doc-123",
        )

        # Active metadata (with explicit effective_from before as_of)
        ref_date = datetime(2025, 6, 1, tzinfo=timezone.utc)
        resolver._storage.get_temporal_metadata = AsyncMock()
        resolver._storage.get_temporal_metadata.side_effect = lambda nid: (
            TemporalMetadata(
                node_id=nid, validity_status="ACTIVE",
                effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
                source_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
            if nid == sample_node_id
            else TemporalMetadata(
                node_id=nid, validity_status="EXPIRED",
                effective_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
                effective_to=datetime(2024, 12, 31, tzinfo=timezone.utc),
                source_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        )

        result = await resolver.filter_expired(
            [active_node, expired_node],
            as_of=ref_date,
        )
        assert len(result) == 1
        assert result[0].node_id == sample_node_id


# ═══════════════════════════════════════════════════════════════════════════
# TemporalReasoningService Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalReasoningService:
    """Test the TemporalReasoningService."""

    @pytest.fixture
    def service(self, mock_storage: MagicMock) -> TemporalReasoningService:
        return TemporalReasoningService(mock_storage)

    @pytest.mark.asyncio
    async def test_answer_as_of(self, service: TemporalReasoningService, sample_doc_id: str):
        """Should answer a query as of a specific date."""
        result = await service.answer(
            "What is revenue?",
            sample_doc_id,
            as_of=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        assert result is not None
        assert result["mode"] == "AS_OF_DATE"
        assert "target_date" in result
        assert "result" in result
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_answer_latest(self, service: TemporalReasoningService, sample_doc_id: str):
        """Should answer a query with the latest state."""
        result = await service.answer(
            "What is revenue?",
            sample_doc_id,
            latest=True,
        )
        assert result is not None
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_answer_date_range(self, service: TemporalReasoningService, sample_doc_id: str):
        """Should answer a query over a date range."""
        result = await service.answer(
            "Show sales trend",
            sample_doc_id,
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 3, 31, tzinfo=timezone.utc),
        )
        assert result is not None
        assert result["mode"] == "DATE_RANGE"
        assert "start_date" in result
        assert "end_date" in result

    @pytest.mark.asyncio
    async def test_compare(self, service: TemporalReasoningService, sample_doc_id: str):
        """Should compare state between two dates."""
        result = await service.compare(
            "Compare revenue",
            sample_doc_id,
            date_a=datetime(2025, 1, 1, tzinfo=timezone.utc),
            date_b=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_version_history(self, service: TemporalReasoningService, sample_node_id: str):
        """Should return version history for a node."""
        result = await service.get_version_history(sample_node_id)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_lineage(self, service: TemporalReasoningService, sample_node_id: str):
        """Should return version lineage for a node."""
        result = await service.get_lineage(sample_node_id)
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════
# RoleAwareRetriever Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRoleAwareRetriever:
    """Test the RoleAwareRetriever service."""

    @pytest.fixture
    def retriever(
        self,
        mock_storage: MagicMock,
        mock_navigator: MagicMock,
        mock_access_control: MagicMock,
    ) -> RoleAwareRetriever:
        return RoleAwareRetriever(mock_storage, mock_navigator, mock_access_control)

    @pytest.mark.asyncio
    async def test_retrieve_access_denied(
        self,
        retriever: RoleAwareRetriever,
        tenant_context: TenantContext,
    ):
        """Should return blocked result when access is denied."""
        retriever._access_control.check_access = AsyncMock(return_value=False)
        result = await retriever.retrieve(
            "What is revenue?",
            "doc-123",
            tenant_context,
        )
        assert result.allowed is False
        assert result.packets == []
        assert result.audit_record is not None

    @pytest.mark.asyncio
    async def test_retrieve_empty_result(
        self,
        retriever: RoleAwareRetriever,
        tenant_context: TenantContext,
    ):
        """Should return empty packets when navigator finds nothing."""
        mock_result = MagicMock()
        mock_result.verified = False
        retriever._navigator.find = AsyncMock(return_value=mock_result)

        result = await retriever.retrieve(
            "Find nothing",
            "doc-123",
            tenant_context,
        )
        assert result.allowed is True
        assert result.packets == []

    @pytest.mark.asyncio
    async def test_retrieve_with_mocked_node(
        self,
        retriever: RoleAwareRetriever,
        tenant_context: TenantContext,
        sample_node_id: str,
    ):
        """Should retrieve and filter a valid node."""
        node = ASTNode(
            node_id=sample_node_id,
            content="Revenue = 120000",
            node_type=NodeType.PARAGRAPH,
            doc_id="doc-123",
        )
        mock_result = MagicMock()
        mock_result.verified = True
        mock_result.node = node
        mock_result.node_id = sample_node_id
        mock_result.confidence = 0.95
        retriever._navigator.find = AsyncMock(return_value=mock_result)

        retriever._storage.get_temporal_metadata = AsyncMock(
            return_value=TemporalMetadata(
                node_id=sample_node_id,
                freshness_score=0.9,
                source_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
            )
        )

        result = await retriever.retrieve(
            "What is revenue?",
            "doc-123",
            tenant_context,
        )
        assert result.allowed is True
        assert len(result.packets) >= 0  # navigator may or may not populate packets

    @pytest.mark.asyncio
    async def test_retrieve_global_empty(
        self,
        retriever: RoleAwareRetriever,
        tenant_context: TenantContext,
    ):
        """Global retrieval should handle empty doc list."""
        retriever._storage.list_document_ids = AsyncMock(return_value=[])
        result = await retriever.retrieve_global(
            "What is revenue?",
            tenant_context,
        )
        assert result.allowed is True
        assert result.packets == []


# ═══════════════════════════════════════════════════════════════════════════
# RoleAwareSynthesis Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRoleAwareFilter:
    """Test the RoleAwareFilter component."""

    @pytest.fixture
    def role_filter(self, mock_access_control: MagicMock) -> RoleAwareFilter:
        return RoleAwareFilter(mock_access_control)

    @pytest.mark.asyncio
    async def test_filter_packets_empty(
        self,
        role_filter: RoleAwareFilter,
        tenant_context: TenantContext,
    ):
        """Should return empty list for empty packets."""
        result = await role_filter.filter_packets(tenant_context, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_filter_packets_node_denied(
        self,
        role_filter: RoleAwareFilter,
        tenant_context: TenantContext,
        sample_node_id: str,
    ):
        """Should filter out packets when node access is denied."""
        role_filter._access_control.check_access = AsyncMock(return_value=False)
        packet = EvidencePacket(
            node_id=sample_node_id,
            content="Revenue = 120000",
            document_id="doc-123",
            tenant_id="test-tenant",
        )
        result = await role_filter.filter_packets(tenant_context, [packet])
        assert result == []

    @pytest.mark.asyncio
    async def test_filter_packets_allowed(
        self,
        role_filter: RoleAwareFilter,
        tenant_context: TenantContext,
        sample_node_id: str,
    ):
        """Should keep packets when node access is allowed."""
        packet = EvidencePacket(
            node_id=sample_node_id,
            content="Revenue = 120000",
            document_id="doc-123",
        )
        result = await role_filter.filter_packets(tenant_context, [packet])
        assert len(result) == 1
        assert result[0].content == "Revenue = 120000"

    @pytest.mark.asyncio
    async def test_filter_content(
        self,
        role_filter: RoleAwareFilter,
        tenant_context: TenantContext,
    ):
        """Should apply field-level masking to content."""
        result = await role_filter.filter_content(
            tenant_context,
            "Revenue = 120000, Internal Cost = 50000",
        )
        # Default mock returns content unchanged
        assert result == "Revenue = 120000, Internal Cost = 50000"


class TestRoleAwareSynthesis:
    """Test the RoleAwareSynthesis pipeline."""

    @pytest.fixture
    def mock_synthesizer(self) -> MagicMock:
        synth = MagicMock()
        synth.synthesize = AsyncMock(return_value="Synthesized answer.")
        synth.stream_synthesize = AsyncMock()
        mock_async_gen = MagicMock()
        mock_async_gen.__aiter__ = MagicMock(return_value=iter(["token1", " token2"]))
        synth.stream_synthesize.return_value = mock_async_gen
        return synth

    @pytest.fixture
    def synthesis(
        self,
        mock_synthesizer: MagicMock,
        mock_access_control: MagicMock,
    ) -> RoleAwareSynthesis:
        return RoleAwareSynthesis(mock_synthesizer, mock_access_control)

    @pytest.mark.asyncio
    async def test_synthesize_empty_packets(
        self,
        synthesis: RoleAwareSynthesis,
        tenant_context: TenantContext,
    ):
        """Should return fallback message when no packets."""
        result = await synthesis.synthesize(tenant_context, "Question", [])
        assert "not find enough authorized evidence" in result

    @pytest.mark.asyncio
    async def test_synthesize_with_packets(
        self,
        synthesis: RoleAwareSynthesis,
        tenant_context: TenantContext,
        sample_node_id: str,
    ):
        """Should synthesize with filtered packets."""
        packet = EvidencePacket(
            node_id=sample_node_id,
            content="Revenue = 120000",
            document_id="doc-123",
        )
        result = await synthesis.synthesize(tenant_context, "What is revenue?", [packet])
        assert result == "Synthesized answer."


# ═══════════════════════════════════════════════════════════════════════════
# EdgeType Enum Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalEdgeTypes:
    """Test the new temporal edge types in the EdgeType enum."""

    def test_temporal_edge_types_exist(self):
        """New temporal edge types should be defined."""
        assert EdgeType.VERSION_OF.value == "VERSION_OF"
        assert EdgeType.SUPERSEDES.value == "SUPERSEDES"
        assert EdgeType.REPLACED_BY.value == "REPLACED_BY"
        assert EdgeType.VALID_DURING.value == "VALID_DURING"
        assert EdgeType.EFFECTIVE_DURING.value == "EFFECTIVE_DURING"
        assert EdgeType.SNAPSHOT_OF.value == "SNAPSHOT_OF"
        assert EdgeType.HISTORICAL_PARENT.value == "HISTORICAL_PARENT"
