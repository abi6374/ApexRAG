"""
tests/test_version_service.py — Tests for the immutable TemporalVersionService (Phase 1).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone

import pytest

from apex_rag.temporal.version_service import TemporalVersionService


class TestTemporalVersionService:
    """Unit tests for pure logic methods."""

    def test_compute_content_hash(self) -> None:
        """Should compute SHA-256 hex digest."""
        h = TemporalVersionService._compute_content_hash("hello")
        assert len(h) == 64  # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_content_hash_consistency(self) -> None:
        """Same content should produce same hash."""
        h1 = TemporalVersionService._compute_content_hash("Revenue = 120,000")
        h2 = TemporalVersionService._compute_content_hash("Revenue = 120,000")
        assert h1 == h2

    def test_compute_content_hash_different(self) -> None:
        """Different content should produce different hashes."""
        h1 = TemporalVersionService._compute_content_hash("Revenue = 100,000")
        h2 = TemporalVersionService._compute_content_hash("Revenue = 200,000")
        assert h1 != h2

    def test_verify_content_integrity_match(self) -> None:
        """Should return True when hash matches content."""
        from apex_rag.ingestion.apex_storage import NodeVersionRow

        version = NodeVersionRow(
            version_id=str(uuid.uuid4()),
            node_id=str(uuid.uuid4()),
            content="Revenue = 120,000",
            content_hash=TemporalVersionService._compute_content_hash("Revenue = 120,000"),
            doc_id="doc-123",
            tenant_id="default",
            is_current=True,
        )
        import asyncio
        result = asyncio.run(TemporalVersionService.verify_content_integrity(version))
        assert result is True

    def test_verify_content_integrity_mismatch(self) -> None:
        """Should return False when hash doesn't match content."""
        from apex_rag.ingestion.apex_storage import NodeVersionRow

        version = NodeVersionRow(
            version_id=str(uuid.uuid4()),
            node_id=str(uuid.uuid4()),
            content="Revenue = 120,000",
            content_hash="0000000000000000000000000000000000000000000000000000000000000000",
            doc_id="doc-123",
            tenant_id="default",
            is_current=True,
        )
        import asyncio
        result = asyncio.run(TemporalVersionService.verify_content_integrity(version))
        assert result is False


class TestTemporalVersionServiceIntegration:
    """Integration tests with actual SQLite databases."""

    @pytest.fixture
    def db_path(self) -> str:
        """Create a unique temp database file per test."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return tmp.name

    @pytest.fixture
    async def service(self, db_path: str) -> TemporalVersionService:
        """Create a service with a fresh SQLite database."""
        from apex_rag.ingestion.apex_storage import ApexStorage
        db_url = f"sqlite+aiosqlite:///{db_path}"
        storage = await ApexStorage.create(db_url)
        yield TemporalVersionService(storage)
        await storage.dispose()

    @pytest.fixture
    async def seeded_service(self, db_path: str) -> TemporalVersionService:
        """Create a service with a parent AST node already inserted."""
        from apex_rag.ingestion.apex_storage import ApexStorage, ASTNodeRow
        db_url = f"sqlite+aiosqlite:///{db_path}"
        storage = await ApexStorage.create(db_url)

        # Insert a parent ASTNode to satisfy FK constraints
        async with storage.session() as session:
            session.add(ASTNodeRow(
                node_id="parent-001",
                content="Test root",
                node_type="PARAGRAPH",
                depth=0,
                doc_id="doc-001",
                tenant_id="default",
                children_json="[]",
                embedding_json="[]",
                ingestion_date=datetime.now(timezone.utc),
            ))

        yield TemporalVersionService(storage)
        await storage.dispose()
        # Cleanup temp file
        try:
            os.unlink(db_path)
        except OSError:
            pass

    @pytest.mark.asyncio
    async def test_create_first_version(self, seeded_service, db_path) -> None:
        """Should create the first version successfully."""
        version = await seeded_service.create_version(
            node_id="parent-001",
            content="Revenue = 100,000",
            doc_id="doc-001",
        )
        assert version is not None
        assert version.version_number == 1
        assert version.is_current is True
        assert version.effective_to is None
        assert version.content_hash != ""

    @pytest.mark.asyncio
    async def test_version_increments(self, seeded_service, db_path) -> None:
        """Should increment version_number for subsequent versions."""
        v1 = await seeded_service.create_version(
            node_id="parent-001", content="V1", doc_id="doc-001",
        )
        assert v1.version_number == 1

        v2 = await seeded_service.create_version(
            node_id="parent-001", content="V2", doc_id="doc-001",
        )
        assert v2.version_number == 2

        v3 = await seeded_service.create_version(
            node_id="parent-001", content="V3", doc_id="doc-001",
        )
        assert v3.version_number == 3

    @pytest.mark.asyncio
    async def test_immutable_history(self, seeded_service, db_path) -> None:
        """Verify that historical data is NEVER overwritten."""
        await seeded_service.create_version(
            node_id="parent-001", content='{"revenue": 100000}', doc_id="doc-001",
        )
        await seeded_service.create_version(
            node_id="parent-001", content='{"revenue": 120000}', doc_id="doc-001",
        )
        await seeded_service.create_version(
            node_id="parent-001", content='{"revenue": 150000}', doc_id="doc-001",
        )

        chain = await seeded_service.get_version_chain("parent-001")
        assert len(chain) == 3

        # OLD versions should have is_current=False and effective_to set
        assert chain[0].is_current is False
        assert chain[0].effective_to is not None

        # LATEST version should be current
        assert chain[2].is_current is True
        assert chain[2].effective_to is None

    @pytest.mark.asyncio
    async def test_get_latest_version(self, seeded_service, db_path) -> None:
        """Should return the latest version."""
        await seeded_service.create_version(
            node_id="parent-001", content="V1", doc_id="doc-001",
        )
        await seeded_service.create_version(
            node_id="parent-001", content="V2", doc_id="doc-001",
        )

        latest = await seeded_service.get_latest_version("parent-001")
        assert latest is not None
        assert latest.version_number == 2
        assert latest.is_current is True

    @pytest.mark.asyncio
    async def test_content_hash_integrity(self, seeded_service, db_path) -> None:
        """Content hashes should match their content."""
        v1 = await seeded_service.create_version(
            node_id="parent-001", content="Revenue = 100,000", doc_id="doc-001",
        )

        # Verify with static method
        assert await TemporalVersionService.verify_content_integrity(v1)

        # Manipulate content -> should fail integrity
        v1.content = "Tampered content"
        assert not await TemporalVersionService.verify_content_integrity(v1)

    @pytest.mark.asyncio
    async def test_latest_after_cleanup(self, seeded_service, db_path) -> None:
        """Cleanup temp file after each test."""
        # Test is done, cleanup happens in fixture teardown
        pass


@pytest.mark.asyncio
async def test_no_overwrite_guarantee_with_cleanup():
    """Standalone test: ensure no temp files leak."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        from apex_rag.ingestion.apex_storage import ApexStorage, ASTNodeRow
        db_url = f"sqlite+aiosqlite:///{tmp.name}"
        storage = await ApexStorage.create(db_url)

        async with storage.session() as session:
            session.add(ASTNodeRow(
                node_id="test-node",
                content="Test",
                node_type="PARAGRAPH",
                depth=0,
                doc_id="doc-test",
                tenant_id="default",
                children_json="[]",
                embedding_json="[]",
                ingestion_date=datetime.now(timezone.utc),
            ))

        service = TemporalVersionService(storage)

        # Create 3 versions
        v1 = await service.create_version(
            node_id="test-node", content='{"val": 1}', doc_id="doc-test",
        )
        v2 = await service.create_version(
            node_id="test-node", content='{"val": 2}', doc_id="doc-test",
        )
        v3 = await service.create_version(
            node_id="test-node", content='{"val": 3}', doc_id="doc-test",
        )

        chain = await service.get_version_chain("test-node")
        assert len(chain) == 3
        assert chain[0].is_current is False
        assert chain[2].is_current is True

        # V1 content preserved
        assert '{"val": 1}' in chain[0].content
        assert '{"val": 3}' in chain[2].content

        await storage.dispose()
    finally:
        os.unlink(tmp.name)
