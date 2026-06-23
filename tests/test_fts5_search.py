"""
tests/test_fts5_search.py — Tests for the FTS5 search (Phase 10).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone

import pytest

from apex_rag.retrieval.search.fts5 import FTS5Search


class TestFTS5Sanitise:
    """Tests for FTS5 query sanitisation."""

    def test_single_word(self) -> None:
        result = FTS5Search._sanitise_query("revenue")
        assert result == '"revenue"*'

    def test_multi_word(self) -> None:
        result = FTS5Search._sanitise_query("revenue growth")
        assert result == '"revenue growth"'

    def test_empty_query(self) -> None:
        result = FTS5Search._sanitise_query("")
        assert result == ""

    def test_preserves_operators(self) -> None:
        result = FTS5Search._sanitise_query("revenue AND growth")
        assert "AND" in result

    def test_preserves_phrase(self) -> None:
        result = FTS5Search._sanitise_query('"Q3 revenue"')
        assert '"Q3 revenue"' in result

    def test_preserves_prefix(self) -> None:
        result = FTS5Search._sanitise_query("rev*")
        assert "rev*" in result

    def test_preserves_boolean(self) -> None:
        result = FTS5Search._sanitise_query("revenue OR profit")
        assert result == "revenue OR profit"


class TestFTS5SearchIntegration:
    """Integration tests for FTS5Search (requires SQLite)."""

    @pytest.fixture
    async def storage(self):
        from apex_rag.ingestion.apex_storage import ApexStorage
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_url = f"sqlite+aiosqlite:///{tmp.name}"
        storage = await ApexStorage.create(db_url)
        yield storage
        await storage.dispose()
        os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_ensure_index_creates_table(self, storage) -> None:
        """Should create the FTS5 virtual table."""
        fts = FTS5Search(storage)
        await fts.ensure_index()  # Should not raise

    @pytest.mark.asyncio
    async def test_rebuild_empty_database(self, storage) -> None:
        """Should handle empty databases gracefully."""
        fts = FTS5Search(storage)
        await fts.ensure_index()
        count = await fts.rebuild_index()
        assert count == 0  # No nodes to index

    @pytest.mark.asyncio
    async def test_search_empty(self, storage) -> None:
        """Should return empty results when no data."""
        fts = FTS5Search(storage)
        await fts.ensure_index()
        results = await fts.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_index_and_search_node(self, storage) -> None:
        """Should index a node and find it via FTS5 search."""
        from apex_rag.ingestion.apex_storage import ASTNodeRow

        fts = FTS5Search(storage)
        await fts.ensure_index()

        # Insert a node
        async with storage.session() as session:
            node = ASTNodeRow(
                node_id=str(uuid.uuid4()),
                content="Q3 revenue increased by 15% to $2.5 million",
                node_type="PARAGRAPH",
                depth=1,
                doc_id="doc-123",
                tenant_id="default",
                children_json="[]",
                embedding_json="[]",
                ingestion_date=datetime.now(timezone.utc),
            )
            session.add(node)

        # Rebuild index
        count = await fts.rebuild_index()
        assert count == 1

        # Search
        results = await fts.search("revenue")
        assert len(results) >= 1
        assert "revenue" in results[0]["content"].lower()
