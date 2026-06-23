import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from apex_rag.ingestion.apex_storage import ApexStorage, NodeVersionRow, StateSnapshotRow, TimelineEventRow, ChangeHistoryRow
from apex_rag.temporal.temporal_retriever import TemporalRetriever
from apex_rag.temporal.state_reconstructor import StateReconstructor
from apex_rag.temporal.analyzers import ChangeAnalyzer, TrendAnalyzer
from apex_rag.temporal.temporal_agent import TemporalReasoningAgent

@pytest.mark.asyncio
class TestTemporalIntelligence:

    @pytest.fixture
    async def storage(self) -> ApexStorage:
        # Create SQLite in-memory database for testing
        return await ApexStorage.create("sqlite+aiosqlite:///:memory:")

    async def test_temporal_retriever_query_modes(self, storage: ApexStorage) -> None:
        retriever = TemporalRetriever(storage)
        
        # Seed test node versions
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 10, tzinfo=timezone.utc)
        
        node_uuid = "4552c447-004f-420e-94cd-e55fc4bcf799"
        row1 = NodeVersionRow(
            version_id="v1",
            node_id=node_uuid,
            content="Stock = 500",
            effective_from=t1,
            effective_to=t2,
            version_number=1,
            is_current=False,
            doc_id="doc-x",
            tenant_id="tenant-1"
        )
        row2 = NodeVersionRow(
            version_id="v2",
            node_id=node_uuid,
            content="Stock = 350",
            effective_from=t2,
            effective_to=None,
            version_number=2,
            is_current=True,
            doc_id="doc-x",
            tenant_id="tenant-1"
        )
        
        async with storage.session() as session:
            session.add(row1)
            session.add(row2)

        # Test LATEST
        latest = await retriever.get_latest_nodes("doc-x")
        assert len(latest) == 1
        assert latest[0].content == "Stock = 350"

        # Test AS_OF_DATE (t1 + 2 days)
        as_of = await retriever.get_nodes_as_of("doc-x", t1 + timedelta(days=2))
        assert len(as_of) == 1
        assert as_of[0].content == "Stock = 500"

        # Test history
        hist = await retriever.get_node_history(node_uuid)
        assert len(hist) == 2
        assert hist[0].content == "Stock = 500"
        assert hist[1].content == "Stock = 350"

    async def test_state_reconstructor(self, storage: ApexStorage) -> None:
        reconstructor = StateReconstructor(storage)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        
        # Seed state snapshot
        snap = StateSnapshotRow(
            snapshot_id="snap1",
            doc_id="metrics-doc",
            snapshot_date=t1,
            snapshot_data='{"Revenue": 100000.0, "Stock": 500.0}'
        )
        async with storage.session() as session:
            session.add(snap)

        metrics = await reconstructor.reconstruct_metrics("metrics-doc", t1)
        assert metrics["Revenue"] == 100000.0
        assert metrics["Stock"] == 500.0

    async def test_change_analyzer(self) -> None:
        analyzer = ChangeAnalyzer()
        
        # Test metric comparison
        comp = analyzer.compare_metrics(100000.0, 120000.0)
        assert comp["difference"] == 20000.0
        assert comp["percentage_change"] == 20.0
        assert comp["direction"] == "increase"

        # Test version text comparison
        diff = analyzer.compare_versions("Line 1\nLine 2", "Line 1\nLine 3")
        assert "Line 3" in diff["added_lines"]
        assert "Line 2" in diff["removed_lines"]
        assert diff["changes_count"] == 2

    async def test_trend_analyzer(self) -> None:
        analyzer = TrendAnalyzer()
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        t3 = datetime(2025, 1, 3, tzinfo=timezone.utc)
        
        points = [
            (t1, 1000.0),
            (t2, 1200.0),
            (t3, 1500.0)
        ]
        
        res = analyzer.analyze_trend(points)
        assert res["direction"] == "growth"
        assert res["percentage_change"] == 50.0
        assert len(res["moving_averages"]) == 3

    async def test_temporal_reasoning_agent(self, storage: ApexStorage) -> None:
        retriever = TemporalRetriever(storage)
        reconstructor = StateReconstructor(storage)
        agent = TemporalReasoningAgent(retriever, reconstructor)

        # Test detection
        assert agent.detect_time_query("What was revenue on 2025-01-10?") is True
        assert agent.detect_time_query("Show sales trend from January to March") is True
        assert agent.detect_time_query("Normal query without date context") is False

        # Test parsing dates
        dates = agent.parse_dates("What was revenue on 2025-01-10?")
        assert len(dates) == 1
        assert dates[0].year == 2025
        assert dates[0].month == 1
        assert dates[0].day == 10
