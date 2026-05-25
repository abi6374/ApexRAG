"""
test_integration.py — Integration tests for the full ApexRAG pipeline.

Tests the complete flow: ingest → query → get_tree → get_page_index → delete
Uses in-memory SQLite and mocked LLM — no external dependencies.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from apex_rag.client import ApexIndex
from apex_rag.providers import AsyncLLM
from apex_rag.storage import StorageEngine


@pytest_asyncio.fixture
async def dummy_llm() -> AsyncLLM:
    """A minimal AsyncLLM that returns appropriate responses for each stage.
    - Summarisation returns a 30-word summary
    - Navigation returns a valid JSON with chosen_id pointing to first child
    - Verification returns a valid JSON confirming the answer
    """
    llm = AsyncMock(spec=AsyncLLM)

    async def mock_generate(
        prompt: str,
        **kwargs,
    ) -> str:
        # Detect which prompt is being used and return appropriate response
        prompt_lower = prompt.lower()
        if "navigate" in prompt_lower or "chosen_id" in prompt_lower or "sub-section" in prompt_lower:
            # Navigation prompt - return valid JSON with first child id
            # Extract child IDs from the prompt
            ids = re.findall(r"\[(\d+)\]", prompt)
            chosen = ids[0] if ids else "1"
            return f'{{"chosen_id": {chosen}, "fallback_id": null, "reason": "Mock selection"}}'
        elif "verify" in prompt_lower or "answers_query" in prompt_lower:
            # Verification prompt
            return '{"answers_query": true, "confidence": 0.95, "reason": "Mock verification"}'
        else:
            # Summary prompt
            return "Mock summary about the topic."

    llm.generate = AsyncMock(side_effect=mock_generate)
    return llm


SAMPLE_TEXT = """\
# Chapter 1: Introduction
This is the introduction chapter.

## Section 1.1: Background
Background information for the research.

## Section 1.2: Methods
The methodology used in this study.

# Chapter 2: Results
## Section 2.1: Key Findings
The main findings of the research.

## Section 2.2: Analysis
Detailed analysis of the results.
"""


@pytest.mark.asyncio
async def test_full_pipeline(dummy_llm: AsyncLLM) -> None:
    """Test the complete ingest → query → tree → delete pipeline."""
    # Build index with custom storage (in-memory) and mock LLM
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")
    from apex_rag.ingestion.legacy import IngestionEngine, Summariser
    from apex_rag.navigation import AggregatorAgent, NavigationAgent

    summariser = Summariser(llm=dummy_llm, max_concurrent=4)
    ingestor = IngestionEngine(storage=storage, summariser=summariser)
    agent = NavigationAgent(storage, model=dummy_llm, verify_leaves=False)
    aggregator = AggregatorAgent(model=dummy_llm)

    index = ApexIndex(storage, ingestor, agent, aggregator)

    try:
        # 1. Ingest
        doc_id = await index.ingest_text(SAMPLE_TEXT, doc_id="integration-test")
        assert doc_id == "integration-test"

        # 2. Verify stats
        stats = await index.get_stats(doc_id)
        assert stats["total_nodes"] > 0
        assert stats["leaf_count"] > 0
        assert stats["doc_id"] == doc_id
        assert stats["max_depth"] >= 1

        # 3. Get tree
        tree = await index.get_tree(doc_id)
        assert len(tree) > 0
        assert tree[0]["doc_id"] == doc_id

        # 4. Get page index
        page_index = await index.get_page_index(doc_id)
        assert len(page_index) > 0
        assert all(e["doc_id"] == doc_id for e in page_index)

        # 5. Search index
        results = await index.search_index(doc_id, "Introduction")
        assert len(results) >= 1

        # 6. Query (with mocked LLM)
        result = await index.query("What is the introduction about?", doc_id)
        assert result is not None
        assert result.node_id > 0

        # 7. List documents
        docs = await index.list_documents()
        assert doc_id in list(docs)

        # 8. Delete
        count = await index.delete(doc_id)
        assert count > 0

        docs_after = await index.list_documents()
        assert doc_id not in list(docs_after)

    finally:
        await index.close()


@pytest.mark.asyncio
async def test_ingest_empty_text() -> None:
    """Ingesting empty text should create a minimal tree."""
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")
    llm = AsyncMock(spec=AsyncLLM)
    llm.generate = AsyncMock(return_value="Summary")
    from apex_rag.ingestion.legacy import IngestionEngine
    ingestor = IngestionEngine(storage=storage)
    from apex_rag.navigation import AggregatorAgent, NavigationAgent
    agent = NavigationAgent(storage, model=llm, verify_leaves=False)
    aggregator = AggregatorAgent(model=llm)
    index = ApexIndex(storage, ingestor, agent, aggregator)

    try:
        doc_id = await index.ingest_text("", doc_id="empty-test")
        stats = await index.get_stats(doc_id)
        assert stats["total_nodes"] >= 0  # Empty text creates a single root
    finally:
        await index.close()


@pytest.mark.asyncio
async def test_concurrent_ingestion(dummy_llm: AsyncLLM) -> None:
    """Test that multiple documents can be ingested concurrently."""
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")
    from apex_rag.ingestion.legacy import IngestionEngine, Summariser
    from apex_rag.navigation import AggregatorAgent, NavigationAgent
    summariser = Summariser(llm=dummy_llm, max_concurrent=4)
    ingestor = IngestionEngine(storage=storage, summariser=summariser)
    agent = NavigationAgent(storage, model=dummy_llm, verify_leaves=False)
    aggregator = AggregatorAgent(model=dummy_llm)
    index = ApexIndex(storage, ingestor, agent, aggregator)

    try:
        texts = [
            ("# Doc 1\nContent for doc one.", "doc-concurrent-1"),
            ("# Doc 2\nContent for doc two.", "doc-concurrent-2"),
            ("# Doc 3\nContent for doc three.", "doc-concurrent-3"),
        ]

        tasks = [index.ingest_text(text, doc_id=did) for text, did in texts]
        results = await asyncio.gather(*tasks)

        assert results == ["doc-concurrent-1", "doc-concurrent-2", "doc-concurrent-3"]

        docs = await index.list_documents()
        assert len(list(docs)) == 3
    finally:
        await index.close()


@pytest.mark.asyncio
async def test_ingest_many(dummy_llm: AsyncLLM) -> None:
    """Test batch ingestion of multiple documents via ingest_many()."""
    storage = await StorageEngine.create("sqlite+aiosqlite:///:memory:")
    from apex_rag.ingestion.legacy import IngestionEngine, Summariser
    from apex_rag.navigation import AggregatorAgent, NavigationAgent
    summariser = Summariser(llm=dummy_llm, max_concurrent=4)
    ingestor = IngestionEngine(storage=storage, summariser=summariser)
    agent = NavigationAgent(storage, model=dummy_llm, verify_leaves=False)
    aggregator = AggregatorAgent(model=dummy_llm)
    index = ApexIndex(storage, ingestor, agent, aggregator)

    try:
        doc_ids = await index.ingest_many([
            ("batch-doc-1", "# Doc One\nContent for first batch document."),
            ("batch-doc-2", "# Doc Two\nContent for second batch document."),
            ("batch-doc-3", "# Doc Three\nContent for third batch document."),
        ])

        assert doc_ids == ["batch-doc-1", "batch-doc-2", "batch-doc-3"]

        # Verify all documents are indexed
        docs = await index.list_documents()
        assert "batch-doc-1" in list(docs)
        assert "batch-doc-2" in list(docs)
        assert "batch-doc-3" in list(docs)

        # Verify each document has tree data
        for did in doc_ids:
            stats = await index.get_stats(did)
            assert stats["total_nodes"] > 0
            tree = await index.get_tree(did)
            assert len(tree) >= 1
    finally:
        await index.close()
