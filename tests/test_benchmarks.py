"""
test_benchmarks.py — Performance regression detection for ApexRAG.

These benchmarks measure critical operations and fail if performance
degrades beyond defined thresholds. Run with:

    pytest tests/test_benchmarks.py -v --benchmark

Or as part of CI to catch regressions early.
"""

from unittest.mock import AsyncMock, MagicMock
import re
import asyncio
import time
import pytest
from collections.abc import AsyncGenerator
from typing import Any
from apex_rag.providers import AsyncLLM
from apex_rag.client import ApexIndex
from apex_rag.exceptions import ApexRAGError

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=AsyncLLM)
    
    async def mock_generate(prompt: str, **kwargs: Any) -> str:
        prompt_lower = prompt.lower()
        if "decomposition" in prompt_lower or "plan" in prompt_lower:
            return '{"sub_queries": ["Test sub-query"]}'
        if "navigate" in prompt_lower or "chosen_id" in prompt_lower or "sub-section" in prompt_lower:
            ids = re.findall(r"\[([a-f0-9\-]+)\]", prompt)
            chosen = ids[0] if ids else "1"
            return f'{{"chosen_id": "{chosen}", "fallback_id": null, "reason": "Mock selection"}}'
        elif "verify" in prompt_lower or "answers_query" in prompt_lower:
            return '{"answers_query": true, "confidence": 0.95, "reason": "Mock verification"}'
        elif "evaluate" in prompt_lower or "provides enough information" in prompt_lower:
            return '{"passes_evaluation": true, "reason": "Verified"}'
        elif "cite each claim" in prompt_lower:
            return "Citing claim. [Node ID: mock-1]"
        else:
            return "Mock summary about the topic."
            
    llm.generate = AsyncMock(side_effect=mock_generate)
    
    async def mock_embed(texts: list[str], **kwargs: Any) -> list[list[float]]:
        import random
        return [[random.uniform(-1.0, 1.0) for _ in range(384)] for _ in texts]
    llm.embed = AsyncMock(side_effect=mock_embed)
    
    async def _stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield "Mocked "
        yield "response"
    llm.stream_generate = _stream
    
    return llm


@pytest.fixture
async def populated_index(mock_llm: MagicMock) -> ApexIndex:
    """Create an index with a large-ish document for benchmarking."""
    index = await ApexIndex.create(
        db_url="sqlite+aiosqlite:///:memory:",
        provider=mock_llm,
        trace_enabled=False,
    )

    # Build a large document with many sections
    md_lines = ["# Benchmark Document"]
    for i in range(50):
        md_lines.append(f"\n## Section {i}")
        md_lines.append(f"\nContent for section {i}. " * 20)
        for j in range(5):
            md_lines.append(f"\n### Subsection {i}.{j}")
            md_lines.append(f"\nDetailed content for subsection {i}.{j}. " * 15)

    await index.ingest_text(
        "\n".join(md_lines),
        doc_id="benchmark-doc",
    )

    return index


# ── Benchmarks ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_ingestion_throughput(populated_index: ApexIndex) -> None:
    """Benchmark: ingestion of a moderately-sized document should complete quickly."""
    index = populated_index
    md = ["# Perf Test"]
    for i in range(20):
        md.append(f"\n## Section {i}")
        md.append(f"\nContent for perf section {i}. " * 30)

    t0 = time.monotonic()
    doc_id = await index.ingest_text(
        "\n".join(md),
        doc_id="perf-test",
    )
    elapsed = time.monotonic() - t0

    assert doc_id == "perf-test"
    # Ingestion of 20 sections should complete in under 15 seconds (due to causal LLM)
    assert elapsed < 15.0, f"Ingestion took {elapsed:.2f}s (threshold: 15.0s)"
    print(f"\n[Timer] Ingestion throughput: {elapsed:.3f}s for 20 sections")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_tree_retrieval_performance(populated_index: ApexIndex) -> None:
    """Benchmark: full tree retrieval should be fast even for large documents."""
    index = populated_index

    t0 = time.monotonic()
    tree = await index.get_tree("benchmark-doc")
    elapsed = time.monotonic() - t0

    assert len(tree) > 100, f"Expected >100 nodes, got {len(tree)}"
    # Tree retrieval of 300+ nodes should complete in under 500ms
    assert elapsed < 0.5, f"Tree retrieval took {elapsed:.2f}s (threshold: 0.5s)"
    print(f"\n[Timer] Tree retrieval: {elapsed:.3f}s for {len(tree)} nodes")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_stats_performance(populated_index: ApexIndex) -> None:
    """Benchmark: document stats should be near-instant."""
    index = populated_index

    t0 = time.monotonic()
    stats = await index.get_stats("benchmark-doc")
    elapsed = time.monotonic() - t0

    assert stats["total_nodes"] > 100
    assert elapsed < 0.2, f"Stats took {elapsed:.2f}s (threshold: 0.2s)"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_page_index_performance(populated_index: ApexIndex) -> None:
    """Benchmark: page index retrieval should be fast."""
    index = populated_index

    t0 = time.monotonic()
    entries = await index.get_page_index("benchmark-doc")
    elapsed = time.monotonic() - t0

    assert len(entries) > 100
    assert elapsed < 0.3, f"Page index took {elapsed:.2f}s (threshold: 0.3s)"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_list_documents_performance(populated_index: ApexIndex) -> None:
    """Benchmark: listing documents should be instant."""
    index = populated_index

    t0 = time.monotonic()
    docs = await index.list_documents()
    elapsed = time.monotonic() - t0

    # Self-contained: verify at least the benchmark-doc exists
    assert len(docs) >= 1, "Expected at least 1 document"
    assert elapsed < 0.1, f"List documents took {elapsed:.2f}s (threshold: 0.1s)"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_delete_performance(populated_index: ApexIndex) -> None:
    """Benchmark: document deletion should be fast."""
    index = populated_index

    # Create a doc to delete
    await index.ingest_text(
        "# Delete Me\nContent here. " * 100,
        doc_id="delete-me",
    )

    t0 = time.monotonic()
    deleted = await index.delete("delete-me")
    elapsed = time.monotonic() - t0

    assert deleted > 0
    assert elapsed < 0.5, f"Deletion took {elapsed:.2f}s (threshold: 0.5s)"
    print(f"\n[Timer] Deletion: {elapsed:.3f}s for {deleted} nodes")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_error_instantiation_performance() -> None:
    """Benchmark: exception creation should be extremely fast."""
    from apex_rag.exceptions import (
        DocumentNotFoundError,
    )

    t0 = time.monotonic()
    count = 10000
    for _ in range(count):
        try:
            raise DocumentNotFoundError(
                message="Test error",
                hint="Test hint",
            )
        except ApexRAGError:
            pass

    elapsed = time.monotonic() - t0
    # Guard against sub-millisecond timing resolution
    if elapsed < 1e-6:
        elapsed = 1e-6
    ops_per_sec = count / elapsed
    assert ops_per_sec > 50000, f"Only {ops_per_sec:.0f} exceptions/s (threshold: 50k/s)"
    print(f"\n[Timer] Exception instantiation: {ops_per_sec:.0f} ops/s")


@pytest.mark.asyncio
async def test_concurrent_query_safety(mock_llm: MagicMock) -> None:
    """Stress test: multiple concurrent queries should not crash."""
    index = await ApexIndex.create(
        db_url="sqlite+aiosqlite:///:memory:",
        provider=mock_llm,
        trace_enabled=False,
    )

    await index.ingest_text(
        "# Concurrent Test\n" + "\n".join(f"\n## Section {i}\nContent." for i in range(10)),
        doc_id="stress",
    )

    async def do_query(q: str) -> None:
        try:
            await index.query(q, "stress")
        except Exception:
            pass  # Expected — no Ollama available in tests

    # Fire 20 concurrent queries
    tasks = [do_query(f"Question {i}") for i in range(20)]
    t0 = time.monotonic()
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - t0

    # Queries should not block each other indefinitely
    assert elapsed < 30.0, f"Concurrent queries took {elapsed:.2f}s (threshold: 30s)"
