#!/usr/bin/env python3
"""
run_dag_latency_benchmark.py — Real latency benchmark: eager vs adaptive DAG modes.

Uses time.perf_counter() to measure p50/p95 ingest latency for both modes,
reusing the same document structure and mock LLM from the existing test fixtures.

Usage:
    python benchmarks/run_dag_latency_benchmark.py [--iterations N] [--sections N]

Output:
    Prints p50/p95 latency table and saves to benchmarks/dag_latency_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from apex_rag.client import ApexIndex
from apex_rag.providers import AsyncLLM


def _make_mock_llm() -> MagicMock:
    """Replicate the mock_llm fixture from tests/test_benchmarks.py."""
    llm = MagicMock(spec=AsyncLLM)

    async def mock_generate(prompt: str, **_kwargs: Any) -> str:
        prompt_lower = prompt.lower()
        if "decomposition" in prompt_lower or "plan" in prompt_lower:
            return '{"sub_queries": ["Test sub-query"]}'
        if "navigate" in prompt_lower or "chosen_id" in prompt_lower or "sub-section" in prompt_lower:
            import re
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

    async def mock_embed(texts: list[str], **_kwargs: Any) -> list[list[float]]:
        import random
        return [[random.uniform(-1.0, 1.0) for _ in range(384)] for _ in texts]

    llm.embed = AsyncMock(side_effect=mock_embed)

    async def _stream(*_args: Any, **_kwargs: Any):
        yield "Mocked "
        yield "response"

    llm.stream_generate = _stream
    return llm


def build_document(sections: int = 10) -> str:
    """Build a markdown document like the test benchmark fixture."""
    md_lines = ["# Benchmark Document"]
    for i in range(sections):
        md_lines.append(f"\n## Section {i}")
        md_lines.append(f"\nContent for section {i}. " * 20)
        for j in range(2):
            md_lines.append(f"\n### Subsection {i}.{j}")
            md_lines.append(f"\nDetailed content for subsection {i}.{j}. " * 15)
    return "\n".join(md_lines)


async def run_single_ingest(
    mode: str,
    doc_text: str,
    doc_id: str,
    mock_llm: MagicMock,
) -> float:
    """Run a single ingest in the given mode and return elapsed seconds."""
    index = await ApexIndex.create(
        db_url="sqlite+aiosqlite:///:memory:",
        provider=mock_llm,
        trace_enabled=False,
        graph_construction_mode=mode,
    )

    t0 = time.perf_counter()
    result_id = await index.ingest_text(doc_text, doc_id=doc_id)
    elapsed = time.perf_counter() - t0

    assert result_id == doc_id
    return elapsed


def compute_percentiles(times: list[float]) -> dict[str, float]:
    """Compute p50, p95, min, max, mean from a sorted list of times."""
    sorted_t = sorted(times)
    n = len(sorted_t)
    p50 = sorted_t[int(n * 0.50)]
    p95 = sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1]
    return {
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "min": round(min(times), 4),
        "max": round(max(times), 4),
        "mean": round(statistics.mean(times), 4),
        "stdev": round(statistics.stdev(times), 4) if n > 1 else 0.0,
    }


async def main(iterations: int = 10, sections: int = 10) -> None:
    print(f"{'='*70}")
    print(f"  DAG Latency Benchmark: eager vs adaptive")
    print(f"  Iterations: {iterations} | Sections: {sections}")
    print(f"  (Warmup run excluded from results)")
    print(f"{'='*70}")

    doc_text = build_document(sections)

    # Create mock LLM once (stateless) to reduce GC noise
    llm = _make_mock_llm()

    # --- Warmup: one run per mode to eliminate cold-start bias ---
    print("  Warming up...")
    await run_single_ingest("eager", doc_text, "warmup-eager", llm)
    await run_single_ingest("adaptive", doc_text, "warmup-adaptive", llm)

    eager_times: list[float] = []
    adaptive_times: list[float] = []

    for i in range(iterations):
        # --- Eager mode ---
        t = await run_single_ingest("eager", doc_text, f"eager-doc-{i:03d}", llm)
        eager_times.append(t)
        print(f"  [{i+1:3d}/{iterations}] eager:    {t:.4f}s")

        # --- Adaptive mode ---
        t = await run_single_ingest("adaptive", doc_text, f"adaptive-doc-{i:03d}", llm)
        adaptive_times.append(t)
        print(f"  [{i+1:3d}/{iterations}] adaptive: {t:.4f}s")

    eager_stats = compute_percentiles(eager_times)
    adaptive_stats = compute_percentiles(adaptive_times)

    # Speedup
    p50_speedup = eager_stats["p50"] / adaptive_stats["p50"]
    p95_speedup = eager_stats["p95"] / adaptive_stats["p95"]
    mean_speedup = eager_stats["mean"] / adaptive_stats["mean"]

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  {'Metric':>12} | {'Eager':>10} | {'Adaptive':>10} | {'Speedup':>8}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'p50':>12} | {eager_stats['p50']:>8.4f}s | {adaptive_stats['p50']:>8.4f}s | {p50_speedup:>6.2f}x")
    print(f"  {'p95':>12} | {eager_stats['p95']:>8.4f}s | {adaptive_stats['p95']:>8.4f}s | {p95_speedup:>6.2f}x")
    print(f"  {'mean':>12} | {eager_stats['mean']:>8.4f}s | {adaptive_stats['mean']:>8.4f}s | {mean_speedup:>6.2f}x")
    print(f"  {'min':>12} | {eager_stats['min']:>8.4f}s | {adaptive_stats['min']:>8.4f}s |")
    print(f"  {'max':>12} | {eager_stats['max']:>8.4f}s | {adaptive_stats['max']:>8.4f}s |")
    print(f"  {'stdev':>12} | {eager_stats['stdev']:>8.4f}s | {adaptive_stats['stdev']:>8.4f}s |")
    print(f"{'='*70}")

    # Build comparison dict
    results = {
        "methodology": {
            "iterations": iterations,
            "sections": sections,
            "db_url": "sqlite+aiosqlite:///:memory:",
            "llm": "MockLLM (no real API calls)",
            "timer": "time.perf_counter()",
            "modes": ["eager", "adaptive"],
        },
        "eager": eager_stats,
        "adaptive": adaptive_stats,
        "speedup": {
            "p50": round(p50_speedup, 2),
            "p95": round(p95_speedup, 2),
            "mean": round(mean_speedup, 2),
        },
        "all_times": {
            "eager": [round(t, 4) for t in eager_times],
            "adaptive": [round(t, 4) for t in adaptive_times],
        },
    }

    out_path = "benchmarks/dag_latency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [✓] Full results saved to {out_path}")

    # One-paragraph summary for the benchmark doc
    print(f"\n  SUMMARY")
    print(f"  Adaptive mode cuts ingest p50 by ~{p50_speedup:.1f}x and p95 by ~{p95_speedup:.1f}x")
    print(f"  by building only DocumentDAG eagerly vs all DAGs in eager mode.")
    print(f"  (Measured with mock LLM — real LLM overhead would amplify the gap.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAG latency benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Number of ingest iterations per mode")
    parser.add_argument("--sections", type=int, default=10, help="Number of document sections per ingest")
    args = parser.parse_args()
    asyncio.run(main(iterations=args.iterations, sections=args.sections))
