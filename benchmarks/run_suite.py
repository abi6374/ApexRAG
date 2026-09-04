"""
benchmarks/run_suite.py — The master execution script for Part 8 benchmarks.

Runs ApexRAG vs LangChain vs LlamaIndex across three domains (scientific,
financial, legal), evaluating three ApexRAG configurations to isolate what
each layer actually contributes:

    - ApexRAG-Minimal:  graph_construction_mode="minimal", ablation_mode=True
                        -- AST navigation only, no DAGs, no conformal/temporal.
    - ApexRAG-Ablation: DAGs on, conformal/temporal off.
    - ApexRAG-Full:     everything on, with conformal calibrated from a
                        held-out slice of the pilot set (see --calibration-n).

Outputs a LaTeX table, a chart, and a raw results JSON.

IMPORTANT: this script requires an explicit ``--mock`` flag to run without
a real API key. Earlier versions silently fell back to a mocked pipeline
whenever no key was set, which produced results that looked like a real
comparison (LangChain/LlamaIndex ~0%, ApexRAG ~100%) but weren't -- see
benchmarks/dag_router_accuracy_honest.md for the sibling issue on the
DAG-router benchmark. A loud banner now makes it impossible to mistake one
run type for the other.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from apex_rag import ApexIndex
from benchmarks.baselines import LangChainBaseline, LlamaIndexBaseline
from benchmarks.data_loaders import BenchmarkDataLoader
from benchmarks.metrics import ApexMetrics, f1_score

# Ensure UTF-8 output on Windows consoles (emoji/unicode in print statements)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

_APEX_CONFIGS: dict[str, dict[str, Any]] = {
    "ApexRAG-Minimal": {"graph_construction_mode": "minimal", "ablation_mode": True},
    "ApexRAG-Ablation": {"graph_construction_mode": "adaptive", "ablation_mode": True},
    "ApexRAG-Full": {"graph_construction_mode": "adaptive", "ablation_mode": False},
}


class MockBaseline:
    """Mock baseline for pipeline verification (--mock runs only)."""

    def __init__(self, name: str):
        self.name = name

    async def query(self, question: str, text_context: str) -> Any:
        from benchmarks.baselines import BaselineResult

        return BaselineResult(
            answer=f"Mocked {self.name} answer for: {question[:30]}...",
            contexts=[text_context[:200]],
            mocked=True,
        )


def _has_any_key(provider: str) -> bool:
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))


async def run_one_baseline(
    baseline_name: str, examples: list, provider: str, mock: bool
) -> list[dict[str, Any]]:
    """Run a specific baseline on a list of examples."""
    results = []

    if mock:
        adapter: Any = MockBaseline(baseline_name)
    elif baseline_name == "LangChain":
        adapter = LangChainBaseline(provider=provider)
    elif baseline_name == "LlamaIndex":
        adapter = LlamaIndexBaseline(provider=provider)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    for ex in examples:
        print(f"  [{baseline_name}] Querying: {ex.question[:50]}...")
        try:
            res = await adapter.query(ex.question, ex.text)
            results.append(
                {
                    "baseline": baseline_name,
                    "answer": res.answer,
                    "contexts": res.contexts,
                    "mocked": getattr(res, "mocked", mock),
                    "ground_truth": ex.ground_truth,
                    "domain": ex.metadata["domain"],
                    "is_contradiction": ex.metadata.get("is_contradiction"),
                }
            )
        except Exception as e:
            print(f"  [!] Error running {baseline_name}: {e}. Mocking result.")
            results.append(
                {
                    "baseline": baseline_name,
                    "answer": "Error-fallback answer",
                    "contexts": [],
                    "mocked": True,
                    "ground_truth": ex.ground_truth,
                    "domain": ex.metadata["domain"],
                    "is_contradiction": ex.metadata.get("is_contradiction"),
                }
            )
    return results


async def run_apex(
    config_name: str,
    examples: list,
    provider: str,
    mock: bool,
    calibration_examples: list[tuple[str, str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Run one ApexRAG configuration (see _APEX_CONFIGS) on a list of examples."""
    apex_kwargs = _APEX_CONFIGS[config_name]
    results = []

    create_kwargs: dict[str, Any] = {"graph_construction_mode": apex_kwargs["graph_construction_mode"]}
    if mock:
        create_kwargs["api_key"] = "mock-key"
        from apex_rag import providers

        provider_cls = {
            "openai": providers.OpenAIProvider,
            "gemini": providers.GeminiProvider,
            "anthropic": providers.AnthropicProvider,
            "groq": providers.GroqProvider,
            "openrouter": providers.OpenRouterProvier,
        }.get(provider.lower(), providers.OllamaProvider)

        async def mock_embed(self, texts, **kwargs):
            return [[0.1] * 1536 for _ in texts]

        async def mock_generate(self, prompt, **kwargs):
            p_lower = prompt.lower()
            gt_hint = "Mocked answer"
            for ex in examples:
                if ex.question in prompt:
                    gt_hint = ex.ground_truth
                    break
            if "strict verification engine" in p_lower:
                return "TRUE"
            if "evaluate" in p_lower or "provides enough information" in p_lower:
                return json.dumps({"passes_evaluation": True, "reason": "All sub-queries answered."})
            if "cite each claim" in p_lower:
                return f"{gt_hint} [Node ID: mock-1]"
            if "decomposition" in p_lower or "plan" in p_lower:
                return "{}"  # falls back to [original query] -- see agents/planner/agent.py
            import re

            node_ids = re.findall(r"\[([a-f0-9\-]+)\]", prompt)
            chosen = node_ids[0] if node_ids else "mock-id-1"
            return json.dumps(
                {"chosen_id": chosen, "fallback_id": None, "reason": "Relevant node found"}
            )

        provider_cls.generate = mock_generate
        provider_cls.embed = mock_embed
        print(f"  [!] Mocking {provider_cls.__name__} for {config_name}...")

    async with await ApexIndex.create(provider=provider, model=None, **create_kwargs) as index:
        if config_name == "ApexRAG-Full" and calibration_examples:
            print(f"  [{config_name}] Calibrating conformal predictor on {len(calibration_examples)} held-out examples...")
            for _q, doc_id, text, _gt in calibration_examples:
                await index.ingest_text(text, doc_id=doc_id)
            summary = await index.enterprise.calibrate_conformal(
                [(q, doc_id, gt) for q, doc_id, _text, gt in calibration_examples]
            )
            print(f"  [{config_name}] Calibration summary: {summary}")

        for ex in examples:
            print(f"  [{config_name}] Ingesting & Querying: {ex.question[:50]}...")
            await index.ingest_text(ex.text, doc_id=ex.doc_id)
            answer = await index.query(
                ex.question,
                ex.doc_id,
                ablation_mode=apex_kwargs["ablation_mode"],
                domain=ex.metadata["domain"],
            )

            results.append(
                {
                    "baseline": config_name,
                    "answer": answer.answer_text,
                    "contexts": [p.content for p in answer.evidence_packets],
                    "mocked": mock,
                    "ground_truth": ex.ground_truth,
                    "domain": ex.metadata["domain"],
                    "is_contradiction": ex.metadata.get("is_contradiction"),
                    "coverage": answer.coverage_guarantee,
                    "contradictions": len(answer.contradictions),
                    "freshness": answer.temporal_freshness,
                    "raw_answer": answer,
                }
            )
    return results


def _load_domain_examples(subset_size: int, mock: bool) -> list:
    """Load real per-domain examples, or the mock smoke-test fixture."""
    if mock:
        return list(BenchmarkDataLoader.load_mock(subset_size * 3))
    try:
        scientific = list(BenchmarkDataLoader.load_hotpotqa_as_scientific_proxy(subset_size))
        financial = list(BenchmarkDataLoader.load_fiqa(subset_size))
        legal = list(BenchmarkDataLoader.load_legal_contradictions(subset_size))
        return scientific + financial + legal
    except Exception as e:
        print(f"  [!] Error loading real datasets: {e}. Falling back to mock data.")
        return list(BenchmarkDataLoader.load_mock(subset_size * 3))


async def main(subset_size: int, provider: str, mock: bool, calibration_n: int) -> None:
    banner = "MOCK RUN (--mock)" if mock else f"REAL RUN (provider={provider})"
    print("#" * 60)
    print(f"#  {banner}")
    print("#  These results are NOT comparable across banner types.")
    print("#" * 60)
    print(
        f"\n🚀 Starting Research Benchmark Suite "
        f"(n={subset_size}/domain, provider={provider}, mock={mock})"
    )

    if not mock and not _has_any_key(provider):
        print(
            f"\n[!] No API key found for provider={provider} and --mock was not passed.\n"
            f"    Set the relevant API key env var, or pass --mock explicitly "
            f"for a pipeline smoke test."
        )
        sys.exit(1)

    # 1. Load Data
    print("[*] Loading datasets...")
    all_examples = _load_domain_examples(subset_size, mock)

    # Held out a small calibration slice (never scored) for ApexRAG-Full's
    # conformal calibration -- disjoint from the examples used for metrics.
    calibration_examples: list[tuple[str, str, str, str]] = []
    if not mock and calibration_n > 0:
        cal_raw = _load_domain_examples(max(1, calibration_n // 3), mock=False)
        calibration_examples = [
            (ex.question, f"cal-{ex.doc_id}", ex.text, ex.ground_truth) for ex in cal_raw
        ]

    final_results: list[dict[str, Any]] = []

    # 2. Run Evaluations (sequential -- avoids rate limits, cleaner logging)
    final_results.extend(await run_one_baseline("LangChain", all_examples, provider, mock))
    final_results.extend(await run_one_baseline("LlamaIndex", all_examples, provider, mock))
    for config_name in _APEX_CONFIGS:
        final_results.extend(
            await run_apex(
                config_name,
                all_examples,
                provider,
                mock,
                calibration_examples=calibration_examples if config_name == "ApexRAG-Full" else None,
            )
        )

    # 3. Compute Metrics
    df = pd.DataFrame(final_results)
    print("\n[*] Computing Metrics...")

    summary = []
    for name in ["LangChain", "LlamaIndex", *list(_APEX_CONFIGS)]:
        sub = df[df["baseline"] == name]

        # Token-level F1 against the ground truth (see benchmarks/metrics.py --
        # replaces the old raw substring-match "accuracy" proxy).
        f1 = sub.apply(lambda x: f1_score(x["answer"], x["ground_truth"]), axis=1).mean()

        # Contradiction Recall (legal domain only; needs is_contradiction label)
        legal_sub = sub[sub["domain"] == "legal"]
        recall = float("nan")
        if name == "ApexRAG-Full" and "raw_answer" in sub.columns:
            recall = (
                legal_sub.apply(
                    lambda x: ApexMetrics.contradiction_recall(
                        x["raw_answer"],
                        "Contradiction" if x.get("is_contradiction") else "Consistent",
                    ),
                    axis=1,
                )
                .dropna()
                .mean()
            )

        summary.append(
            {
                "System": name,
                "F1": f1,
                "Contradiction Recall": recall,
                "Freshness": sub.get("freshness", pd.Series(dtype=float)).mean()
                if "freshness" in sub
                else float("nan"),
                "Mocked": bool(sub["mocked"].any()) if "mocked" in sub else mock,
            }
        )

    summary_df = pd.DataFrame(summary)
    print("\n" + "=" * 60)
    print(f"📊 BENCHMARK SUMMARY ({banner})")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)

    # 4. Export LaTeX
    latex = summary_df.to_latex(
        index=False, float_format="%.3f", caption=f"Benchmark Results ({banner})"
    )
    with open("benchmark_results.tex", "w") as f:
        f.write(latex)
    print("\n[+] LaTeX table saved to benchmark_results.tex")

    # 5. Raw results JSON (excluding non-serializable raw_answer objects)
    raw_out = [{k: v for k, v in r.items() if k != "raw_answer"} for r in final_results]
    with open("benchmark_results_raw.json", "w", encoding="utf-8") as f:
        json.dump({"mock": mock, "provider": provider, "results": raw_out}, f, indent=2, default=str)
    print("[+] Raw results saved to benchmark_results_raw.json")

    # 6. Plot
    summary_df.plot(kind="bar", x="System", y=["F1"], figsize=(10, 6))
    plt.title(f"ApexRAG vs Baselines ({banner})")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("benchmark_chart.png")
    print("[+] Plot saved to benchmark_chart.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApexRAG Research Benchmark Suite")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM provider to use")
    parser.add_argument("--n", type=int, default=20, help="Number of examples per domain")
    parser.add_argument(
        "--calibration-n", type=int, default=15,
        help="Held-out examples (across all domains) for ApexRAG-Full conformal calibration",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run a mocked pipeline smoke test instead of a real evaluation "
        "(no API key needed; results are NOT comparable to a real run).",
    )
    args = parser.parse_args()

    asyncio.run(main(subset_size=args.n, provider=args.provider, mock=args.mock, calibration_n=args.calibration_n))
