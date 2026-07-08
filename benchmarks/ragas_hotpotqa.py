"""
benchmarks/ragas_hotpotqa.py — Comprehensive HotpotQA Benchmark.

Compares **ApexRAG** (structural AST navigation) against standard RAG
pipelines on the HotpotQA distractor dataset:

  - Baseline A: **LangChain** (RecursiveCharacterTextSplitter + FAISS + RetrievalQA)
  - Baseline B: **LlamaIndex** (SentenceWindowNodeParser + VectorStoreIndex)
  - Target:     **ApexRAG** (Agentic AST Navigation + Conformal Prediction)

Metrics (RAGAS 0.4.x):
  - ``Faithfulness`` — Are the claims in the answer supported by context?
  - ``AnswerRelevancy`` — How relevant is the answer to the question?
  - ``ContextPrecision`` — How many retrieved chunks are actually useful?

Usage:
    # Real run (requires OPENAI_API_KEY for baselines + RAGAS)
    python benchmarks/ragas_hotpotqa.py --n 10

    # Mock run (no API keys, synthetic data, mock LLM, GT-accuracy only)
    python benchmarks/ragas_hotpotqa.py --n 5 --mock
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
import time
import types
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex_rag.benchmarks.hotpotqa")

# ── RAGAS import patch (must run before any ragas import) ──────────────
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _ragas_patch = types.ModuleType("langchain_community.chat_models.vertexai")
    _ragas_patch.__dict__["ChatVertexAI"] = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _ragas_patch

# ── Project imports ───────────────────────────────────────────────────
from benchmarks.baselines import LangChainBaseline, LlamaIndexBaseline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# Data Loading (HotpotQA + mock fallback)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class HotpotQASample:
    """A single Q/A pair from HotpotQA."""

    question: str
    answer: str
    context_text: str  # Concatenated Wikipedia passages as Markdown


def load_hotpotqa_samples(
    subset_size: int, use_mock: bool = False
) -> list[HotpotQASample]:
    """Load ``subset_size`` samples from HotpotQA (distractor split).

    Falls back to synthetic data when the dataset can't be downloaded or
    ``use_mock=True``.
    """
    if use_mock:
        logger.info("Using MOCK HotpotQA data")
        return _generate_mock_samples(subset_size)

    try:
        from datasets import load_dataset  # noqa: E402

        logger.info("Loading HotpotQA distractor split...")
        dataset = load_dataset(
            "hotpot_qa", "distractor", split="train", streaming=True
        )
        samples: list[HotpotQASample] = []
        for i, item in enumerate(dataset):
            if i >= subset_size:
                break

            md_parts: list[str] = []
            for title, sentences in item["context"]:
                md_parts.append(f"# {title}\n" + " ".join(sentences))
            context_text = "\n\n".join(md_parts)

            samples.append(
                HotpotQASample(
                    question=item["question"],
                    answer=item["answer"],
                    context_text=context_text,
                )
            )

        logger.info("Loaded %d real HotpotQA samples", len(samples))
        return samples

    except Exception as exc:
        logger.warning(
            "Could not load HotpotQA (%s). Falling back to mock data.", exc
        )
        return _generate_mock_samples(subset_size)


def _generate_mock_samples(n: int) -> list[HotpotQASample]:
    """Generate synthetic Q/A pairs matching the HotpotQA structure."""
    mock_data = [
        HotpotQASample(
            question="What is the capital of France?",
            answer="Paris",
            context_text=(
                "# France\nFrance is a country in Western Europe.\n"
                "## Paris\nParis is the capital of France."
            ),
        ),
        HotpotQASample(
            question="When was the Eiffel Tower built?",
            answer="1889",
            context_text=(
                "# Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower.\n"
                "## Construction\nIt was completed in 1889."
            ),
        ),
        HotpotQASample(
            question="What is the chemical symbol for gold?",
            answer="Au",
            context_text=(
                "# Gold\nGold is a chemical element.\n"
                "## Properties\nSymbol: Au. Atomic number: 79."
            ),
        ),
        HotpotQASample(
            question="Who wrote Romeo and Juliet?",
            answer="William Shakespeare",
            context_text=(
                "# Romeo and Juliet\nRomeo and Juliet is a tragedy.\n"
                "## Author\nWilliam Shakespeare was an English playwright."
            ),
        ),
        HotpotQASample(
            question="What is the largest planet in our solar system?",
            answer="Jupiter",
            context_text=(
                "# Jupiter\nJupiter is the fifth planet from the Sun.\n"
                "## Size\nJupiter is the largest planet in the Solar System."
            ),
        ),
        HotpotQASample(
            question="Which element has the atomic number 1?",
            answer="Hydrogen",
            context_text=(
                "# Hydrogen\nHydrogen is the first element.\n"
                "## Properties\nAtomic number: 1. Symbol: H."
            ),
        ),
        HotpotQASample(
            question="What year did World War II end?",
            answer="1945",
            context_text=(
                "# World War II\nWorld War II was a global war.\n"
                "## End\nWorld War II ended in 1945."
            ),
        ),
        HotpotQASample(
            question="Who painted the Mona Lisa?",
            answer="Leonardo da Vinci",
            context_text=(
                "# Mona Lisa\nThe Mona Lisa is a half-length portrait painting.\n"
                "## Artist\nLeonardo da Vinci painted the Mona Lisa between 1503 and 1506."
            ),
        ),
    ]
    while len(mock_data) < n:
        mock_data.append(mock_data[0])
    return mock_data[:n]


# ═══════════════════════════════════════════════════════════════════════
# Mock LLM for ApexRAG (used when no API key is available)
# ═══════════════════════════════════════════════════════════════════════


class MockLLM:
    """Minimal mock provider returning canned JSON responses that
    ApexRAG's agents expect."""

    async def generate(self, prompt: str, **kwargs) -> str:  # noqa: ARG002
        prompt_lower = prompt.lower()
        if "decomposition" in prompt_lower:
            return '{"sub_queries": ["Find the answer in the document"]}'
        if "chosen_id" in prompt_lower:
            return (
                '{"chosen_id": "mock-id-1", "fallback_id": null, '
                '"reason": "Mock selection"}'
            )
        if "answers_query" in prompt_lower or "verify" in prompt_lower:
            return (
                '{"answers_query": true, "confidence": 0.95, '
                '"reason": "Mock verification"}'
            )
        if "evaluate" in prompt_lower:
            return '{"passes_evaluation": true, "reason": "Verified"}'
        if "cite each claim" in prompt_lower or "synthesize" in prompt_lower:
            return "Mocked answer with citation [Node ID: mock-1]."
        return "Mocked answer."

    async def embed(
        self, texts: list[str], **kwargs  # noqa: ARG002
    ) -> list[list[float]]:
        return [[random.uniform(-1.0, 1.0) for _ in range(384)] for _ in texts]

    async def stream_generate(self, prompt: str, **kwargs):  # noqa: ARG002
        yield "Mocked "
        yield "response"


# ═══════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════


async def run_langchain(
    samples: list[HotpotQASample],
) -> tuple[list[str], list[list[str]]]:
    """Run LangChain baseline on all samples."""
    logger.info("─── Running LangChain Baseline ───")
    baseline = LangChainBaseline()
    answers: list[str] = []
    contexts: list[list[str]] = []
    for idx, sample in enumerate(samples):
        t0 = time.monotonic()
        result = await baseline.query(sample.question, sample.context_text)
        elapsed = time.monotonic() - t0
        answers.append(result.answer)
        contexts.append(
            result.contexts if result.contexts else [sample.context_text[:200]]
        )
        logger.info(
            "  [%d/%d] LangChain — %.1fs — %s",
            idx + 1,
            len(samples),
            elapsed,
            sample.question[:60],
        )
    return answers, contexts


async def run_llamaindex(
    samples: list[HotpotQASample],
) -> tuple[list[str], list[list[str]]]:
    """Run LlamaIndex baseline on all samples."""
    logger.info("─── Running LlamaIndex Baseline ───")
    baseline = LlamaIndexBaseline()
    answers: list[str] = []
    contexts: list[list[str]] = []
    for idx, sample in enumerate(samples):
        t0 = time.monotonic()
        result = await baseline.query(sample.question, sample.context_text)
        elapsed = time.monotonic() - t0
        answers.append(result.answer)
        contexts.append(
            result.contexts if result.contexts else [sample.context_text[:200]]
        )
        logger.info(
            "  [%d/%d] LlamaIndex — %.1fs — %s",
            idx + 1,
            len(samples),
            elapsed,
            sample.question[:60],
        )
    return answers, contexts


async def run_apexrag(
    samples: list[HotpotQASample],
    use_mock: bool = False,
) -> tuple[list[str], list[list[str]], list[dict]]:
    """Run ApexRAG on all samples.

    Returns (answers, contexts, extra_metrics).
    """
    from apex_rag import ApexIndex  # noqa: E402

    logger.info("─── Running ApexRAG ───")
    answers: list[str] = []
    contexts: list[list[str]] = []
    extras: list[dict] = []

    # When mocking, pass MockLLM as model so ApexIndex.create() detects it
    # via ``hasattr(model, "generate")``.
    if use_mock or not os.getenv("OPENAI_API_KEY"):
        mock_llm = MockLLM()
        target_model = mock_llm
    else:
        target_model = "gpt-4o-mini"

    async with await ApexIndex.create(
        provider="openai",
        model=target_model,
        db_url="sqlite+aiosqlite:///:memory:",
    ) as index:
        for idx, sample in enumerate(samples):
            t0 = time.monotonic()
            doc_id = await index.ingest_text(
                sample.context_text,
                doc_id=f"hp-{abs(hash(sample.question))}",
                synthesize_summaries=not use_mock,
            )
            answer = await index.query(
                sample.question,
                doc_id,
                coverage=0.90,
                domain="general",
            )
            elapsed = time.monotonic() - t0

            ans_text = (
                answer.answer_text if answer.answer_text else "No answer found."
            )
            answers.append(ans_text)
            ctx = [p.content for p in answer.evidence_packets if p.content]
            contexts.append(ctx if ctx else [sample.context_text[:200]])

            extras.append(
                {
                    "coverage_guarantee": answer.coverage_guarantee,
                    "prediction_set_size": answer.prediction_set_size,
                    "contradictions": len(answer.contradictions),
                    "latency_ms": answer.latency_ms,
                    "nodes_visited": answer.nodes_visited,
                    "llm_calls": answer.llm_calls,
                }
            )
            logger.info(
                "  [%d/%d] ApexRAG — %.1fs — %s (coverage=%.2f, nodes=%d)",
                idx + 1,
                len(samples),
                elapsed,
                sample.question[:50],
                answer.coverage_guarantee,
                answer.nodes_visited,
            )

    return answers, contexts, extras


# ═══════════════════════════════════════════════════════════════════════
# RAGAS Evaluation
# ═══════════════════════════════════════════════════════════════════════


def compute_ground_truth_accuracy(
    answers: list[str],
    ground_truths: list[str],
) -> float:
    """Simple string-overlap accuracy: fraction of answers containing
    the ground truth (case-insensitive)."""
    if not answers:
        return 0.0
    hits = sum(
        1
        for ans, gt in zip(answers, ground_truths, strict=True)
        if gt.lower() in ans.lower()
    )
    return hits / len(answers)


async def evaluate_one_system(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    name: str,
    use_ragas: bool,
) -> dict[str, float]:
    """Evaluate one system's outputs.

    When ``use_ragas=True`` and API keys are available, runs full RAGAS
    evaluation (Faithfulness, AnswerRelevancy, ContextPrecision).
    Always computes ground-truth string-overlap accuracy.
    """
    acc = compute_ground_truth_accuracy(answers, ground_truths)
    scores: dict[str, float] = {
        "accuracy_gt": acc,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
    }

    if not use_ragas:
        return scores

    try:
        from datasets import Dataset as HFDataset  # noqa: E402
        from ragas import evaluate  # noqa: E402
        from ragas.metrics.collections import (  # noqa: E402
            AnswerRelevancy,
            ContextPrecision,
            Faithfulness,
        )

        ds = HFDataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        logger.info("  Evaluating %s with RAGAS...", name)
        metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
        result = evaluate(
            dataset=ds, metrics=metrics, show_progress=False
        )

        for metric in metrics:
            key = metric.name
            val = getattr(result, key, None)
            if val is None:
                val = getattr(result, key.lower(), None)
            scores[key] = float(val) if val is not None else 0.0

        scores["accuracy_gt"] = acc

    except Exception as exc:
        logger.warning("RAGAS evaluation failed for %s: %s", name, exc)

    return scores


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def _print_results(
    results: list[dict],
    system_names: list[str],
    apex_extras: list[dict] | None,
) -> None:
    """Pretty-print the benchmark results table."""
    print()
    print("=" * 78)
    print("  ApexRAG vs LangChain vs LlamaIndex -- HotpotQA Benchmark")
    print("=" * 78)

    header = (
        f"{'System':<20} {'Faithfulness':<15} {'AnswerRel.':<15} "
        f"{'ContextPrec.':<15} {'Acc@GT':<10}"
    )
    print(header)
    print("-" * len(header))

    for idx, name in enumerate(system_names):
        r = results[idx]
        acc = r.get("accuracy_gt", 0.0)
        print(
            f"{name:<20} "
            f"{r.get('faithfulness', 0):.4f}        "
            f"{r.get('answer_relevancy', 0):.4f}        "
            f"{r.get('context_precision', 0):.4f}        "
            f"{acc:.2%}"
        )

    print("-" * len(header))

    if apex_extras:
        print()
        print("  ApexRAG Extended Metrics")
        print("-" * 40)
        n = max(len(apex_extras), 1)
        avg_coverage = sum(e["coverage_guarantee"] for e in apex_extras) / n
        avg_latency = sum(e["latency_ms"] for e in apex_extras) / n
        avg_nodes = sum(e["nodes_visited"] for e in apex_extras) / n
        total_contradictions = sum(e["contradictions"] for e in apex_extras)
        print(f"  Avg Coverage Guarantee:  {avg_coverage:.3f}")
        print(f"  Avg Latency:             {avg_latency:.0f} ms")
        print(f"  Avg Nodes Visited:       {avg_nodes:.1f}")
        print(f"  Total Contradictions:    {total_contradictions}")

    print("=" * 78)


def _export_latex(
    results: list[dict],
    system_names: list[str],
    use_mock: bool,
) -> None:
    """Save a LaTeX table to ``benchmark_results.tex``."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"System & Faithfulness & AnswerRel. & ContextPrec. & Acc@GT \\",
        r"\midrule",
    ]
    for name, r in zip(system_names, results, strict=True):
        lines.append(
            f"  {name} & {r.get('faithfulness', 0):.3f} & "
            f"{r.get('answer_relevancy', 0):.3f} & "
            f"{r.get('context_precision', 0):.3f} & "
            f"{r.get('accuracy_gt', 0):.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{HotpotQA Benchmark Results ({'MOCK' if use_mock else 'REAL'} mode)}}",
            r"\label{tab:hotpotqa}",
            r"\end{table}",
        ]
    )
    tex = "\n".join(lines)
    with open("benchmark_results.tex", "w") as f:
        f.write(tex)
    print("\n  [✓] LaTeX table saved to benchmark_results.tex")


async def main(subset_size: int = 10, use_mock: bool = False) -> None:
    """Run the full HotpotQA benchmark suite."""
    mode = "MOCK" if use_mock else "REAL"
    print()
    print(f"  ApexRAG HotpotQA Benchmark  |  Mode: {mode}  |  Samples: {subset_size}")
    print()

    # 1. Load data
    samples = load_hotpotqa_samples(subset_size, use_mock=use_mock)
    print(f"  Loaded {len(samples)} Hotpotqa samples.\n")
    questions = [s.question for s in samples]
    ground_truths = [s.answer for s in samples]

    # 2. Run pipelines
    langchain_answers, langchain_contexts = await run_langchain(samples)
    llamaindex_answers, llamaindex_contexts = await run_llamaindex(samples)
    apex_answers, apex_contexts, apex_extras = await run_apexrag(
        samples, use_mock=use_mock
    )

    print()
    print("  All pipelines complete. Computing metrics...")

    # 3. Evaluation
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    use_ragas = has_api_key and not use_mock

    results_list: list[dict] = []
    pipeline_data = [
        ("LangChain", langchain_answers, langchain_contexts),
        ("LlamaIndex", llamaindex_answers, llamaindex_contexts),
        ("ApexRAG", apex_answers, apex_contexts),
    ]

    for name, answers, ctxs in pipeline_data:
        scores = await evaluate_one_system(
            questions=questions,
            answers=answers,
            contexts=ctxs,
            ground_truths=ground_truths,
            name=name,
            use_ragas=use_ragas,
        )
        results_list.append(scores)

    # 4. Print results
    system_names = ["LangChain", "LlamaIndex", "ApexRAG"]
    _print_results(results_list, system_names, apex_extras)

    # 5. Export LaTeX
    _export_latex(results_list, system_names, use_mock)

    # 6. Summary line
    print()
    apex_gt = results_list[2].get("accuracy_gt", 0.0)
    lc_gt = results_list[0].get("accuracy_gt", 0.0)
    li_gt = results_list[1].get("accuracy_gt", 0.0)
    print(
        f"  Summary: ApexRAG Acc@GT={apex_gt:.1%} vs "
        f"LangChain={lc_gt:.1%} vs LlamaIndex={li_gt:.1%}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ApexRAG HotpotQA Benchmark (vs LangChain, LlamaIndex)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of HotpotQA samples to test (default: 5)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data and mock LLM (no API key required)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY") and not args.mock:
        print()
        print("  ⚠️  OPENAI_API_KEY not set. Run with --mock for a synthetic demo:")
        print("     python benchmarks/ragas_hotpotqa.py --n 5 --mock")
        print()
        sys.exit(1)

    asyncio.run(main(subset_size=args.n, use_mock=args.mock))
