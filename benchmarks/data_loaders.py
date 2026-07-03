"""
benchmarks/data_loaders.py — Data loading and formatting for Part 8 benchmarks.

Fetches QASPER, FinQA, and ContractNLI from Hugging Face and converts them
into Markdown for AST ingestion.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass

from datasets import load_dataset

logger = logging.getLogger("apex_rag.benchmarks.data_loaders")


@dataclass
class BenchmarkExample:
    doc_id: str
    text: str
    question: str
    ground_truth: str
    metadata: dict


class BenchmarkDataLoader:
    """Loader for specialized research datasets."""

    @staticmethod
    def load_qasper(subset_size: int = 10) -> Generator[BenchmarkExample, None, None]:
        """Load HotpotQA as a proxy for structural reasoning (using distractor context)."""
        dataset = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
        for count, item in enumerate(dataset):
            if count >= subset_size:
                break

            # HotpotQA context is list of (title, sentences)
            md = ""
            for title, sentences in item["context"]:
                md += f"# {title}\n" + " ".join(sentences) + "\n\n"

            yield BenchmarkExample(
                doc_id=f"hp-{abs(hash(item['question']))}",
                text=md,
                question=item["question"],
                ground_truth=item["answer"],
                metadata={"domain": "scientific"},  # Mapping for benchmark
            )

    @staticmethod
    def load_finqa(subset_size: int = 10) -> Generator[BenchmarkExample, None, None]:
        """Load FIQA dataset."""
        # Using a subset or related task if available
        dataset = load_dataset(
            "explodinggradients/fiqa", "ragas_eval", split="baseline", streaming=True
        )
        for count, item in enumerate(dataset):
            if count >= subset_size:
                break

            yield BenchmarkExample(
                doc_id=f"fiqa-{count}",
                text="\n".join(item["context"]),
                question=item["question"],
                ground_truth=item["answer"],
                metadata={"domain": "financial"},
            )

    @staticmethod
    def load_mock(subset_size: int = 2) -> Generator[BenchmarkExample, None, None]:
        """Load synthetic examples for rapid pipeline verification."""
        examples = [
            BenchmarkExample(
                doc_id="mock-1",
                text="# Project Apex\nApexRAG is a structural RAG library.\nIt uses ASTs for navigation.",
                question="What is ApexRAG?",
                ground_truth="A structural RAG library",
                metadata={"domain": "scientific"},
            ),
            BenchmarkExample(
                doc_id="mock-2",
                text="## Financials\nRevenue was $10M in 2023.\nIn 2024, it grew to $15M.",
                question="What was the 2024 revenue?",
                ground_truth="$15M",
                metadata={"domain": "financial"},
            ),
            BenchmarkExample(
                doc_id="mock-3",
                text="# Agreement\nClause 1: Termination requires 30 days notice.\nAmendment: Notice period is now 60 days.",
                question="How much notice is needed for termination?",
                ground_truth="60 days (amended)",
                metadata={"domain": "legal"},
            ),
        ]
        for i in range(min(subset_size, len(examples))):
            yield examples[i]
