import asyncio
import os

import pandas as pd
from datasets import load_dataset

# Baseline: LlamaIndex
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from ragas import evaluate
from ragas.metrics import answer_relevance, context_precision, faithfulness

# Target: ApexRAG
from apex_rag import ApexIndex


async def run_benchmark(subset_size: int = 10):
    print(f"🚀 Starting ApexRAG vs. LlamaIndex Benchmark (HotpotQA, n={subset_size})")

    # 1. Load HotpotQA Subset
    dataset = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
    samples = list(dataset.take(subset_size))

    questions = [s["question"] for s in samples]
    ground_truths = [s["answer"] for s in samples]

    # --- BASELINE: LlamaIndex (Naive Sentence Window Chunker) ---
    print("\n[*] Initializing LlamaIndex Baseline...")
    SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    baseline_answers = []
    baseline_contexts = []

    for sample in samples:
        # Combine HotpotQA context into LlamaIndex Documents
        flat_context = "\n".join([" ".join(c[1]) for c in sample["context"]])
        doc = Document(text=flat_context)

        index = VectorStoreIndex.from_documents([doc])
        query_engine = index.as_query_engine(similarity_top_k=2)

        response = query_engine.query(sample["question"])
        baseline_answers.append(response.response)
        baseline_contexts.append([n.node.text for n in response.source_nodes])

    # --- TARGET: ApexRAG (Structural AST Navigation) ---
    print("\n[*] Initializing ApexRAG (AST Navigation Agent)...")

    apex_answers = []
    apex_contexts = []

    # Use the new provider-string factory
    async with await ApexIndex.create(provider="openai", model="gpt-4o") as index:
        for sample in samples:
            # HotpotQA context is list of (title, sentences)
            # We convert to Markdown to let ApexRAG build an AST
            md_context = ""
            for title, sentences in sample["context"]:
                md_context += f"# {title}\n" + " ".join(sentences) + "\n\n"

            doc_id = await index.ingest_text(
                md_context, doc_id=f"hp-{abs(hash(sample['question']))}"
            )

            # Use the new query API with coverage guarantee
            answer = await index.query(sample["question"], doc_id, coverage=0.95)

            # ApexAnswer contains answer_text and evidence_packets
            apex_answers.append(answer.answer_text if answer.answer_text else "No answer found.")

            # Extract actual evidence used for the contexts metric
            contexts = [p.node.content for p in answer.evidence_packets]
            apex_contexts.append(contexts if contexts else [md_context])

    # --- EVALUATION: RAGAS ---
    print("\n[*] Running RAGAS Evaluation...")

    baseline_data = {
        "question": questions,
        "answer": baseline_answers,
        "contexts": baseline_contexts,
        "ground_truth": ground_truths,
    }

    apex_data = {
        "question": questions,
        "answer": apex_answers,
        "contexts": apex_contexts,
        "ground_truth": ground_truths,
    }

    baseline_results = evaluate(
        pd.DataFrame(baseline_data),
        metrics=[faithfulness, answer_relevance, context_precision],
    )

    apex_results = evaluate(
        pd.DataFrame(apex_data),
        metrics=[faithfulness, answer_relevance, context_precision],
    )

    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS")
    print("=" * 50)
    print(f"{'Metric':<20} | {'LlamaIndex':<12} | {'ApexRAG':<12}")
    print("-" * 50)
    for metric in ["faithfulness", "answer_relevance", "context_precision"]:
        print(f"{metric:<20} | {baseline_results[metric]:.4f}       | {apex_results[metric]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set.")
    else:
        asyncio.run(run_benchmark(subset_size=5))
