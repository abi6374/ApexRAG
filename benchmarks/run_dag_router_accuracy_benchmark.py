#!/usr/bin/env python3
"""
benchmarks/run_dag_router_accuracy_benchmark.py

Accuracy benchmark: heuristic DAGRouter vs. two ML variants:

1. **TF-IDF + LogisticRegression, in-sample fit** — trained AND scored on
   this file's own 69-query `_EVAL_SET`. This number measures how well the
   model can memorize this specific set, not how it generalizes. Kept for
   reference only; never compare it to the heuristic as if it were a fair
   fight (see ``methodology`` in the output JSON).
2. **Real MiniLM + LogisticRegression, held-out generalization** — the
   actual production classifier (`apex_rag/agents/planner/ml_router.py`),
   trained by ``scripts/train_dag_router.py`` on an independent 200-query
   templated dataset (`models/training_data.json`) and evaluated here,
   zero-shot, on this file's 69-query `_EVAL_SET`. Since the eval set was
   never seen during training, this is a legitimate generalization number.
   Requires ``models/dag_router_model.joblib`` to exist — run
   ``python scripts/train_dag_router.py --n 200 --cv 5`` first.

Dataset:
    - 17 existing test cases from tests/test_dag_gating.py (TestDAGRouter)
    - 52 hand-labeled edge cases covering known heuristic weaknesses
    - 69 queries total

Metrics:
    - Per-label accuracy, precision, recall, F1
    - Exact-match accuracy (all 3 labels correct)
    - Per-query latency (us)
    - Confusion matrix per label

Usage:
    python benchmarks/run_dag_router_accuracy_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from apex_rag.agents.planner.dag_router import DAGRouter

_REAL_MODEL_PATH = Path("models/dag_router_model.joblib")
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dag_router_benchmark")

_LABEL_NAMES = ["entity", "citation", "policy"]

# ===========================================================================
# Evaluation Dataset: 17 existing test cases + 50 hand-labeled edge cases
# ===========================================================================

_EVAL_SET: list[dict[str, Any]] = [
    # ---- 1-17: Existing test cases from test_dag_gating.py ---------------
    {"query": "Who is the CEO?", "labels": {"entity"}, "category": "existing"},
    {"query": "What entities are mentioned?", "labels": {"entity"}, "category": "existing"},
    {"query": "Which company employs John?", "labels": {"entity"}, "category": "existing"},
    {"query": "List all organizations", "labels": {"entity"}, "category": "existing"},
    {"query": "What does Section 3.2 cite?", "labels": {"citation"}, "category": "existing"},
    {"query": "Find references in the bibliography", "labels": {"citation"}, "category": "existing"},
    {"query": "According to Smith et al., what is...", "labels": {"citation"}, "category": "existing"},
    {"query": "See page regression for details", "labels": frozenset(), "category": "existing"},
    {"query": "What are the compliance requirements?", "labels": {"policy"}, "category": "existing"},
    {"query": "Which GDPR regulations apply?", "labels": {"policy"}, "category": "existing"},
    {"query": "All employees shall comply with...", "labels": {"entity", "policy"}, "category": "existing"},
    {"query": "Legal obligations under HIPAA", "labels": {"policy"}, "category": "existing"},
    {"query": "Who must comply with ISO standards?", "labels": {"entity", "policy"}, "category": "existing"},
    {"query": "Cite the company policy on GDPR", "labels": {"citation", "entity", "policy"}, "category": "existing"},
    {"query": "What is the revenue for Q3?", "labels": frozenset(), "category": "existing"},
    {"query": "Summarize the document", "labels": frozenset(), "category": "existing"},
    {"query": "Explain the main topics", "labels": frozenset(), "category": "existing"},
    # ---- 18-42: Edge cases - semantic understanding ----------------------
    {"query": "See page 42 for the data source", "labels": {"citation"}, "category": "edge"},
    {"query": "Chapter 3 lists the references", "labels": {"citation"}, "category": "edge"},
    {"query": "Turn to appendix B for citations", "labels": {"citation"}, "category": "edge"},
    {"query": "The related work section has sources", "labels": {"citation"}, "category": "edge"},
    {"query": "Where does this data come from?", "labels": {"citation"}, "category": "edge"},
    {"query": "What is the source of this claim?", "labels": {"citation"}, "category": "edge"},
    {"query": "Who published this research?", "labels": {"entity", "citation"}, "category": "edge"},
    {"query": "What study is this based on?", "labels": {"citation"}, "category": "edge"},
    {"query": "What are our obligations under the new rules?", "labels": {"policy"}, "category": "edge"},
    {"query": "Do we have permission to share this data?", "labels": {"policy"}, "category": "edge"},
    {"query": "What restrictions apply to contractors?", "labels": {"policy"}, "category": "edge"},
    {"query": "Is this activity allowed under current rules?", "labels": {"policy"}, "category": "edge"},
    {"query": "What happens if we don't follow the guidelines?", "labels": {"policy"}, "category": "edge"},
    {"query": "Are there any constraints on data usage?", "labels": {"policy"}, "category": "edge"},
    {"query": "What are the dos and don'ts for this process?", "labels": {"policy"}, "category": "edge"},
    {"query": "What should we do to stay compliant?", "labels": {"policy"}, "category": "edge"},
    {"query": "Who is mentioned as the contact person?", "labels": {"entity"}, "category": "edge"},
    {"query": "What teams are involved in this project?", "labels": {"entity"}, "category": "edge"},
    {"query": "Which department owns this process?", "labels": {"entity"}, "category": "edge"},
    {"query": "Name the signatories on this agreement", "labels": {"entity"}, "category": "edge"},
    {"query": "List the committee members", "labels": {"entity"}, "category": "edge"},
    {"query": "The employee roster needs updating", "labels": {"entity"}, "category": "edge"},
    # ---- 43-67: Multi-intent queries ------------------------------------
    {"query": "Which parties are bound by this agreement and what rules apply?", "labels": {"entity", "policy"}, "category": "edge"},
    {"query": "Who wrote the policy cited in section 5?", "labels": {"citation", "entity", "policy"}, "category": "edge"},
    {"query": "Name the regulatory bodies referenced", "labels": {"citation", "entity", "policy"}, "category": "edge"},
    {"query": "What standards must our vendors meet according to the contract?", "labels": {"citation", "entity", "policy"}, "category": "edge"},
    {"query": "List all personnel who need to follow the code of conduct", "labels": {"entity", "policy"}, "category": "edge"},
    {"query": "According to the compliance team, what are the rules?", "labels": {"entity", "policy"}, "category": "edge"},
    {"query": "What regulations from the authorities apply to us?", "labels": {"entity", "policy"}, "category": "edge"},
    {"query": "Cite the person responsible for governance", "labels": {"citation", "entity", "policy"}, "category": "edge"},
    {"query": "Reference the company policy on who can approve expenses", "labels": {"citation", "entity", "policy"}, "category": "edge"},
    {"query": "Which employees need authorisation according to section 4?", "labels": {"citation", "entity", "policy"}, "category": "edge"},
    # ---- 68-77: Domain-specific signals ---------------------------------
    {"query": "What are the material weakness disclosures?", "labels": {"policy"}, "category": "edge"},
    {"query": "How do we report under SOX?", "labels": {"policy"}, "category": "edge"},
    {"query": "What are the AML/KYC requirements for new clients?", "labels": {"entity", "policy"}, "category": "edge"},
    {"query": "Who is the data protection officer?", "labels": {"entity"}, "category": "edge"},
    {"query": "What is our data retention schedule?", "labels": {"policy"}, "category": "edge"},
    {"query": "Which ISO 27001 controls apply?", "labels": {"policy"}, "category": "edge"},
    {"query": "What safety protocols are mandated by OSHA?", "labels": {"citation", "policy"}, "category": "edge"},
    {"query": "Who are the key contacts for the audit committee?", "labels": {"entity"}, "category": "edge"},
    {"query": "What are the board's fiduciary duties?", "labels": {"entity", "policy"}, "category": "edge"},
    {"query": "Which courts have jurisdiction per the agreement?", "labels": {"citation", "entity"}, "category": "edge"},
    # ---- 78-87: Easy negatives ------------------------------------------
    {"query": "What was the total revenue in 2024?", "labels": frozenset(), "category": "edge"},
    {"query": "Show me the quarterly growth chart", "labels": frozenset(), "category": "edge"},
    {"query": "How many customers did we add this quarter?", "labels": frozenset(), "category": "edge"},
    {"query": "What is the NPS score trending?", "labels": frozenset(), "category": "edge"},
    {"query": "Calculate the year-over-year growth rate", "labels": frozenset(), "category": "edge"},
    {"query": "Compare Q3 and Q4 performance", "labels": frozenset(), "category": "edge"},
    {"query": "What is our current headcount?", "labels": frozenset(), "category": "edge"},
    {"query": "Show the org chart for the engineering team", "labels": frozenset(), "category": "edge"},
    {"query": "What projects are in the pipeline?", "labels": frozenset(), "category": "edge"},
    {"query": "When is the next board meeting?", "labels": frozenset(), "category": "edge"},
]

# ===========================================================================
# ML Classifier
# ===========================================================================


def train_ml_classifier(
    queries: list[str],
    labels: list[set[str]],
) -> tuple[TfidfVectorizer, MultiOutputClassifier]:
    mlb = MultiLabelBinarizer(classes=_LABEL_NAMES)
    y = mlb.fit_transform(labels)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2000,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(queries)

    base = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42,
    )
    clf = MultiOutputClassifier(base, n_jobs=1)
    clf.fit(X, y)
    return vectorizer, clf


def evaluate_real_model_holdout(
    queries: list[str],
) -> tuple[list[frozenset[str]], list[float]] | None:
    """Zero-shot-evaluate the real, production MiniLM+LogReg model against
    ``queries`` (the 69-query _EVAL_SET, which the model has never seen).

    Returns ``None`` if ``models/dag_router_model.joblib`` doesn't exist yet
    (i.e. ``scripts/train_dag_router.py`` hasn't been run), so callers can
    skip this section gracefully instead of crashing.
    """
    if not _REAL_MODEL_PATH.exists():
        logger.warning(
            "  [!] %s not found -- skipping real held-out MiniLM+LogReg eval.\n"
            "      Run `python scripts/train_dag_router.py --n 200 --cv 5` first.",
            _REAL_MODEL_PATH,
        )
        return None

    import joblib
    from sentence_transformers import SentenceTransformer

    clf = joblib.load(_REAL_MODEL_PATH)
    embedder = SentenceTransformer(_EMBEDDING_MODEL)
    embedder.encode("warmup", normalize_embeddings=True)  # exclude cold-start from timings

    preds: list[frozenset[str]] = []
    times: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        emb = embedder.encode(q, normalize_embeddings=True).reshape(1, -1)
        y_pred = clf.predict(emb)
        elapsed = time.perf_counter() - t0
        preds.append(frozenset({_LABEL_NAMES[j] for j in range(3) if y_pred[0, j] == 1}))
        times.append(elapsed)
    return preds, times


# ===========================================================================
# Metrics
# ===========================================================================


def compute_metrics(
    y_true: list[frozenset[str]],
    y_pred: list[frozenset[str]],
) -> dict[str, Any]:
    """Compute per-label and overall metrics."""
    n = len(y_true)
    results: dict[str, Any] = {"n_queries": n}

    exact_matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    results["exact_match_accuracy"] = round(exact_matches / n, 4)

    for label in _LABEL_NAMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if label in t and label in p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if label not in t and label in p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if label in t and label not in p)
        tn = sum(1 for t, p in zip(y_true, y_pred) if label not in t and label not in p)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0

        results[f"{label}_accuracy"] = round(accuracy, 4)
        results[f"{label}_precision"] = round(precision, 4)
        results[f"{label}_recall"] = round(recall, 4)
        results[f"{label}_f1"] = round(f1, 4)
        results[f"{label}_tp"] = tp
        results[f"{label}_fp"] = fp
        results[f"{label}_fn"] = fn
        results[f"{label}_tn"] = tn

    return results


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    print("=" * 78)
    print("  DAG Router Accuracy Benchmark")
    print("  Heuristic (DAGRouter) vs ML (TF-IDF + LogisticRegression)")
    print("=" * 78)

    # Dataset stats
    existing_idx = [i for i, s in enumerate(_EVAL_SET) if s["category"] == "existing"]
    edge_idx = [i for i, s in enumerate(_EVAL_SET) if s["category"] == "edge"]
    n_existing = len(existing_idx)
    n_edge = len(edge_idx)
    print(f"\n  Dataset: {len(_EVAL_SET)} queries ({n_existing} existing + {n_edge} edge cases)")

    label_counts: dict[str, int] = {}
    for s in _EVAL_SET:
        for lbl in s["labels"]:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print("  Label distribution:")
    for lbl in _LABEL_NAMES:
        print(f"    {lbl}: {label_counts.get(lbl, 0)}")
    print(f"    none: {sum(1 for s in _EVAL_SET if not s['labels'])}")

    # Prepare data
    queries = [s["query"] for s in _EVAL_SET]
    labels_gt = [s["labels"] for s in _EVAL_SET]

    # NOTE: ML trained on full dataset (no hold-out). Scores measure fit, not generalization.
    logger.info("\n  Training TF-IDF + LogisticRegression classifier (in-sample fit)...")
    vectorizer, ml_clf = train_ml_classifier(queries, [set(l) for l in labels_gt])
    X_eval = vectorizer.transform(queries)

    logger.info(
        "\n  Evaluating real MiniLM + LogisticRegression classifier "
        "(held-out generalization -- trained on an independent 200-query set)..."
    )
    real_result = evaluate_real_model_holdout(queries)

    # Run heuristic
    heuristic = DAGRouter()
    heuristic_preds: list[frozenset[str]] = []
    heuristic_times: list[float] = []

    for s in _EVAL_SET:
        t0 = time.perf_counter()
        pred = heuristic.classify(s["query"])
        elapsed = time.perf_counter() - t0
        heuristic_preds.append(pred)
        heuristic_times.append(elapsed)

    # Run ML classifier
    ml_preds: list[frozenset[str]] = []
    ml_times: list[float] = []

    for i in range(len(_EVAL_SET)):
        t0 = time.perf_counter()
        y_pred = ml_clf.predict(X_eval[i])
        pred_labels = frozenset({_LABEL_NAMES[j] for j in range(3) if y_pred[0, j] == 1})
        elapsed = time.perf_counter() - t0
        ml_preds.append(pred_labels)
        ml_times.append(elapsed)

    # Compute metrics
    heuristic_metrics = compute_metrics(labels_gt, heuristic_preds)
    ml_metrics = compute_metrics(labels_gt, ml_preds)

    real_metrics: dict[str, Any] | None = None
    real_avg = real_p50 = None
    if real_result is not None:
        real_preds, real_times = real_result
        real_metrics = compute_metrics(labels_gt, real_preds)
        real_avg = sum(real_times) / len(real_times)
        real_p50 = sorted(real_times)[len(real_times) // 2]

    # Print results
    print(f"\n{'=' * 78}")
    print("  Results  (TF-IDF column = in-sample fit on these 69 queries, NOT generalization)")
    print(f"{'=' * 78}")
    if real_metrics is not None:
        print(
            f"  Real MiniLM+LogReg held-out exact-match accuracy: "
            f"{real_metrics['exact_match_accuracy']:.2%}  "
            f"(trained on an independent 200-query set, never seen this eval set)"
        )
    else:
        print(
            "  [!] Real MiniLM+LogReg held-out eval skipped -- "
            "run `python scripts/train_dag_router.py --n 200 --cv 5` first."
        )

    # Exact match
    print(f"\n  {'Metric':<30} {'Heuristic':>12} {'ML (TF-IDF)':>12} {'Diff':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
    hema = heuristic_metrics["exact_match_accuracy"]
    mlema = ml_metrics["exact_match_accuracy"]
    print(f"  {'Exact match accuracy':<30} {hema:>10.2%}  {mlema:>10.2%}  {mlema - hema:>+7.2%}")
    if real_metrics is not None:
        rema = real_metrics["exact_match_accuracy"]
        print(f"  {'  (real MiniLM, held-out)':<30} {'':>12} {rema:>10.2%}  {rema - hema:>+7.2%} vs heuristic")

    # Per-label
    for label in _LABEL_NAMES:
        h = heuristic_metrics
        m = ml_metrics
        print(f"\n  -- {label.upper()} --")
        print(f"  {'Accuracy':<30} {h[f'{label}_accuracy']:>10.2%}  {m[f'{label}_accuracy']:>10.2%}  {m[f'{label}_accuracy'] - h[f'{label}_accuracy']:>+7.2%}")
        print(f"  {'Precision':<30} {h[f'{label}_precision']:>10.2%}  {m[f'{label}_precision']:>10.2%}  {m[f'{label}_precision'] - h[f'{label}_precision']:>+7.2%}")
        print(f"  {'Recall':<30} {h[f'{label}_recall']:>10.2%}  {m[f'{label}_recall']:>10.2%}  {m[f'{label}_recall'] - h[f'{label}_recall']:>+7.2%}")
        print(f"  {'F1':<30} {h[f'{label}_f1']:>10.2%}  {m[f'{label}_f1']:>10.2%}  {m[f'{label}_f1'] - h[f'{label}_f1']:>+7.2%}")

    # Confusion matrices
    print(f"\n  -- Confusion matrices --")
    for label in _LABEL_NAMES:
        h = heuristic_metrics
        m = ml_metrics
        print(f"  {label}: "
              f"TP={h[f'{label}_tp']} FP={h[f'{label}_fp']} FN={h[f'{label}_fn']} TN={h[f'{label}_tn']}  |  "
              f"ML: TP={m[f'{label}_tp']} FP={m[f'{label}_fp']} FN={m[f'{label}_fn']} TN={m[f'{label}_tn']}")

    # Latency
    h_avg = sum(heuristic_times) / len(heuristic_times)
    m_avg = sum(ml_times) / len(ml_times)
    h_p50 = sorted(heuristic_times)[len(heuristic_times) // 2]
    m_p50 = sorted(ml_times)[len(ml_times) // 2]
    print(f"\n  {'-'*78}")
    print(f"  {'Latency (per query)':<30} {'Heuristic':>12} {'ML (TF-IDF)':>12}")
    print(f"  {'Mean':<30} {h_avg*1e6:>10.1f}us  {m_avg*1e6:>10.1f}us")
    print(f"  {'p50':<30} {h_p50*1e6:>10.1f}us  {m_p50*1e6:>10.1f}us")
    if real_avg is not None and real_p50 is not None:
        print(f"  {'Real MiniLM+LogReg mean':<30} {real_avg*1e6:>10.1f}us  (includes embedding cost)")
        print(f"  {'Real MiniLM+LogReg p50':<30} {real_p50*1e6:>10.1f}us")

    # Category breakdown
    print(f"\n  -- Category breakdown (exact match) --")
    for cat, name in [("existing", "Existing tests"), ("edge", "Edge cases")]:
        idx = [i for i, s in enumerate(_EVAL_SET) if s["category"] == cat]
        if not idx:
            continue
        h_acc = sum(1 for i in idx if heuristic_preds[i] == labels_gt[i]) / len(idx)
        m_acc = sum(1 for i in idx if ml_preds[i] == labels_gt[i]) / len(idx)
        line = f"  {name:<20} n={len(idx):>3}  heuristic={h_acc:.0%}  ml(in-sample)={m_acc:.0%}"
        if real_metrics is not None:
            real_preds_for_cat = real_result[0]  # type: ignore[index]
            r_acc = sum(1 for i in idx if real_preds_for_cat[i] == labels_gt[i]) / len(idx)
            line += f"  ml(held-out)={r_acc:.0%}"
        print(line)

    # Error analysis
    print(f"\n  -- Where ML beats heuristic (examples) --")
    count = 0
    for i, (gt, h_pred, m_pred) in enumerate(zip(labels_gt, heuristic_preds, ml_preds)):
        if h_pred != gt and m_pred == gt:
            print(f"  + {_EVAL_SET[i]['query'][:65]:<67} gt={set(gt)}")
            count += 1
            if count >= 8:
                break

    print(f"\n  -- Where heuristic beats ML (examples) --")
    count = 0
    for i, (gt, h_pred, m_pred) in enumerate(zip(labels_gt, heuristic_preds, ml_preds)):
        if h_pred == gt and m_pred != gt:
            print(f"  - {_EVAL_SET[i]['query'][:65]:<67} gt={set(gt)}")
            count += 1
            if count >= 8:
                break

    # Save results
    results = {
        "dataset_size": len(_EVAL_SET),
        "n_existing": n_existing,
        "n_edge": n_edge,
        "label_counts": {lbl: label_counts.get(lbl, 0) for lbl in _LABEL_NAMES},
        "methodology": (
            "'ml' (TF-IDF+LogReg) is trained AND evaluated on this same "
            f"{len(_EVAL_SET)}-query dataset (no hold-out) -- it measures "
            "in-sample fit, not generalization, and should not be compared "
            "directly to the heuristic as if it were a fair evaluation. "
            "'ml_real_holdout' (when present) is the actual production "
            "MiniLM+LogReg classifier (apex_rag/agents/planner/ml_router.py), "
            "trained by scripts/train_dag_router.py on an independent "
            "200-query templated dataset (models/training_data.json) that "
            "does not overlap this eval set -- its score here is a genuine "
            "zero-shot generalization number."
        ),
        "heuristic": heuristic_metrics,
        "ml_in_sample_fit": ml_metrics,
        "ml_real_holdout": real_metrics,
        "latency_us": {
            "heuristic_mean_us": round(h_avg * 1e6, 1),
            "heuristic_p50_us": round(h_p50 * 1e6, 1),
            "ml_in_sample_fit_mean_us": round(m_avg * 1e6, 1),
            "ml_in_sample_fit_p50_us": round(m_p50 * 1e6, 1),
            "ml_real_holdout_mean_us": round(real_avg * 1e6, 1) if real_avg is not None else None,
            "ml_real_holdout_p50_us": round(real_p50 * 1e6, 1) if real_p50 is not None else None,
        },
        "exact_match_by_category": {
            "existing_heuristic": round(
                sum(1 for i in existing_idx if heuristic_preds[i] == labels_gt[i]) / n_existing, 4
            ) if n_existing else 0,
            "existing_ml": round(
                sum(1 for i in existing_idx if ml_preds[i] == labels_gt[i]) / n_existing, 4
            ) if n_existing else 0,
            "edge_heuristic": round(
                sum(1 for i in edge_idx if heuristic_preds[i] == labels_gt[i]) / n_edge, 4
            ) if n_edge else 0,
            "edge_ml": round(
                sum(1 for i in edge_idx if ml_preds[i] == labels_gt[i]) / n_edge, 4
            ) if n_edge else 0,
            "existing_ml_real_holdout": (
                round(sum(1 for i in existing_idx if real_result[0][i] == labels_gt[i]) / n_existing, 4)
                if real_result is not None and n_existing else None
            ),
            "edge_ml_real_holdout": (
                round(sum(1 for i in edge_idx if real_result[0][i] == labels_gt[i]) / n_edge, 4)
                if real_result is not None and n_edge else None
            ),
        },
        "improvements": [
            {"query": _EVAL_SET[i]["query"], "gt": list(gt), "heuristic_pred": list(h_pred)}
            for i, (gt, h_pred, m_pred) in enumerate(zip(labels_gt, heuristic_preds, ml_preds))
            if h_pred != gt and m_pred == gt
        ],
    }

    out_path = Path("benchmarks/dag_router_accuracy_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [ok] Full results saved to {out_path}")


if __name__ == "__main__":
    main()
