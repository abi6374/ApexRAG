# DAG Router Accuracy — Honest Held-Out Evaluation

## Why this file exists

`benchmarks/dag_router_accuracy_results.json` previously reported a **100%
exact-match accuracy** for the "ML" router. That number was real, but
misleading: it came from a TF‑IDF+LogisticRegression model **trained and
scored on the same 69-query set** — the file's own `methodology` field said
so explicitly ("measures fit, not generalization"). A model that memorizes
its own eval set will always look perfect; it says nothing about how it
performs on a query it hasn't seen.

This file reports the number that actually matters: how the **real,
production classifier** (`apex_rag/agents/planner/ml_router.py` —
MiniLM embeddings + LogisticRegression) performs on the 69-query set
**zero-shot**, having been trained only on an independent 200-query
templated dataset (`models/training_data.json`) that shares no rows with
the eval set.

## Methodology (Actual Measured Values)

1. `python scripts/train_dag_router.py --n 200 --cv 5` — trains the real
   MiniLM+LogReg classifier on 200 templated queries, with a genuine
   80/20 held-out split (`sklearn.model_selection.train_test_split`,
   `stratify=y`, `random_state=42`) plus 5-fold cross-validation. Saves
   `models/dag_router_model.joblib`.
2. `python benchmarks/run_dag_router_accuracy_benchmark.py` — loads that
   saved model and evaluates it on the 69-query hand-curated
   `_EVAL_SET` (17 rows mirrored from `tests/test_dag_gating.py` + 52
   hand-labeled edge cases). None of these 69 queries were used in step 1.

## Results (Measured)

### Training-time held-out numbers (200-query templated set, from step 1)

| Metric | Value |
|---|---|
| 5-fold CV accuracy | 85.0% ± 5.9% |
| Held-out test accuracy (40 rows, 20% split) | 92.5% |
| Held-out per-label accuracy (entity / citation / policy) | 97.5% / 97.5% / 97.5% |

### Zero-shot generalization on the independent 69-query eval set

| System | Exact-match accuracy |
|---|---:|
| Heuristic (`DAGRouter`, regex-based) | 63.77% |
| **Real MiniLM+LogReg (held-out, never seen these 69 queries)** | **66.67%** |
| ~~TF-IDF+LogReg, in-sample fit~~ (trained AND scored on these same 69 rows — not a fair number, kept only for reference) | ~~100.00%~~ |

| | Heuristic | Real MiniLM (held-out) |
|---|---:|---:|
| Entity F1 | 76.60% | 80.77% |
| Citation F1 | 72.73% | 78.79% |
| Policy F1 | 83.02% | 87.27% |
| Existing-test subset (n=17) exact match | 100% | 82% |
| Edge-case subset (n=52) exact match | 52% | 62% |

Per-query latency: heuristic ~10µs, real MiniLM+LogReg ~16.5ms p50
(dominated by the sentence-embedding forward pass — no GPU, single-query
calls, no batching).

## Analysis

- The real model **does** generalize better than the heuristic overall
  (+2.9 points exact-match, and a clear win on per-label F1 across all
  three labels), but the margin is modest — nothing like the fake 100%
  previously reported.
- It's *worse* than the heuristic on the 17 "existing" test cases (82% vs
  100%) — those are simple, canonical phrasings the regex heuristic was
  explicitly written to catch. The real model earns its edge entirely on
  the 52 harder "edge case" queries (62% vs 52%), which is exactly the
  set the heuristic was known to struggle with (implicit references,
  indirect policy language, etc.) — see `improvements` in
  `dag_router_accuracy_results.json` for the specific queries.
- Latency cost is real: ~1,600× slower per query than the heuristic
  (16.5ms vs 10µs). For a router that runs on every query, this is a
  genuine accuracy/latency trade-off, not a free win — worth keeping in
  mind before defaulting `router_backend` to `"ml"` in production.
- Both the 92.5% training-set held-out number and the 66.67% cross-dataset
  number are legitimate, but they're not the same claim: 92.5% is
  in-distribution generalization (held-out rows drawn from the same
  templated distribution as training), while 66.67% is a harder,
  out-of-distribution test against an independently hand-curated set with
  different phrasing conventions. Report the 66.67% number, not the 92.5%
  one, when the question is "does this classifier work on real queries it
  wasn't tuned for."

## Reproducing

```bash
pip install scikit-learn sentence-transformers numpy   # or: pip install -e ".[ml]"
python scripts/train_dag_router.py --n 200 --cv 5
python benchmarks/run_dag_router_accuracy_benchmark.py
```

## File Changes

| File | Change |
|------|--------|
| `models/dag_router_model.joblib` | **New** — real trained MiniLM+LogReg classifier |
| `benchmarks/run_dag_router_accuracy_benchmark.py` | Added `evaluate_real_model_holdout()`; relabeled the old in-sample TF-IDF number instead of presenting it as comparable; fixed stale 67/50 counts to 69/52 |
| `benchmarks/dag_router_accuracy_results.json` | Regenerated with `ml_in_sample_fit` (old number, relabeled) and `ml_real_holdout` (new, genuine) sections |
| `benchmarks/dag_router_accuracy_honest.md` | **New** — this file |
