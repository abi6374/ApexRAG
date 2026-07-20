#!/usr/bin/env python3
"""
scripts/train_dag_router.py — Train the ML-based DAGRouter classifier.

Generates synthetic multi-intent training data using Gemini (or a fallback
template-based generator), embeds queries with all-MiniLM-L6-v2, trains a
multi-label LogisticRegression classifier, and saves it to ``models/``.

Usage:
    # Train with Gemini-generated data (requires GEMINI_API_KEY)
    python scripts/train_dag_router.py --provider gemini --n 200

    # Train with template-generated data (no API key needed, ~200 samples)
    python scripts/train_dag_router.py --n 200 --templates

    # Show cross-validation accuracy
    python scripts/train_dag_router.py --n 100 --cv 5

Output:
    - models/dag_router_model.joblib — the trained sklearn classifier
    - models/training_data.json       — the generated labeled queries
    - Prints accuracy metrics to stdout
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_dag_router")

# ── Configuration ──────────────────────────────────────────────────────────

_MODEL_DIR = Path("models")
_MODEL_PATH = _MODEL_DIR / "dag_router_model.joblib"
_TRAINING_DATA_PATH = _MODEL_DIR / "training_data.json"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_LABEL_NAMES = ["entity", "citation", "policy"]

# ── Template-based query generator (fallback when no API key) ──────────────

_TEMPLATES: list[tuple[str, set[str]]] = [
    # ── Entity queries ─────────────────────────────────────────────────
    ("Who is the CEO of the company?", {"entity"}),
    ("Who are the key stakeholders?", {"entity"}),
    ("What entities are mentioned in this document?", {"entity"}),
    ("List all organizations referenced", {"entity"}),
    ("Which company is the vendor?", {"entity"}),
    ("Who is responsible for compliance?", {"entity"}),
    ("Identify all people named in the report", {"entity"}),
    ("Which employees are affected?", {"entity"}),
    ("Name the board members", {"entity"}),
    ("Who are the partners in this agreement?", {"entity"}),
    ("What parties are involved?", {"entity"}),
    ("Who is the signatory?", {"entity"}),
    ("Which department owns this policy?", {"entity"}),
    ("List the authors of this document", {"entity"}),
    ("Who approved the budget?", {"entity"}),
    ("What teams are working on this project?", {"entity"}),
    ("Identify the contractors", {"entity"}),
    ("Who is the point of contact?", {"entity"}),
    ("Which subsidiaries are mentioned?", {"entity"}),
    ("Name the executive sponsors", {"entity"}),
    # ── Citation queries ──────────────────────────────────────────────
    ("What does Section 4.2 cite?", {"citation"}),
    ("Find references in the bibliography", {"citation"}),
    ("According to Smith et al., what is the finding?", {"citation"}),
    ("Cite the source for this claim", {"citation"}),
    ("What is the reference for this data?", {"citation"}),
    ("Show me the footnotes in this document", {"citation"}),
    ("Where is this regulation cited?", {"citation"}),
    ("Which sources are referenced?", {"citation"}),
    ("List all works cited", {"citation"}),
    ("What does the appendix reference?", {"citation"}),
    ("Find the citation for the revenue figure", {"citation"}),
    ("What is the authority for this statement?", {"citation"}),
    ("Show references to relevant case law", {"citation"}),
    ("Which documents are cited in the introduction?", {"citation"}),
    ("Find the footnote that supports this claim", {"citation"}),
    ("What does 'see section 5' refer to?", {"citation"}),
    ("List the external references", {"citation"}),
    ("What is cited in the methodology section?", {"citation"}),
    ("Show all endnotes", {"citation"}),
    ("What papers are referenced in this analysis?", {"citation"}),
    # ── Policy queries ────────────────────────────────────────────────
    ("What are the compliance requirements?", {"policy"}),
    ("Which GDPR regulations apply?", {"policy"}),
    ("What is the company policy on data retention?", {"policy"}),
    ("What are the mandatory procedures?", {"policy"}),
    ("List all regulatory obligations", {"policy"}),
    ("What does the governance policy say?", {"policy"}),
    ("What are the security protocols?", {"policy"}),
    ("What is required by HIPAA?", {"policy"}),
    ("What are the SOX compliance rules?", {"policy"}),
    ("What policies govern this process?", {"policy"}),
    ("What are the legal requirements?", {"policy"}),
    ("What rules apply to contractors?", {"policy"}),
    ("What are the ESG reporting standards?", {"policy"}),
    ("What does the code of conduct require?", {"policy"}),
    ("What procedures must be followed?", {"policy"}),
    ("What are the authorization requirements?", {"policy"}),
    ("What restrictions apply?", {"policy"}),
    ("What is prohibited by policy?", {"policy"}),
    ("What are the CCPA compliance obligations?", {"policy"}),
    ("What standards must we meet?", {"policy"}),
    # ── Negative (no DAG needed) ──────────────────────────────────────
    ("What is the revenue for Q3?", set()),
    ("Summarize the document", set()),
    ("Explain the main topics", set()),
    ("What is the budget for next year?", set()),
    ("How many pages are in this document?", set()),
    ("What is the date of this report?", set()),
    ("When was this document published?", set()),
    ("What is the total headcount?", set()),
    ("Show me the executive summary", set()),
    ("What are the key metrics?", set()),
    ("Compare Q1 and Q2 results", set()),
    ("What is the growth rate?", set()),
    ("Show me the org chart", set()),
    ("What is the project timeline?", set()),
    ("List the top 5 customers", set()),
    ("What is our market share?", set()),
    ("How many products do we have?", set()),
    ("What is the net promoter score?", set()),
    ("Show the quarterly trends", set()),
    ("What is the headcount distribution?", set()),
    # ── Multi-intent (complex) ─────────────────────────────────────────
    ("Who is responsible under this policy?", {"entity", "policy"}),
    ("Which company's policy is cited in section 5?", {"entity", "citation", "policy"}),
    ("What entities must comply with GDPR?", {"entity", "policy"}),
    ("Cite the regulation that governs this industry", {"citation", "policy"}),
    ("Which parties are referenced in the compliance policy?", {"entity", "citation", "policy"}),
    ("Who does this policy apply to according to section 3?", {"entity", "citation", "policy"}),
    ("What organizations are cited in the governance document?", {"entity", "citation"}),
    ("Which employees are affected by the new policy?", {"entity", "policy"}),
    ("Find the regulation cited by the compliance officer", {"citation", "policy"}),
    ("What standards do our vendors need to meet?", {"entity", "policy"}),
    # ── Edge cases (hard for regex) ───────────────────────────────────
    ("See page 42 for the reference", {"citation"}),
    ("Chapter 5 covers the rules we must follow", {"policy"}),
    ("The agreement names several contractors", {"entity"}),
    ("This document outlines mandatory steps", {"policy"}),
    ("Our legal team identified obligations", {"policy"}),
    ("The board referenced external research", {"citation"}),
    ("Parties to this agreement include...", {"entity"}),
    ("We need to follow the guidelines", {"policy"}),
    ("The study mentions prior work", {"citation"}),
    ("Key individuals are listed below", {"entity"}),
]

# ── Synthetic data generation using Gemini ────────────────────────────────

_GENERATION_PROMPT = """\
You are generating training data for a multi-label query classifier. \
The classifier needs to decide which of three DAG (Directed Acyclic Graph) \
types are needed for a given query:

- **entity**: Query asks about people, organizations, companies, roles, stakeholders, parties
- **citation**: Query asks about references, sources, footnotes, bibliography, cited works
- **policy**: Query asks about regulations, compliance, rules, policies, governance, requirements

Generate {n} diverse queries in JSON format. Each query should be a JSON object:
{{"query": "...", "labels": ["entity", "citation", "policy"]}}

The labels must be a JSON array containing 0-3 of the label strings. \
If no DAG is needed, use an empty array [].

Cover ALL of these categories:
1. Simple queries needing only entity (15%)
2. Simple queries needing only citation (15%)
3. Simple queries needing only policy (15%)
4. Multi-intent queries needing 2-3 labels (20%)
5. Negative queries needing no DAG (20%)
6. Edge cases with subtle wording that a regex classifier would get wrong (15%)

Return ONLY the JSON array, no other text.
"""


def generate_templates(n: int) -> list[dict[str, Any]]:
    """Generate training data from the template list.

    Repeats templates with slight random variations to reach n samples.
    """
    samples: list[dict[str, Any]] = []
    for query, labels in _TEMPLATES:
        samples.append({"query": query, "labels": sorted(labels)})

    # Add variations to reach n
    while len(samples) < n:
        base = random.choice(_TEMPLATES)
        query, labels = base
        # Slight perturbation
        prefix = random.choice(["", "Please ", "Can you ", "I need to know "])
        suffix = random.choice(["", "?", "?"])
        varied = f"{prefix}{query}{suffix}"
        samples.append({"query": varied, "labels": sorted(labels)})

    random.shuffle(samples)
    return samples[:n]


async def generate_with_gemini(n: int, api_key: str) -> list[dict[str, Any]]:
    """Generate training data using the Gemini API."""
    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = _GENERATION_PROMPT.format(n=n)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("\n", 1)[0]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            logger.info("Generated %d samples via Gemini", len(data))
            return data
    except json.JSONDecodeError:
        logger.warning("Gemini response was not valid JSON. Falling back to templates.")

    return generate_templates(n)


async def main(
    n: int = 200,
    provider: str = "templates",
    cv_folds: int = 0,
) -> None:
    logger.info("=" * 60)
    logger.info("DAG Router Classifier Training")
    logger.info("=" * 60)

    # ── 1. Generate training data ──────────────────────────────────────
    logger.info("Generating %d training samples via %s...", n, provider)

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error(
                "GEMINI_API_KEY not set. Run with --templates or set the env var."
            )
            sys.exit(1)
        samples = await generate_with_gemini(n, api_key)
    else:
        samples = generate_templates(n)

    queries = [s["query"] for s in samples]
    labels_list = [s["labels"] for s in samples]

    # Save raw training data
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_TRAINING_DATA_PATH, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info("Saved training data to %s", _TRAINING_DATA_PATH)

    # ── 2. Binarize labels ─────────────────────────────────────────────
    mlb = MultiLabelBinarizer(classes=_LABEL_NAMES)
    y = mlb.fit_transform(labels_list)

    # Label distribution stats
    label_counts = {name: int(y[:, i].sum()) for i, name in enumerate(_LABEL_NAMES)}
    logger.info("Label distribution: %s", label_counts)

    # ── 3. Embed queries ───────────────────────────────────────────────
    logger.info("Loading embedding model %s...", _EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(_EMBEDDING_MODEL)
    logger.info("Embedding %d queries...", len(queries))
    X = embedder.encode(queries, normalize_embeddings=True, show_progress_bar=True)
    logger.info("Embeddings shape: %s", X.shape)

    # ── 4. Train / evaluate ────────────────────────────────────────────
    base_clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    clf = MultiOutputClassifier(base_clf, n_jobs=-1)

    if cv_folds > 1:
        logger.info("Performing %d-fold cross-validation...", cv_folds)
        scores = cross_val_score(
            MultiOutputClassifier(
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"),
                n_jobs=-1,
            ),
            X,
            y,
            cv=cv_folds,
            scoring="accuracy",
        )
        logger.info(
            "Cross-validation accuracy: %.3f ± %.3f",
            scores.mean(),
            scores.std(),
        )

    # Train final model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y, axis=0)) > 1 else None
    )
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    logger.info("Train accuracy: %.4f", train_acc)
    logger.info("Test accuracy:  %.4f", test_acc)

    # Per-label accuracy
    y_pred = clf.predict(X_test)
    for i, name in enumerate(_LABEL_NAMES):
        label_acc = (y_pred[:, i] == y_test[:, i]).mean()
        logger.info("  %-10s accuracy: %.4f", name, label_acc)

    # ── 5. Save model ──────────────────────────────────────────────────
    joblib.dump(clf, _MODEL_PATH)
    logger.info("Saved classifier to %s", _MODEL_PATH)

    # ── 6. Quick sanity check on test queries ──────────────────────────
    logger.info("\n--- Sanity checks ---")
    test_queries = [
        ("Who is the CEO?", {"entity"}),
        ("What does Section 3 cite?", {"citation"}),
        ("What are the compliance rules?", {"policy"}),
        ("Summarize the document", set()),
        ("Who must comply with ISO standards?", {"entity", "policy"}),
        ("Cite the policy on GDPR", {"citation", "entity", "policy"}),
    ]
    for q, expected in test_queries:
        emb = embedder.encode(q, normalize_embeddings=True).reshape(1, -1)
        pred = clf.predict(emb)
        pred_labels = {_LABEL_NAMES[i] for i in range(3) if pred[0, i] == 1}
        ok = "✓" if pred_labels == expected else "✗"
        logger.info("  %s query=%r  expected=%s  got=%s", ok, q, expected, pred_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DAG Router ML classifier")
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of training samples to generate (default: 200)",
    )
    parser.add_argument(
        "--provider", choices=["templates", "gemini"], default="templates",
        help="Data generation method (default: templates; use 'gemini' for real LLM data)",
    )
    parser.add_argument(
        "--cv", type=int, default=0,
        help="Cross-validation folds (0 = skip, default: 0)",
    )
    args = parser.parse_args()

    asyncio.run(main(n=args.n, provider=args.provider, cv_folds=args.cv))
