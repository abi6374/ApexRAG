"""
benchmarks/metrics.py — Custom research metrics for ApexRAG evaluation.

Implements Empirical Coverage Rate, Contradiction Recall, and Temporal Accuracy.
"""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

from apex_rag.models.unified_models import ApexAnswer


def _normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization: lowercase, strip punctuation/articles,
    collapse whitespace. Shared by exact_match and f1_score below."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """1.0 if the normalized prediction equals the normalized ground truth."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token-level F1 between a predicted answer string and the
    ground truth. Unlike a raw substring check, this rewards a prediction
    that contains the right words without needing an exact phrase match,
    and penalizes a prediction that's mostly irrelevant filler around the
    right answer -- so it doesn't snap to clean 1.0/0.0 the way substring
    containment does.
    """
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


class ApexMetrics:
    """Calculator for novel research metrics."""

    @staticmethod
    def empirical_coverage(answer: ApexAnswer, ground_truth: str) -> float:
        """
        Token-level F1 between the synthesized answer text and the ground
        truth (SQuAD-style). Falls back to 0.0 if no answer was produced.

        Previously this checked raw substring containment of the ground
        truth inside a retrieved evidence packet's content, which snapped
        to clean 1.0/0.0 for almost every example (either the exact ground
        truth string was verbatim in a chunk, or it wasn't) and didn't
        actually score the synthesized *answer* at all. F1 against the
        answer text is closer to how standard QA benchmarks report accuracy.
        """
        if not answer.answer_text:
            return 0.0
        return f1_score(answer.answer_text, ground_truth)

    @staticmethod
    def contradiction_recall(answer: ApexAnswer, expected_label: str) -> float:
        """
        % of labeled contradictions successfully surfaced.
        Only applicable to ContractNLI 'Contradiction' examples.
        """
        if expected_label != "Contradiction":
            return np.nan  # Not applicable

        return 1.0 if len(answer.contradictions) > 0 else 0.0

    @staticmethod
    def temporal_accuracy(answer: ApexAnswer) -> float:
        """
        Check if the freshest evidence is ranked highest.
        """
        if not answer.evidence_packets:
            return 0.0

        scores = [p.temporal_metadata.freshness_score for p in answer.evidence_packets]
        if len(scores) < 2:
            return 1.0  # Trivial

        # Is the first one the max?
        return 1.0 if scores[0] == max(scores) else 0.0
