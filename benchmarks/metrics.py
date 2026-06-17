"""
benchmarks/metrics.py — Custom research metrics for ApexRAG evaluation.

Implements Empirical Coverage Rate, Contradiction Recall, and Temporal Accuracy.
"""

from __future__ import annotations

import numpy as np
from typing import Any
from apex_rag.models.unified_models import ApexAnswer


class ApexMetrics:
    """Calculator for novel research metrics."""

    @staticmethod
    def empirical_coverage(answer: ApexAnswer, ground_truth: str) -> float:
        """
        Check if the prediction set contains the answer.
        Since conformal prediction returns a set of nodes, we check if 
        any node in the set contains the ground truth or is relevant.
        (Simplified: 1.0 if answer_text is faithful to ground_truth).
        """
        # In a strict research setting, we'd check if the node_id of the 
        # correct passage is in answer.evidence_packets.
        # For this library, we'll use string overlap as a proxy.
        gt_lower = ground_truth.lower()
        for pkt in answer.evidence_packets:
            if gt_lower in pkt.node.content.lower():
                return 1.0
        return 0.0

    @staticmethod
    def contradiction_recall(answer: ApexAnswer, expected_label: str) -> float:
        """
        % of labeled contradictions successfully surfaced.
        Only applicable to ContractNLI 'Contradiction' examples.
        """
        if expected_label != "Contradiction":
            return np.nan # Not applicable
        
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
            return 1.0 # Trivial
        
        # Is the first one the max?
        return 1.0 if scores[0] == max(scores) else 0.0
