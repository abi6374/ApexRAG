"""
apex_rag.temporal — Temporal Intelligence Layer (Part 3).

Components:
    - TemporalExtractor:     3-strategy document date extraction (metadata, regex, LLM)
    - FreshnessScorer:       Exponential-decay freshness computation
    - TemporalContradictionDetector:  3-step conflict detection between nodes
"""
from apex_rag.temporal.temporal_retriever import TemporalRetriever
from apex_rag.temporal.state_reconstructor import StateReconstructor
from apex_rag.temporal.analyzers import ChangeAnalyzer, TrendAnalyzer
from apex_rag.temporal.temporal_agent import TemporalReasoningAgent

__all__ = [
    "TemporalRetriever",
    "StateReconstructor",
    "ChangeAnalyzer",
    "TrendAnalyzer",
    "TemporalReasoningAgent",
]
