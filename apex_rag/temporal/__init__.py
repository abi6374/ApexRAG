"""
apex_rag.temporal — Temporal Intelligence Layer.

Components:
    - TemporalExtractor:          3-strategy document date extraction (metadata, regex, LLM)
    - FreshnessScorer:            Exponential-decay freshness computation
    - TemporalContradictionDetector:  3-step conflict detection between nodes
    - ChangeAnalyzer:             Structured diff analysis between versions
    - TrendAnalyzer:              Direction, growth/decline, anomaly detection
    - VersionResolver:            Version resolution and authoritative node selection
    - TemporalReasoningService:   Enterprise service for time-aware queries
"""
from apex_rag.temporal.analyzers import ChangeAnalyzer, TrendAnalyzer
from apex_rag.temporal.state_reconstructor import StateReconstructor
from apex_rag.temporal.temporal_agent import TemporalReasoningAgent
from apex_rag.temporal.temporal_retriever import TemporalRetriever
from apex_rag.temporal.version_resolver import VersionResolver
from apex_rag.temporal.reasoning_service import TemporalReasoningService
from apex_rag.temporal.version_service import TemporalVersionService

__all__ = [
    "TemporalRetriever",
    "StateReconstructor",
    "ChangeAnalyzer",
    "TrendAnalyzer",
    "TemporalReasoningAgent",
    "VersionResolver",
    "TemporalReasoningService",
    "TemporalVersionService",
]
