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
    - HistoricalStateEngine:      Delta computation and state traversal over time
    - SnapshotEngine:             Lazy snapshot construction and caching
"""
from apex_rag.temporal.analyzers import ChangeAnalyzer, TrendAnalyzer
from apex_rag.temporal.state_reconstructor import StateReconstructor
from apex_rag.temporal.temporal_agent import TemporalReasoningAgent
from apex_rag.temporal.temporal_retriever import TemporalRetriever
from apex_rag.temporal.version_resolver import VersionResolver
from apex_rag.temporal.reasoning_service import TemporalReasoningService
from apex_rag.temporal.version_service import TemporalVersionService
from apex_rag.temporal.fact_store import FactStore, TemporalFact, FactRow
from apex_rag.temporal.fact_extractor import FactExtractor
from apex_rag.temporal.fact_lineage import FactLineageEngine, LineageValidator
from apex_rag.temporal.fact_validity import FactValidityResolver
from apex_rag.temporal.snapshot_models import SnapshotDelta, SnapshotManifest, StatePatch
from apex_rag.temporal.historical_state import HistoricalStateEngine
from apex_rag.temporal.snapshot_engine import SnapshotEngine
from apex_rag.temporal.fact_contradiction import (
    ContradictionReport,
    ContradictionType,
    FactContradiction,
    FactContradictionDetector,
    Severity,
)
from apex_rag.temporal.consistency import (
    CheckSeverity,
    CheckType,
    ConsistencyVerifier,
    VerificationIssue,
    VerificationReport,
)
# Sprint 6 — Chain Reconciliation
from apex_rag.temporal.chain_reconciler import (
    AnomalyType,
    ChainAnomaly,
    ChainDiagnosticReport,
    ChainGapDetector,
    ChainReconciliationReport,
    CrossChainStateReconstructor,
    ReconciledChain,
    VersionChainReconciler,
)

__all__ = [
    "TemporalRetriever",
    "StateReconstructor",
    "ChangeAnalyzer",
    "TrendAnalyzer",
    "TemporalReasoningAgent",
    "VersionResolver",
    "TemporalReasoningService",
    "TemporalVersionService",
    # Sprint 2 — Fact Layer
    "FactStore",
    "TemporalFact",
    "FactRow",
    "FactExtractor",
    "FactLineageEngine",
    "LineageValidator",
    # Sprint 3 — Fact Validity
    "FactValidityResolver",
    # Sprint 4 — Snapshots & State History
    "SnapshotDelta",
    "SnapshotManifest",
    "StatePatch",
    "HistoricalStateEngine",
    "SnapshotEngine",
    # Sprint 5 — Contradiction Detection & Consistency
    "FactContradictionDetector",
    "ContradictionType",
    "Severity",
    "FactContradiction",
    "ContradictionReport",
    "ConsistencyVerifier",
    "CheckType",
    "CheckSeverity",
    "VerificationIssue",
    "VerificationReport",
    # Sprint 6 — Chain Reconciliation
    "ChainGapDetector",
    "AnomalyType",
    "ChainAnomaly",
    "ChainDiagnosticReport",
    "VersionChainReconciler",
    "ReconciledChain",
    "ChainReconciliationReport",
    "CrossChainStateReconstructor",
]
