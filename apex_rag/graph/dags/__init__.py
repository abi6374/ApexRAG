"""
graph/dags/ — Knowledge DAG Builders.

Each builder implements one of the 8 DAG projections:
    - DocumentDagBuilder  → projection=["document"]
    - EntityDagBuilder    → projection=["entity"]
    - CitationDagBuilder  → projection=["citation"]
    - TemporalDagBuilder  → projection=["temporal"]
    - VersionDagBuilder   → projection=["version"]
    - PolicyDagBuilder    → projection=["policy"]
    - FactDagBuilder      → projection=["fact"]
    - ReasoningDagBuilder → projection=["reasoning"]

Builders are deterministic (no LLM calls) and designed to run during
ingestion or as background jobs via the FactPipeline.

All edges are persisted via ApexStorage.save_knowledge_edge() with
the appropriate projection tag.
"""

from apex_rag.graph.dags.citation_dag import CitationDagBuilder
from apex_rag.graph.dags.document_dag import DocumentDagBuilder
from apex_rag.graph.dags.entity_dag import EntityDagBuilder
from apex_rag.graph.dags.fact_dag import FactDagBuilder
from apex_rag.graph.dags.policy_dag import PolicyDagBuilder
from apex_rag.graph.dags.reasoning_dag import ReasoningDagBuilder
from apex_rag.graph.dags.temporal_dag import TemporalDagBuilder
from apex_rag.graph.dags.version_dag import VersionDagBuilder

__all__ = [
    "CitationDagBuilder",
    "DocumentDagBuilder",
    "EntityDagBuilder",
    "FactDagBuilder",
    "PolicyDagBuilder",
    "ReasoningDagBuilder",
    "TemporalDagBuilder",
    "VersionDagBuilder",
]
