# ApexRAG Architecture

ApexRAG is a **Multi-Agent, Structural Reasoning Engine** that eliminates hallucination and context-loss issues inherent in traditional Vector RAG systems through deterministic, graph-aware, multi-agent architecture.

## Core Pipeline

```
Document (PDF/MD/Code/Image)
        │
        ▼ ApexParser
  Universal AST Nodes (typed ASTNode tree, preserves headings→paragraphs→tables→code)
        │
        ├──► Semantic Signposts (30-word summaries per node for agentic navigation)
        ├──► CausalGraphBuilder (LLM-extracted semantic edges: SUPPORTS, CONTRADICTS, OVERRIDES)
        ├──► 8 Knowledge DAG Builders (deterministic edge generation for each projection)
        └──► ApexStorage (SQLite / PostgreSQL — unified edge store with projection tags)

User Query
        │
        ▼ QueryPlannerAgent  →  ASTNavigationAgent  →  EvaluationCriticAgent
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                    TemporalAuditAgent              ConformalWrapperAgent
                                    (freshness scoring,              (coverage guarantee,
                                     contradiction detection)         prediction set size)
                                              │                               │
                                              └───────────────┬───────────────┘
                                                              ▼
                                                    EvidenceSynthesizerAgent
                                                              │
                                                              ▼
                                                  ApexAnswer + Coverage Guarantee
                                                              │
                                                              ▼
                                                  ReasoningDagBuilder
                                                  (saves trace → KnowledgeEdge store
                                                   with projection=["reasoning"])
```

## Data Model: Unified Knowledge Edge Store

All edges — causal, structural, temporal, and reasoning — use the single `KnowledgeEdge` model:

```python
class KnowledgeEdge(BaseModel):
    edge_id: str          # UUID4
    source_node_id: str   # FK → ASTNode
    target_node_id: str   # FK → ASTNode
    edge_type: EdgeType   # SUPPORTS, CONTRADICTS, REFINES, etc.
    strength: float       # [0, 1]
    evidence: str         # Human-readable justification
    projections: list[str]# DAG membership tags: ["document"], ["entity","citation"], etc.
    metadata: dict        # Arbitrary key-value pairs
```

Each edge can belong to **multiple DAGs** via its `projections` list. For example, a SUPPORTS edge between two fact statements might carry `projections=["fact", "reasoning"]`.

## 8 Knowledge DAG Projections

| DAG | Builder | Edge Types | When Built | Description |
|-----|---------|------------|------------|-------------|
| **DocumentDAG** | `DocumentDagBuilder` | REFINES, SUPPORTS | Ingestion | Structural tree relationships between sections |
| **EntityDAG** | `EntityDagBuilder` | Entity linking edges | Ingestion | Named entity extraction and cross-reference linking |
| **CitationDAG** | `CitationDagBuilder` | Citation edges | Ingestion | Citation and cross-reference detection |
| **TemporalDAG** | `TemporalDagBuilder` | SUCCESSOR, PREDECESSOR, VALID_DURING | Ingestion | Chronological ordering of dated content |
| **VersionDAG** | `VersionDagBuilder` | VERSION_OF, SUPERSEDES, REPLACED_BY, SNAPSHOT_OF | Version creation | Version lineage tracking |
| **PolicyDAG** | `PolicyDagBuilder` | GOVERNS | Ingestion | Policy/regulation extraction from content |
| **FactDAG** | `FactDagBuilder` | SUPPORTS, CONTRADICTS, DEPENDS_ON, SAME_TOPIC | Fact pipeline | Fact relationship extraction and contradiction detection |
| **ReasoningDAG** | `ReasoningDagBuilder` | REASONING_CHAIN, DERIVES_FROM, INFERS, USES | Query time | Orchestrator trace events captured as typed reasoning edges |

All 8 builders are deterministic (no additional LLM calls during building). They analyze existing nodes, edges, and metadata to produce typed relationships.

## Agentic Navigation Flow

When a complex query enters the system, it follows this path:

1. **Query Planning** (`QueryPlannerAgent`) — Decomposes the query into sub-queries (e.g., "Compare Q2 and Q3 metrics" → `["Get Q2 metrics", "Get Q3 metrics"]`).

2. **Deterministic Filtering** (`KeywordDeterministicRetriever`) — Before any LLM call, narrows the AST to the top 5 most relevant structural branches using BM25 and heading-overlap scoring.

3. **Agentic Navigation** (`ASTNavigationAgent`) — Walks the AST tree by reading Semantic Map summaries and deciding which branch to descend.

4. **Verification** (`StrictLeafVerifier`) — At each leaf node, enforces a strict TRUE/FALSE check: *Does this node empirically answer the query?*

5. **Critic Evaluation** (`EvaluationCriticAgent`) — Audits the aggregated context to ensure nothing was missed from the original plan.

6. **Temporal Audit** (`TemporalAuditAgent`) — Scores freshness and detects contradictions across evidence packets (skipped in ablation mode).

7. **Conformal Prediction** (`ConformalWrapperAgent`) — Produces a statistically-grounded coverage guarantee for the answer, *once calibrated*. The threshold defaults to an uncalibrated `0.0` (all packets pass, `coverage_guarantee` reports `0.0`) until `index.enterprise.calibrate_conformal(...)` is run against a held-out labeled set — see the README's Conformal Calibration section.

8. **Synthesis** (`EvidenceSynthesizerAgent`) — Generates the final answer with inline citations.

9. **ReasoningDAG Capture** (`ReasoningDagBuilder`) — Persists all orchestrator trace events as typed reasoning edges in the unified edge store.

## Enterprise Scaling

- **Multi-Tenancy** — Every database transaction and API request is scoped by a `TenantContext`.
- **Temporal Querying** — Versioned node history enables querying documents *as they were* at any point in time.
- **Distributed Ingestion** — The `DistributedIndexer` offloads AST parsing and DAG building to worker queues.
- **Code Intelligence** — `PythonCodeParser` extracts ASTs from `.py` source files for precise code reasoning.
- **OpenTelemetry** — Every agent action emits named spans and attributes, exportable to any OTLP backend.
