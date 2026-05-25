# ApexRAG: Structural AI Retrieval Infrastructure

ApexRAG is designed to eliminate the hallucination and context-loss issues inherent in traditional Vector RAG systems. It achieves this through a deterministic, graph-aware, multi-agent architecture.

## Core Flow: Multi-Agent Structural Reasoning

When a complex query enters the system, it does not immediately hit an embeddings database. Instead, it is routed through the **Orchestrator**:

1.  **Query Planning (`QueryPlannerAgent`)**: The Planner analyzes the query and breaks it down. For example, "Compare Q2 and Q3 metrics" becomes `["Get Q2 metrics", "Get Q3 metrics"]`.
2.  **Deterministic Filtering (`KeywordDeterministicRetriever`)**: Before an LLM is used to navigate, the engine filters the Universal AST down to the top 5 most relevant structural branches using BM25 and heading-overlap scoring.
3.  **Agentic Navigation (`ASTNavigationAgent`)**: The Navigator acts as a tree-walker. It reads the `SemanticModel` (a strict 30-word intent summary of a branch) and decides which path to traverse down the AST.
4.  **Verification (`StrictLeafVerifier`)**: When the Navigator hits a leaf node (e.g., a Paragraph or Table), the Verifier enforces a strict `TRUE/FALSE` check: *Does this node empirically answer the query?*
5.  **Critic Evaluation (`EvaluationCriticAgent`)**: Once the Navigator returns context for all sub-queries, the Critic evaluates the aggregated text to ensure no parts of the original plan were missed.
6.  **Synthesis**: The verified context is synthesized and returned to the user.

## Data Layer: The Universal Document AST

Documents are parsed into a normalized database structure, maintaining strict structural fidelity:

*   **`NodeData`**: Stores the content, type (`Section`, `Paragraph`, `Table`), precise `bounding_box`, and `page_num`. Critically, it enforces `tenant_id` for enterprise data isolation.
*   **`SemanticModelData`**: Maps 1:1 to a `NodeData`. Instead of a raw chunk, this provides the LLM with `intent_tags`, extracted `entities`, and a `concise_summary` to guide navigation.
*   **`GraphEdgeData`**: Turns the hierarchical tree into a Structural Retrieval Graph (SRG) by mapping dependencies between nodes (e.g., a function `CALLS_FUNCTION` another function, or a paragraph `REFERENCES_TABLE` a table).

## Enterprise Scaling

ApexRAG is built for production:
*   **Multi-Tenancy**: Every database transaction and API request is scoped by a `TenantContext`.
*   **Distributed Indexing**: The `DistributedIndexer` protocol allows the computationally heavy AST parsing and semantic map generation to be offloaded to worker queues.
*   **Observability**: Fully integrated with **OpenTelemetry**. Every phase of the Orchestrator loop emits named spans and attributes, allowing tracing of LLM logic in Datadog or Grafana.
