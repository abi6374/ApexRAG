<p align="center">
  <img src="https://img.shields.io/badge/ApexRAG-v0.2.0--dev-6366f1?style=for-the-badge" alt="ApexRAG">
</p>

<p align="center">
  <strong>The High-Accuracy, Local-First Structural Retrieval Infrastructure.</strong><br>
  <em>Stop guessing with vectors. Start navigating with agents.</em>
</p>

## 🚀 Overview

ApexRAG has evolved from a simple RAG wrapper into a **Multi-Agent, Structural Reasoning Engine** built for enterprise deployments.

Traditional RAG relies on vector proximity, chopping documents into arbitrary chunks and losing crucial structural context (like section headings, table definitions, and document hierarchy), leading to hallucinations. 

ApexRAG solves this by converting documents into a **Universal Document AST (Abstract Syntax Tree)** and using an Orchestrator of specialized LLM Agents (Planner, Navigator, Critic) to explicitly "walk" the document's structure to find exact, verifiable answers.

## 🏗️ The 3-Phase Architecture

### Phase 1: Structural Foundation
- **Universal Document AST:** Documents (PDFs, Markdown, source code) are parsed into strict hierarchical trees (`ASTNode`), preserving exact paragraph-to-heading relationships.
- **Deterministic Retrievers:** Initial filtering uses keyword density, FTS5, and structural heading overlap to locate candidate branches without expensive LLM calls.
- **Strict Verification:** A `StrictLeafVerifier` engine empirically checks if a found node actually answers the query, acting as a firewall against hallucination.

### Phase 2: Structural Reasoning Engine
- **Multi-Agent Orchestrator:** Complex queries are managed by three agents working in concert:
    - **Planner Agent:** Breaks down complex, multi-hop queries (e.g., "Compare Q2 and Q3 revenue") into discrete sub-queries.
    - **Navigator Agent:** Explores the AST tree and Semantic Map signposts to find the correct data for each sub-query.
    - **Critic Agent:** Evaluates the aggregated context to ensure *all* sub-queries were answered before synthesizing the final response.
- **Structural Retrieval Graph (SRG):** Nodes can have `GraphEdge` relations to other nodes (e.g., `REFERENCES_TABLE`), enabling non-linear reasoning.

### Phase 3: Enterprise Ecosystem Platform
- **Multi-Tenant RBAC:** Core SQLAlchemy models (`NodeData`, `SemanticModelData`, `GraphEdgeData`) and FastAPI middlewares strictly enforce data isolation via `tenant_id` boundaries.
- **Distributed Ingestion:** A `DistributedIndexer` protocol allows for massive horizontal scaling of document parsing using Celery or Redis queues.
- **Code Intelligence:** Includes a `PythonCodeParser` that extracts ASTs from source code to enable structural code reasoning.
- **OpenTelemetry:** Every agent action (`[PLANNING]`, `[NAVIGATING]`) is wrapped in distributed traces for production monitoring.

## 📦 Quick Start

```bash
pip install apex-rag
```

```python
import asyncio
from apex_rag import ApexIndex, Orchestrator
from apex_rag.enterprise.auth.models import TenantContext

async def main():
    # Setup Tenant Context
    ctx = TenantContext(tenant_id="corp-abc", user_id="user-1", roles=["admin"])
    
    async with await ApexIndex.create() as index:
        # Ingest preserving structure
        doc_id = await index.ingest("financial_report.md", tenant_id=ctx.tenant_id)
        
        # Multi-Agent Reasoning Query
        # Uses the Planner -> Navigator -> Critic loop internally
        result = await index.orchestrate_query("Compare the revenue between Q2 and Q3", doc_id)
        print(result)

asyncio.run(main())
```

## 📖 Documentation

- [Full Architecture Details](docs/ARCHITECTURE.md)
- [API Reference](docs/api/apex-index.md)

## 📄 License
MIT License.
