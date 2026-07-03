# Hybrid Search

ApexRAG's hybrid search combines **three** retrieval strategies for maximum
accuracy. It's the only RAG library that unifies structural, semantic, AND
keyword search in a single agentic pipeline.

## The Three Tiers

```
                 ┌──────────────────────┐
                 │  1. Semantic Cache    │  ← Instant replay of past queries
                 └──────────┬───────────┘
                            ↓ (miss)
                 ┌──────────────────────┐
                 │  2. Vector Search    │  ← Semantic similarity (optional)
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │  3. Keyword (FTS5)   │  ← BM25 full-text search
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │  4. Agentic Nav.     │  ← LLM tree walking (final)
                 └──────────────────────┘
```

## Enabling Hybrid Search

```bash
pip install apex-rag[vectors]
```

```python
from apex_rag import ApexIndex

async with await ApexIndex.create() as index:
    # Setting the domain parameter to 'financial' or 'analytical' automatically 
    # enables hybrid search under the hood!
    answer = await index.query(
        "What was the revenue in Q3?",
        doc_id,
        domain="financial",
    )
```

## How It Works

When hybrid search is enabled, the query flows through:

1. **Vector Similarity** (40% weight) — Sentence embeddings via
   `all-MiniLM-L6-v2` compute semantic similarity between the query and all
   leaf node contents.

2. **Keyword BM25** (30% weight) — Term frequency scoring across title
   (weighted 3×) and content.

3. **Structural Position** (30% weight) — Earlier sections in the document
   get a small positional bonus.

The top-ranked candidates are collected and their section titles are injected
as hints into the agent's navigation prompt, guiding the LLM toward the most
relevant branches.

## Weighted Ranking

You can customize the weights for each component:

```python
from apex_rag.search import HybridSearch

# Retrieve the storage and embedder from ApexIndex internals
storage = index._storage
embedder = index._embedder

searcher = HybridSearch(storage, embeddings=embedder)
ranked = await searcher.hybrid_rank(
    query=query,
    doc_id=doc_id,
    vector_weight=0.4,
    keyword_weight=0.3,
    structural_weight=0.3,
)
```

## Global Hybrid Search

Vector search also powers cross-document retrieval:

```python
answer = await index.query_global(
    "What is our total R&D spend?",
    domain="financial",
)
```

The agent ranks all documents by their root summary embeddings and navigates
the top-3 candidates.

## When to Use Hybrid Search

| Scenario | Vector | Keyword | Agentic | Recommendation |
|----------|--------|---------|---------|----------------|
| Simple fact lookup | ✅ | ✅ | ✅ | Set ``domain="financial"`` for speed |
| Complex analytical query | ❌ | ❌ | ✅ | Agentic-only (default) for precision |
| Large document (100+ pages) | ✅ | ✅ | ✅ | Hybrid with ``domain="analytical"`` |
| No internet/GPU | ❌ | ✅ | ✅ | Agentic + FTS5 (no vectors needed) |

## Performance Impact

Hybrid search adds ~100-500ms for embedding computation but reduces the number
of LLM navigation steps by 30-50% on average, resulting in faster overall
queries for complex documents.
