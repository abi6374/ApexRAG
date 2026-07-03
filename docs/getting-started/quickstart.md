# Quick Start

Get up and running with ApexRAG in under 5 minutes.

## 1. Install

```bash
pip install apex-rag
```

That's it. Core install has only 5 lightweight dependencies.

## 2. Zero-Dependency Demo

This runs entirely offline — no external services needed:

```python
import asyncio
from apex_rag import ApexIndex

async def demo():
    index = await ApexIndex.create(
        db_url="sqlite+aiosqlite:///:memory:",
        trace_enabled=False,
    )
    try:
        # Ingest raw text — no files, no LLM needed
        doc_id = await index.ingest_text(
            text="""
            # Annual Report 2024
            ## Revenue
            Q3 revenue reached $165M, up 28% year-over-year.
            ## Expenses
            Operating expenses totaled $89M for R&D and $134M for sales.
            """,
            doc_id="demo",
            synthesize_summaries=False,  # Skip LLM summaries
        )

        # Get the tree structure
        tree = await index.get_tree(doc_id)
        print(f"Ingested {len(tree)} nodes!")
        print(f"First node: {tree[0]['title']}")
    finally:
        await index.close()

asyncio.run(demo())
# Output:
# Ingested 5 nodes!
# First node: Annual Report 2024
```

## 3. Full Demo (with Ollama)

For real queries, you'll need [Ollama](https://ollama.com) running locally:

```bash
ollama pull llama3.1
ollama serve
```

```python
import asyncio
from apex_rag import ApexIndex

async def main():
    async with await ApexIndex.create(model="llama3.1") as index:
        # Ingest a document
        doc_id = await index.ingest("quarterly_report.pdf")

        # Query with agentic navigation
        answer = await index.query("What was the Q3 revenue growth?", doc_id)

        print(f"Answer: {answer.answer_text}")
        for packet in answer.evidence_packets:
            print(f"📍 Node: {packet.node_id} | Path: {packet.section_path} | Confidence: {packet.confidence:.2f}")

asyncio.run(main())
```

## 4. Start the API Server

```bash
pip install apex-rag[web]
python -m apex_rag serve
```

Open [http://localhost:8000](http://localhost:8000) for the dashboard and
[http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

## Next Steps

- [Installation Guide](installation.md) — Full installation options including Docker
- [Configuration Guide](configuration.md) — All environment variables
- [Ingesting Documents](../guides/ingestion.md) — Deep dive into the ingestion pipeline
- [Querying with Agentic Navigation](../guides/querying.md) — How navigation works
