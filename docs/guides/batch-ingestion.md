# Batch Ingestion

Ingest multiple documents efficiently with ApexRAG's batch API.

## Basic Batch Ingestion

```python
doc_ids = await index.ingest_many([
    ("doc1", "report.pdf"),
    ("doc2", "memo.docx"),
    ("doc3", "# Inline Markdown\nContent here"),
])
```

Each item is a `(doc_id, source)` tuple where `source` is either a file path
or raw text. ApexRAG automatically detects which is which.

## Mixed Files and Text

```python
doc_ids = await index.ingest_many([
    ("financial_report", "Q3_2024_report.pdf"),
    ("meeting_notes", "# Q3 Review\nRevenue grew 28%..."),
    ("engineering_doc", Path("architecture_overview.md")),
])
```

## Parallelism

All items are ingested concurrently using `asyncio.gather`. This is safe
because each ingestion is an independent transaction.

```python
import time

t0 = time.monotonic()
doc_ids = await index.ingest_many([...])  # 10 documents
elapsed = time.monotonic() - t0

print(f"Ingested {len(doc_ids)} documents in {elapsed:.1f}s")
```

## Progress Tracking

For large batches, combine ingestion with your own progress tracking:

```python
sources = [("doc1", "file1.pdf"), ("doc2", "file2.pdf"), ...]
total = len(sources)
completed = 0

for doc_id, source in sources:
    await index.ingest(source, doc_id=doc_id)
    completed += 1
    print(f"[{completed}/{total}] Ingested {doc_id}")

# Or use ingest_many for maximum throughput
doc_ids = await index.ingest_many(sources)
```

## Error Handling

Batch ingestion uses `asyncio.gather` with `return_exceptions=False` by
default, meaning any single failure cancels the remaining tasks. For
resilient batch processing, handle errors per-document:

```python
import asyncio

async def safe_ingest(index, doc_id, source):
    try:
        return await index.ingest(source, doc_id=doc_id)
    except Exception as e:
        print(f"Failed to ingest {doc_id}: {e}")
        return None

tasks = [safe_ingest(index, did, src) for did, src in items]
results = await asyncio.gather(*tasks)
```

## Best Practices

1. **Use unique doc_ids** — Avoid duplicates to prevent confusion.
2. **Batch by group** — Group related documents together for logical organization.
3. **Set concurrency** — Control ingestion parallelism via `max_concurrent_summaries`.
4. **Monitor memory** — Each ingestion holds the Markdown representation in memory.
   For very large documents (>100 pages), ingest them sequentially.
