# Ingesting Documents

The ingestion pipeline converts documents into ApexRAG's structural decision tree.

## Pipeline Overview

```
Raw File → Markdown → Parse Sections → Build Tree → Generate Summaries → Persist
```

## Supported Formats

| Format | Extension | Backend |
|--------|-----------|---------|
| PDF | `.pdf` | markitdown, docling |
| Word | `.docx`, `.doc` | markitdown |
| Markdown | `.md` | plaintext |
| Plain Text | `.txt` | plaintext |
| HTML | `.html`, `.htm` | markitdown |
| PowerPoint | `.pptx` | markitdown |
| Excel | `.xlsx` | markitdown |

## Ingesting Files

```python
from apex_rag import ApexIndex

async with await ApexIndex.create() as index:
    # Auto-generated doc_id (SHA-256 hash of file content)
    doc_id = await index.ingest("report.pdf")

    # Or specify your own doc_id
    doc_id = await index.ingest("report.pdf", doc_id="annual_report_2024")
```

## Ingesting Raw Text

For programmatic ingestion or testing:

```python
text = """
# Meeting Notes — Q2 2024

## Action Items
- Finalize Q3 budget by August 15
- Schedule follow-up with engineering team
"""

doc_id = await index.ingest_text(text, doc_id="meeting_notes")
```

## Parser Backends

### MarkItDown (Default)

Fast, reliable for standard documents:

```python
await ApexIndex.create(parser_backend="markitdown")
```

### Docling (Enterprise)

Uses IBM's Docling for advanced OCR, table extraction, and layout-aware parsing.
Ideal for complex PDFs with tables, charts, and multi-column layouts:

```bash
pip install apex-rag[docling]
```

```python
await ApexIndex.create(parser_backend="docling")
```

### Plaintext

Reads files as raw text without conversion. Best for `.md` and `.txt` files:

```python
await ApexIndex.create(parser_backend="plaintext")
```

## Summary Generation

During ingestion, ApexRAG generates 30-word "Semantic Map" summaries for every
section. This is what the navigation agent uses to decide which branch to explore.

```python
# Skip summaries (useful for testing or if LLM is unavailable)
doc_id = await index.ingest("report.pdf", synthesize_summaries=False)

# Control concurrency (tune to your GPU/CPU)
await ApexIndex.create(max_concurrent_summaries=4)
```

The summaries are generated in parallel using a semaphore-bounded pool of LLM
calls, making ingestion of large documents significantly faster.

## Batch Ingestion

Ingest multiple documents in parallel:

```python
doc_ids = await index.ingest_many([
    ("doc1", "report.pdf"),
    ("doc2", "memo.docx"),
    ("doc3", "# Inline Markdown\nContent here"),
])
```

## Large Document Handling

ApexRAG automatically chunks sections that exceed 3000 characters into
sub-sections, ensuring no single leaf node overflows the LLM's context window.
This is transparent to the user — the agent navigates the chunked tree
normally.

## Tree Structure

After ingestion, you can inspect the tree:

```python
tree = await index.get_tree(doc_id)
for node in tree:
    print(f"Node: {node['node_id']} | Type: {node['node_type']} | Depth: {node['depth']}")
```
