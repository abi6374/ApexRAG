"""
basic_usage.py — End-to-end demonstration of ApexRAG.

Prerequisites:
    1. Ollama running locally: `ollama serve`
    2. llama3.1 pulled: `ollama pull llama3.1`
    3. ApexRAG installed: `pip install -e ".[dev]"`

Run with:
    python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import textwrap

from apex_rag.client import ApexIndex

# ---------------------------------------------------------------------------
# Sample document (plain Markdown — no file needed for demo)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = textwrap.dedent("""\
    # Annual Financial Report 2024

    This report summarises the financial performance of Apex Corp for fiscal year 2024.

    ## Executive Summary

    Apex Corp achieved record revenue growth of 34% year-over-year, driven primarily
    by expansion in the Asia-Pacific region and the successful launch of three new
    product lines in Q2 2024.

    ## Revenue Analysis

    ### Q1 2024 Revenue

    Q1 revenue reached $142M, up from $98M in Q1 2023. The increase was primarily
    driven by strong enterprise software sales and new customer acquisitions in EMEA.

    ### Q2 2024 Revenue

    Q2 marked the highest quarterly revenue in company history at $187M. The launch
    of ApexCloud Pro in April 2024 contributed $42M in its first full quarter.

    ### Q3 2024 Revenue

    Q3 revenue was $165M. Growth slowed slightly due to seasonal factors but
    remained 28% above Q3 2023 figures.

    ### Q4 2024 Revenue

    Q4 closed at $198M, completing a record year with total annual revenue of $692M.

    ## Operating Expenses

    ### Research & Development

    R&D expenditure increased to $89M in 2024 (13% of revenue), reflecting the
    company's commitment to innovation and the development of next-generation
    AI-powered features.

    ### Sales & Marketing

    Sales and marketing spend was $134M, a 22% increase from 2023, supporting
    the global expansion strategy and new product launches.

    ## Risk Factors

    ### Market Risks

    Macroeconomic headwinds and currency fluctuations in the Asia-Pacific region
    represent the primary market risk for 2025 planning.

    ### Regulatory Risks

    Evolving AI regulation in the EU and upcoming data sovereignty requirements
    may impact product delivery timelines in those markets.

    ## Outlook for 2025

    Management projects revenue growth of 20–25% for 2025, targeting $830M–$865M,
    supported by the continued expansion of the cloud platform and entry into
    three new geographic markets.
""")

QUERIES = [
    "What was the Q2 revenue figure?",
    "How much did Apex Corp spend on R&D in 2024?",
    "What are the main regulatory risks for the company?",
    "What is the revenue outlook for 2025?",
]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("\n" + "=" * 70)
    print("  ApexRAG — Agentic Document Navigation Demo")
    print("=" * 70 + "\n")

    async with await ApexIndex.create(
        db_url="sqlite+aiosqlite:///apex_demo.db",
        model="llama3.1",
        trace_enabled=True,
    ) as index:

        # --- INGESTION ---
        print("📄 Ingesting sample financial report…\n")
        doc_id = await index.ingest_text(
            SAMPLE_DOCUMENT,
            doc_id="annual-report-2024",
            synthesize_summaries=True,
        )
        print(f"\n✅ Ingested document: doc_id={doc_id}\n")
        print("-" * 70)

        # --- QUERIES ---
        for query in QUERIES:
            print(f"\n🔍 Query: {query}\n")
            result = await index.query(query, doc_id)

            if result:
                print(f"\n📌 Answer Found (node={result.node_id}, path={result.path}):")
                print(f"   {result.content[:300]}")
                print(f"\n   Navigation trace: {' → '.join(t for _, t in result.trace)}")
            else:
                print("   ❌ No answer found for this query.")
            print("\n" + "-" * 70)

        # --- CLEANUP ---
        print("\n🗑️  Cleaning up demo database…")
        deleted = await index.delete(doc_id)
        print(f"   Deleted {deleted} nodes.\n")


if __name__ == "__main__":
    asyncio.run(main())
