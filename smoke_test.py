#!/usr/bin/env python3
"""Smoke test: verify ingestion works after tenant_context fixes, and no duplicate index warning on fresh DB."""

import asyncio
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Capture stderr during DB creation
stderr_buf = io.StringIO()
old_stderr = sys.stderr
sys.stderr = stderr_buf

logging.basicConfig(level=logging.WARNING)

# Force UTF-8 on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "ignore")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "ignore")


async def main() -> int:
    from apex_rag.client import ApexIndex
    from unittest.mock import AsyncMock, MagicMock

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Mock summary.")

    async def mock_stream_generate(*args, **kwargs):
        yield "Mock "
        yield "response"

    mock_llm.stream_generate = mock_stream_generate

    async def mock_embed(texts, **kwargs):
        import random
        return [[random.uniform(-1.0, 1.0) for _ in range(384)] for _ in texts]

    mock_llm.embed = AsyncMock(side_effect=mock_embed)

    # Remove old smoke DB if present
    db_path = Path("smoke_test.db")
    for p in [db_path, Path("smoke_test.db-shm"), Path("smoke_test.db-wal")]:
        if p.exists():
            p.unlink()

    try:
        # Create index
        print(">> Creating ApexIndex with fresh database...")
        index = await ApexIndex.create(
            model=mock_llm,
            db_url="sqlite+aiosqlite:///smoke_test.db",
            trace_enabled=False,
        )

        # Ingest a simple markdown document
        test_md = "# Test Document\n\n## Section 1\nContent.\n\n## Section 2\nPolicy content.\n\n## Section 3\nSee Section 1."
        print(">> Ingesting test document...")
        doc_id = await index.ingest_text(test_md, doc_id="smoke-test-doc")
        assert doc_id == "smoke-test-doc"
        print("  [OK] ingest_text() succeeded")

        # List documents
        docs = await index.list_documents()
        assert "smoke-test-doc" in docs
        print("  [OK] list_documents()")

        # Get stats
        stats = await index.get_stats(doc_id)
        assert stats["total_nodes"] >= 3
        print("  [OK] get_stats(): %d nodes" % stats["total_nodes"])

        # Get page index
        entries = await index.get_page_index(doc_id)
        print("  [OK] get_page_index(): %d entries" % len(entries))

        await index.close()
        print()
        print("ALL SMOKE TESTS PASSED!")
        return 0

    except Exception as e:
        print()
        print("SMOKE TEST FAILED: %s" % e)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore stderr
        sys.stderr = old_stderr
        stderr_output = stderr_buf.getvalue()
        if "duplicate index" in stderr_output.lower():
            print()
            print("WARNING: duplicate index warning detected!")
            for line in stderr_output.splitlines():
                if line.strip() and ("duplicate" in line.lower() or "already" in line.lower()):
                    print("  %s" % line.strip())
        elif stderr_output:
            print()
            print("DB creation logs (first 5 lines):")
            for i, line in enumerate(stderr_output.splitlines()):
                if i >= 5:
                    break
                if line.strip():
                    print("  %s" % line.strip())

        # Cleanup
        for p in [db_path, Path("smoke_test.db-shm"), Path("smoke_test.db-wal")]:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
