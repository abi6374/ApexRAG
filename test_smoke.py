"""
Smoke test for apex-rag v1.0.6.
Run against the installed wheel (NOT the repo source tree):

    pip install dist/apex_rag-1.0.6-py3-none-any.whl --force-reinstall --no-deps
    python test_smoke.py

Must pass with no exceptions and no ModuleNotFoundError.
"""

import asyncio
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import AsyncGenerator, Any

# Suppress Unicode encoding warnings on Windows consoles
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    warnings.filterwarnings("ignore", message=".*UnicodeEncodeError.*")
    warnings.filterwarnings("ignore", message=".*rich.*")

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class MockProvider:
    """Trivial mock LLM provider -- no API keys needed."""

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        return "Mock answer."

    async def stream_generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        yield "Mock answer."

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


async def main() -> None:
    from apex_rag import ApexIndex

    print("=== apex-rag smoke test ===")

    db_path = Path(tempfile.mkstemp(suffix=".db")[1])
    db_url = f"sqlite+aiosqlite:///{db_path}"

    index = await ApexIndex.create(model=MockProvider(), db_url=db_url)

    try:
        # 1. Ingest a trivial document
        doc_id = await index.ingest_text("# Hello", doc_id="smoke-doc-1")
        print(f"[PASS] ingest_text() -> doc_id={doc_id!r}")

        # 2. List documents
        docs = await index.list_documents()
        assert doc_id in docs, f"Expected {doc_id!r} in {docs}"
        print(f"[PASS] list_documents() -> {docs}")

        # 3. Get stats
        stats = await index.get_stats(doc_id)
        total = stats.get("total_nodes", 0)
        assert total > 0, f"Expected >0 total_nodes, got {stats}"
        print(f"[PASS] get_stats() -> total_nodes={total}")

        # 4. Verify version
        import apex_rag
        print(f"[PASS] apex-rag v{apex_rag.__version__}")

    finally:
        await index.close()
        try:
            db_path.unlink(missing_ok=True)
        except PermissionError:
            pass

    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
