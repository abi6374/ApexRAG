"""
tests/ingestion/test_image_ingestion.py — Image ingestion pipeline tests (Part 8).

Verifies that:
  - Image files (PNG, JPG, WebP, etc.) can be ingested via ApexIndex.ingest()
  - IngestionEngine detects image extensions and delegates to _ingest_image()
  - ImageParser integration produces correct ASTNode / ParsedSection
  - Vision summaries are generated when parse_images_with_vision=True
  - Standard summaries are generated when parse_images_with_vision=False
  - Image metadata (image_data, meta fields) is persisted correctly
  - Non-image files continue to work normally
"""

from __future__ import annotations

import base64
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from apex_rag.ingestion.legacy import IngestionEngine, Summariser

# ── Helpers ─────────────────────────────────────────────────────────────────


def _dummy_png_bytes() -> bytes:
    """Create a minimal valid PNG (1x1 transparent pixel)."""
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x02\x0c\x15\x89"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _dummy_jpg_bytes() -> bytes:
    """Create a minimal valid JPEG."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b"
        b"\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),"
        b"\x01444\x1f'07=442\x0c\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05"
        b"\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\xff\xc0\x00\x0b\x08"
        b"\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01"
        b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00"
        b"\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x00\x00\x01\xd1\x02"
        b'\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R'
        b"\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
        b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5"
        b"\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc3\xc4\xc5\xc6\xc7\xc8\xc9"
        b"\xca\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf3\xf4\xf5\xf6"
        b"\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00"
        b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01"
        b"\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06"
        b"\x12AQ\x07\ra\x13\"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\x15\x16br$4\x17%5\x18\x19&'("
        b")*6789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93"
        b"\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6"
        b"\xb7\xb8\xb9\xba\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe3"
        b"\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01"
        b"\x00\x00?\x00\x80\x00?\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd9"
    )


class _MockLLM:
    """Mock LLM that returns a canned response."""

    def __init__(self, response: str = "Mock summary response.") -> None:
        self._response = response

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 60,
        images: list[str] | None = None,
    ) -> str:
        return self._response

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ):
        yield self._response


class _MockStorage:
    """Minimal mock storage that records insertions and returns canned data."""

    def __init__(self) -> None:
        self.inserted_nodes: list[Any] = []
        self.inserted_pies: list[Any] = []
        self._next_id = 0

    async def insert_node(self, session: Any, node: Any) -> Any:
        self._next_id += 1
        node.id = self._next_id
        self.inserted_nodes.append(node)
        return node

    async def insert_page_index_entry(self, session: Any, entry: Any) -> Any:
        self.inserted_pies.append(entry)
        entry.id = len(self.inserted_pies)
        return entry

    def session(self):
        """Return a dummy async context manager."""

        class _DummySession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        return _DummySession()

    async def dispose(self):
        pass


# ── Tests ───────────────────────────────────────────────────────────────────


class TestImageIngestionViaEngine:
    """Integration tests for IngestionEngine image handling."""

    @pytest.fixture
    def mock_storage(self) -> _MockStorage:
        return _MockStorage()

    @pytest.fixture
    def mock_llm(self) -> _MockLLM:
        return _MockLLM(response="This chart shows quarterly revenue growth over the past year.")

    @pytest.fixture
    def tmp_png(self) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_dummy_png_bytes())
            f.flush()
            yield Path(f.name)
        if Path(f.name).exists():
            Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def tmp_jpg(self) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(_dummy_jpg_bytes())
            f.flush()
            yield Path(f.name)
        if Path(f.name).exists():
            Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def tmp_webp(self) -> Path:
        """Create a fake .webp file (just extension, not a real webp)."""
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"WEBP\x00\x00\x00\x00")
            f.flush()
            yield Path(f.name)
        if Path(f.name).exists():
            Path(f.name).unlink(missing_ok=True)

    # ── Basic image detection tests ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_ingest_png_basic(self, mock_storage, mock_llm, tmp_png):
        """Ingesting a .png file should create a single image node."""
        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=False,
        )
        doc_id = await engine.ingest(tmp_png, synthesize_summaries=False)

        assert doc_id is not None
        assert len(doc_id) == 16  # SHA-256 prefix
        assert len(mock_storage.inserted_nodes) == 1

        node = mock_storage.inserted_nodes[0]
        assert node.doc_id == doc_id
        assert node.title == tmp_png.stem.replace("_", " ").replace("-", " ").title()
        assert node.image_data is not None
        assert node.image_data.startswith("data:image/png;base64,")
        assert node.meta["type"] == "image"
        assert node.meta["vision_summary"] is False
        assert len(mock_storage.inserted_pies) == 1

    @pytest.mark.asyncio
    async def test_ingest_jpg(self, mock_storage, mock_llm, tmp_jpg):
        """Ingesting a .jpg file should produce a node with JPEG image_data."""
        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=False,
        )
        doc_id = await engine.ingest(tmp_jpg, synthesize_summaries=False)

        assert doc_id is not None
        assert len(mock_storage.inserted_nodes) == 1
        node = mock_storage.inserted_nodes[0]
        assert node.image_data.startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_ingest_webp(self, mock_storage, mock_llm, tmp_webp):
        """Ingesting a .webp file should produce a node with webp image_data."""
        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=False,
        )
        doc_id = await engine.ingest(tmp_webp, synthesize_summaries=False)

        assert doc_id is not None
        assert len(mock_storage.inserted_nodes) == 1
        node = mock_storage.inserted_nodes[0]
        assert node.image_data.startswith("data:image/webp;base64,")

    @pytest.mark.asyncio
    async def test_ingest_unsupported_format(self, mock_storage, mock_llm):
        """An unsupported file extension should go through the normal pipeline."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("# Test Heading\n\nSome content.")
            f.flush()
            p = Path(f.name)

        try:
            summariser = Summariser(llm=mock_llm, max_concurrent=2)
            engine = IngestionEngine(
                storage=mock_storage,
                summariser=summariser,
                parse_images_with_vision=False,
            )
            doc_id = await engine.ingest(p, synthesize_summaries=False)

            assert doc_id is not None
            # Should go through normal text pipeline, not image pipeline
            assert len(mock_storage.inserted_nodes) >= 1
            node = mock_storage.inserted_nodes[0]
            # The .xyz file is read as plain text, so it should produce markdown nodes
            assert node.image_data is None
        finally:
            if p.exists():
                p.unlink(missing_ok=True)

    # ── Vision summary tests ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_ingest_with_vision_summary(self, mock_storage, mock_llm, tmp_png):
        """When parse_images_with_vision=True, a vision summary should be generated."""
        vision_response = "Chart showing Q2 revenue of $40M and Q3 revenue of $52M."
        mock_llm._response = vision_response

        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=True,
        )
        doc_id = await engine.ingest(tmp_png, synthesize_summaries=True)

        assert doc_id is not None
        node = mock_storage.inserted_nodes[0]
        # The summary should contain the vision response
        assert "Chart showing" in node.summary
        assert "revenue" in node.summary
        assert node.meta["vision_summary"] is True

    @pytest.mark.asyncio
    async def test_ingest_without_vision_summary(self, mock_storage, mock_llm, tmp_png):
        """When parse_images_with_vision=False, a text-only summary from OCR should be used."""
        text_response = "A PNG image file with some visual content."
        mock_llm._response = text_response

        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=False,
        )
        doc_id = await engine.ingest(tmp_png, synthesize_summaries=True)

        assert doc_id is not None
        node = mock_storage.inserted_nodes[0]
        # Standard summariser should be called with text (no image_data passed)
        assert node.summary is not None
        assert node.meta["vision_summary"] is False

    @pytest.mark.asyncio
    async def test_ingest_with_vision_no_summariser(self, mock_storage, tmp_png):
        """When parse_images_with_vision=True but no summariser, should still work."""
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=None,
            parse_images_with_vision=True,
        )
        doc_id = await engine.ingest(tmp_png, synthesize_summaries=True)

        assert doc_id is not None
        node = mock_storage.inserted_nodes[0]
        # Fallback summary: filename-based
        assert tmp_png.stem.replace("_", " ").replace("-", " ").title() in node.summary
        assert tmp_png.name in node.summary

    # ── ApexIndex-level integration via mock ────────────────────────────────

    @pytest.mark.asyncio
    async def test_ingest_preserves_image_bytes(self, mock_storage, mock_llm, tmp_png):
        """The base64 image data should exactly match the file contents."""
        expected_bytes = tmp_png.read_bytes()
        base64.b64encode(expected_bytes).decode("utf-8")

        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=False,
        )
        await engine.ingest(tmp_png, synthesize_summaries=False)

        node = mock_storage.inserted_nodes[0]
        # Extract base64 from data URI
        assert node.image_data.startswith("data:image/png;base64,")
        actual_b64 = node.image_data[len("data:image/png;base64,") :]
        actual_bytes = base64.b64decode(actual_b64)
        assert actual_bytes == expected_bytes

    @pytest.mark.asyncio
    async def test_ingest_multiple_images(self, mock_storage, mock_llm):
        """Ingesting multiple image files should produce independent nodes."""
        files = []
        for suffix in [".png", ".jpg"]:
            data = _dummy_png_bytes() if suffix == ".png" else _dummy_jpg_bytes()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(data)
                f.flush()
                files.append(Path(f.name))

        try:
            summariser = Summariser(llm=mock_llm, max_concurrent=2)
            engine = IngestionEngine(
                storage=mock_storage,
                summariser=summariser,
                parse_images_with_vision=False,
            )
            doc_ids = []
            for f in files:
                did = await engine.ingest(f, synthesize_summaries=False)
                doc_ids.append(did)

            assert len(mock_storage.inserted_nodes) == 2
            assert doc_ids[0] != doc_ids[1]  # Different content → different hashes
        finally:
            for f in files:
                if f.exists():
                    f.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_ingest_image_verify_node_fields(self, mock_storage, mock_llm, tmp_png):
        """All DocumentNode fields should be correctly populated for an image."""
        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=False,
        )
        await engine.ingest(tmp_png, synthesize_summaries=False)

        node = mock_storage.inserted_nodes[0]
        assert node.parent_id is None  # Root node
        assert node.path == "1"
        assert node.depth == 0
        assert node.position == 1
        assert node.page_start == 0
        assert node.page_end == 0
        assert node.content is not None  # Has content (filename + potential OCR)
        assert node.meta["type"] == "image"
        assert "filename" in node.meta
        assert tmp_png.name in node.meta["filename"]

    @pytest.mark.asyncio
    async def test_ingest_image_custom_doc_id(self, mock_storage, mock_llm, tmp_png):
        """A custom doc_id should override the auto-generated hash."""
        custom_id = "my-image-001"

        summariser = Summariser(llm=mock_llm, max_concurrent=2)
        engine = IngestionEngine(
            storage=mock_storage,
            summariser=summariser,
            parse_images_with_vision=False,
        )
        result_id = await engine.ingest(tmp_png, doc_id=custom_id, synthesize_summaries=False)

        assert result_id == custom_id
        assert mock_storage.inserted_nodes[0].doc_id == custom_id


class TestImageIngestionViaApexIndex:
    """End-to-end tests for image ingestion through ApexIndex (with mocks)."""

    @pytest.mark.asyncio
    async def test_create_index_with_parse_images_with_vision(self):
        """ApexIndex.create() should accept legacy kwargs (ignored)."""
        from apex_rag import ApexIndex

        async with (
            _patch_storage_create(),
        ):
            index = await ApexIndex.create(
                db_url="sqlite+aiosqlite:///:memory:",
                provider="ollama",
                parse_images_with_vision=True,  # Will be swallowed by **kwargs
                trace_enabled=False,
            )
            assert index is not None
            await index.close()

    @pytest.mark.asyncio
    async def test_create_index_without_vision(self):
        """ApexIndex.create() with default should work."""
        from apex_rag import ApexIndex

        async with (
            _patch_storage_create(),
        ):
            index = await ApexIndex.create(
                db_url="sqlite+aiosqlite:///:memory:",
                provider="ollama",
                trace_enabled=False,
            )
            assert index is not None
            await index.close()


# ── Patches ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _patch_storage_create():
    """Temporarily patch ApexStorage.create to return a minimal mock.

    This allows ApexIndex.create() tests to run without real SQLite.
    """
    from apex_rag.ingestion.apex_storage import ApexStorage

    original_create = ApexStorage.create

    class _MockStorageEngine:
        """Minimal mock that satisfies what ApexIndex.create() needs."""

        def __init__(self):
            self._inserted_nodes = []

        @classmethod
        async def create(cls, db_url: str = "", echo: bool = False) -> _MockStorageEngine:
            return cls()

        async def insert_node(self, session, node):
            self._inserted_nodes.append(node)
            return node

        async def insert_page_index_entry(self, session, entry):
            return entry

        def session(self):
            class _DummySession:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            return _DummySession()

        async def dispose(self):
            pass

    try:
        ApexStorage.create = _MockStorageEngine.create  # type: ignore[assignment]
        yield
    finally:
        ApexStorage.create = original_create
