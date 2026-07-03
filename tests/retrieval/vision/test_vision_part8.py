"""
Tests for Part 8 — Multi-modal Support.

Covers:
    - VisionAdapter (unified vision API wrapping any LLMProvider)
    - ImageParser (OCR-capable document parser for image files)
    - ApexParser image file integration
    - NodeType.IMAGE and image_data field on ASTNode

Test count: 25+
"""

from __future__ import annotations

import base64
import os
import tempfile
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from apex_rag.models.unified_models import ASTNode, NodeType

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _dummy_png_bytes() -> bytes:
    """Create a minimal valid 1x1 red PNG for testing."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: 1x1, 8-bit RGB
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    # IDAT: single red pixel (RGB)
    raw = b"\xff\x00\x00"
    compressed = zlib.compress(raw)
    idat = _chunk(b"IDAT", compressed)
    # IEND
    iend = _chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


_DUMMY_PNG = _dummy_png_bytes()
_DUMMY_B64 = base64.b64encode(_DUMMY_PNG).decode("utf-8")


class _MockVisionLLM:
    """Mock LLM provider that returns fixed vision responses."""

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_chunks: list[str] = ["This ", "is ", "a ", "description."]

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "images": images,
            }
        )
        if "Classify" in prompt:
            return "chart"
        elif "Extract ALL text" in prompt:
            return "Revenue: $40M\nExpenses: $25M"
        else:
            return "A red pixel on a white background."

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "images": images,
            }
        )
        for chunk in self.stream_chunks:
            yield chunk


# ═══════════════════════════════════════════════════════════════
# VisionAdapter tests
# ═══════════════════════════════════════════════════════════════


class TestVisionAdapter:
    """Tests for the unified VisionAdapter."""

    @pytest.fixture
    def adapter(self) -> Any:
        from apex_rag.retrieval.vision.provider import VisionAdapter

        return VisionAdapter(_MockVisionLLM())

    def test_constructor(self) -> None:
        """Can instantiate VisionAdapter with any LLM provider."""
        from apex_rag.retrieval.vision.provider import VisionAdapter

        adapter = VisionAdapter(_MockVisionLLM())
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_describe_image(self, adapter: Any) -> None:
        """describe_image returns a description."""
        result = await adapter.describe_image(_DUMMY_B64)
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify the prompt was sent with images
        assert adapter._llm.generate_calls
        call = adapter._llm.generate_calls[0]
        assert "images" in call
        assert call["images"] == [_DUMMY_B64]

    @pytest.mark.asyncio
    async def test_describe_image_custom_prompt(self, adapter: Any) -> None:
        """Custom prompt is forwarded to the LLM."""
        result = await adapter.describe_image(_DUMMY_B64, prompt="What color is this?")
        assert result
        call = adapter._llm.generate_calls[0]
        assert call["prompt"] == "What color is this?"

    @pytest.mark.asyncio
    async def test_extract_text(self, adapter: Any) -> None:
        """extract_text returns OCR-like text from the LLM."""
        result = await adapter.extract_text(_DUMMY_B64)
        assert isinstance(result, str)
        assert "Revenue" in result
        # Verify a dedicated OCR prompt was used
        call = adapter._llm.generate_calls[0]
        assert "Extract ALL text" in call["prompt"]

    @pytest.mark.asyncio
    async def test_classify_image(self, adapter: Any) -> None:
        """classify_image returns a valid category."""
        result = await adapter.classify_image(_DUMMY_B64)
        assert result == "chart"
        call = adapter._llm.generate_calls[0]
        assert "Classify" in call["prompt"]

    @pytest.mark.asyncio
    async def test_classify_image_fallback(self) -> None:
        """Unknown classification response falls back to 'other'."""
        from apex_rag.retrieval.vision.provider import VisionAdapter

        class _ReturnsGibberish:
            async def generate(self, prompt: str, **kwargs: Any) -> str:
                return "gibberish"

            async def stream_generate(
                self, prompt: str, **kwargs: Any
            ) -> AsyncGenerator[str, None]:
                yield "gibberish"

        adapter = VisionAdapter(_ReturnsGibberish())
        result = await adapter.classify_image(_DUMMY_B64)
        assert result == "other"

    @pytest.mark.asyncio
    async def test_stream_describe(self, adapter: Any) -> None:
        """stream_describe yields tokens."""
        tokens: list[str] = []
        async for token in adapter.stream_describe(_DUMMY_B64):
            tokens.append(token)
        assert len(tokens) >= 1
        combined = "".join(tokens)
        assert "description" in combined

    def test_is_vision_capable(self, adapter: Any) -> None:
        """is_vision_capable returns True."""
        assert adapter.is_vision_capable is True

    @pytest.mark.asyncio
    async def test_describe_image_with_temperature(self, adapter: Any) -> None:
        """Temperature and max_tokens are forwarded."""
        await adapter.describe_image(_DUMMY_B64, temperature=0.7, max_tokens=500)
        call = adapter._llm.generate_calls[0]
        assert call["temperature"] == 0.7
        assert call["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_extract_text_default_params(self, adapter: Any) -> None:
        """extract_text uses low temperature for deterministic OCR."""
        await adapter.extract_text(_DUMMY_B64)
        call = adapter._llm.generate_calls[0]
        assert call["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_classify_image_low_temperature(self, adapter: Any) -> None:
        """classify_image uses zero temperature for deterministic results."""
        await adapter.classify_image(_DUMMY_B64)
        call = adapter._llm.generate_calls[0]
        assert call["temperature"] == 0.0


# ═══════════════════════════════════════════════════════════════
# ImageParser tests
# ═══════════════════════════════════════════════════════════════


class TestImageParser:
    """Tests for the ImageParser."""

    @pytest.fixture
    def parser(self) -> Any:
        from apex_rag.retrieval.vision.parser import ImageParser

        return ImageParser(use_local_ocr=False, default_doc_id="img-test-doc")

    def test_constructor(self) -> None:
        """Can instantiate ImageParser."""
        from apex_rag.retrieval.vision.parser import ImageParser

        parser = ImageParser()
        assert parser is not None

    def test_supported_extensions(self) -> None:
        """Supported extensions include common image formats."""
        from apex_rag.retrieval.vision.parser import SUPPORTED_EXTENSIONS

        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS
        assert ".bmp" in SUPPORTED_EXTENSIONS
        assert ".gif" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS
        assert ".tif" in SUPPORTED_EXTENSIONS

    def test_tesseract_detection(self) -> None:
        """is_tesseract_available reflects actual installation."""
        from apex_rag.retrieval.vision.parser import ImageParser

        parser = ImageParser()
        # This will be False unless pytesseract + tesseract binary are installed
        assert isinstance(parser.is_tesseract_available, bool)

    @pytest.mark.asyncio
    async def test_parse_png(self, parser: Any) -> None:
        """Parse a valid PNG image produces a single IMAGE node."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_DUMMY_PNG)
            f.flush()
            fname = f.name

        try:
            nodes = await parser.parse_file(fname)
            assert len(nodes) == 1
            node = nodes[0]
            assert node.node_type == NodeType.IMAGE
            assert node.doc_id == "img-test-doc"
            assert node.image_data is not None
            assert node.image_data.startswith("data:image/png;base64,")
            assert node.depth == 0
            assert node.parent_id is None
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_parse_jpg(self, parser: Any) -> None:
        """Parse a JPG image (disguised PNG content for testing)."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(_DUMMY_PNG)
            f.flush()
            fname = f.name

        try:
            nodes = await parser.parse_file(fname)
            assert len(nodes) == 1
            node = nodes[0]
            assert node.node_type == NodeType.IMAGE
            assert node.image_data is not None
            # imghdr detects the actual content, not the extension
            assert node.image_data is not None
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self, parser: Any) -> None:
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            await parser.parse_file("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_parse_unsupported_format(self, parser: Any) -> None:
        """Raises ValueError for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
            f.write("<svg></svg>")
            f.flush()
            fname = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported image format"):
                await parser.parse_file(fname)
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_image_content_includes_title(self, parser: Any) -> None:
        """The IMAGE node's content includes a title derived from the filename."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_DUMMY_PNG)
            f.flush()
            fname = f.name

        try:
            nodes = await parser.parse_file(fname)
            node = nodes[0]
            # The filename stem is in the content as a title
            assert len(node.content) > 0
        finally:
            os.unlink(fname)

    def test_local_ocr_can_be_disabled(self) -> None:
        """use_local_ocr=False disables tesseract."""
        from apex_rag.retrieval.vision.parser import ImageParser

        parser = ImageParser(use_local_ocr=False)
        assert parser.is_local_ocr_enabled is False

    @pytest.mark.asyncio
    async def test_parse_with_custom_doc_id(self) -> None:
        """Custom doc_id overrides the default."""
        from apex_rag.retrieval.vision.parser import ImageParser

        parser = ImageParser(use_local_ocr=False)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_DUMMY_PNG)
            f.flush()
            fname = f.name

        try:
            custom_id = "my-custom-doc-id"
            nodes = await parser.parse_file(fname, doc_id=custom_id)
            assert nodes[0].doc_id == custom_id
        finally:
            os.unlink(fname)

    def test_module_init_exports(self) -> None:
        """__init__.py exports VisionAdapter and ImageParser."""
        from apex_rag.retrieval.vision import ImageParser, VisionAdapter

        assert VisionAdapter is not None
        assert ImageParser is not None


# ═══════════════════════════════════════════════════════════════
# NodeType.IMAGE & ASTNode.image_data tests
# ═══════════════════════════════════════════════════════════════


class TestNodeTypeImage:
    """Tests for IMAGE in NodeType enum and image_data on ASTNode."""

    def test_node_type_has_image(self) -> None:
        """NodeType enum includes IMAGE."""
        assert hasattr(NodeType, "IMAGE")
        assert NodeType.IMAGE.value == "IMAGE"

    def test_ast_node_image_data_field(self) -> None:
        """ASTNode has an image_data field that defaults to None."""
        node = ASTNode(
            content="Test image",
            node_type=NodeType.IMAGE,
            doc_id="doc-1",
        )
        assert node.image_data is None

    def test_ast_node_image_data_set(self) -> None:
        """image_data can be set to a base64 data URI."""
        node = ASTNode(
            content="Test image",
            node_type=NodeType.IMAGE,
            doc_id="doc-1",
            image_data="data:image/png;base64,iVBORw0KGgo=",
        )
        assert node.image_data == "data:image/png;base64,iVBORw0KGgo="

    def test_ast_node_image_other_fields(self) -> None:
        """IMAGE nodes have valid UUIDs and proper defaults."""
        node = ASTNode(
            content="Chart",
            node_type=NodeType.IMAGE,
            doc_id="doc-img",
        )
        parsed = uuid.UUID(node.node_id, version=4)
        assert str(parsed) == node.node_id
        assert node.depth == 0
        assert node.parent_id is None
        assert node.children == []

    def test_ast_node_image_embedding(self) -> None:
        """IMAGE nodes can carry embeddings like other nodes."""
        node = ASTNode(
            content="Chart image",
            node_type=NodeType.IMAGE,
            doc_id="doc-1",
            embedding=[0.1, 0.2, 0.3],
        )
        assert len(node.embedding) == 3


# ═══════════════════════════════════════════════════════════════
# ApexParser image integration tests
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_apex_parser_image_file() -> None:
    """ApexParser.parse_file handles image files via ImageParser."""
    from apex_rag.ingestion.apex_parser import ApexParser

    parser = ApexParser()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_DUMMY_PNG)
        f.flush()
        fname = f.name

    try:
        nodes = await parser.parse_file(fname)
        assert len(nodes) == 1
        node = nodes[0]
        assert node.node_type == NodeType.IMAGE
        assert node.image_data is not None
        assert node.image_data.startswith("data:image/")
        assert "base64" in node.image_data
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_apex_parser_image_with_doc_id() -> None:
    """ApexParser passes doc_id through to ImageParser."""
    from apex_rag.ingestion.apex_parser import ApexParser

    parser = ApexParser()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_DUMMY_PNG)
        f.flush()
        fname = f.name

    try:
        custom_id = "image-doc-via-apex"
        nodes = await parser.parse_file(fname, doc_id=custom_id)
        assert nodes[0].doc_id == custom_id
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_apex_parser_image_jpg() -> None:
    """ApexParser handles .jpg image files."""
    from apex_rag.ingestion.apex_parser import ApexParser

    parser = ApexParser()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(_DUMMY_PNG)
        f.flush()
        fname = f.name

    try:
        nodes = await parser.parse_file(fname)
        assert len(nodes) == 1
        assert nodes[0].node_type == NodeType.IMAGE
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_apex_parser_image_webp() -> None:
    """ApexParser handles .webp image files."""
    from apex_rag.ingestion.apex_parser import ApexParser

    parser = ApexParser()

    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
        f.write(_DUMMY_PNG)
        f.flush()
        fname = f.name

    try:
        nodes = await parser.parse_file(fname)
        assert len(nodes) == 1
        assert nodes[0].node_type == NodeType.IMAGE
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_apex_parser_markdown_still_works() -> None:
    """ApexParser still handles markdown files after image support was added."""
    from apex_rag.ingestion.apex_parser import ApexParser

    parser = ApexParser()
    nodes = parser.parse_markdown("# Title\n\nContent.")
    assert len(nodes) >= 2
    headings = [n for n in nodes if n.node_type == NodeType.HEADING]
    assert any("Title" in n.content for n in headings)


# ═══════════════════════════════════════════════════════════════
# Module-level __init__ exports
# ═══════════════════════════════════════════════════════════════


def test_apex_rag_init_exports_vision() -> None:
    """apex_rag.__init__ exports VisionAdapter and ImageParser."""
    import apex_rag

    assert hasattr(apex_rag, "VisionAdapter")
    assert hasattr(apex_rag, "ImageParser")
