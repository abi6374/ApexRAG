"""
vision/parser.py — Image document parser (Part 8 Multi-modal).

``ImageParser`` converts image files (PNG, JPG, WebP, etc.) into
:class:`ASTNode` objects with base64-encoded image data and optional
OCR-extracted text.

Supports two extraction paths:
  1. **Local OCR** — via ``pytesseract`` if installed (fast, works offline).
  2. **LLM Vision OCR** — via any ``VisionAdapter`` / ``LLMProvider``
     (higher quality, requires API).

Usage::

    from apex_rag.retrieval.vision.parser import ImageParser

    parser = ImageParser()
    nodes = await parser.parse_file("chart.png")
    # nodes[0].node_type == NodeType.IMAGE
    # nodes[0].image_data == <base64>
    # nodes[0].content   == <OCR text if available>
"""

from __future__ import annotations

import base64
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from apex_rag.models.unified_models import ASTNode, NodeType

logger = logging.getLogger("apex_rag.vision.parser")


# ═══════════════════════════════════════════════════════════════
# Supported image formats
# ═══════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
}

# Map extensions to MIME types
_EXT_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


# ═══════════════════════════════════════════════════════════════
# ImageParser
# ═══════════════════════════════════════════════════════════════


class ImageParser:
    """Parses image files into :class:`ASTNode` objects.

    Supports all common raster image formats.  The parser can optionally
    use ``pytesseract`` for local OCR, and/or an ``LLMProvider`` for
    higher-quality vision-based text extraction.

    Args:
        ocr_language:     Tesseract language code (default ``eng``).
                          Only used when ``pytesseract`` is installed.
        use_local_ocr:    Attempt local OCR via ``pytesseract`` (default
                          ``True`` if ``pytesseract`` is importable).
        default_doc_id:   Override the auto-generated document ID.
    """

    def __init__(
        self,
        ocr_language: str = "eng",
        use_local_ocr: bool | None = None,
        default_doc_id: str | None = None,
    ) -> None:
        self._ocr_language = ocr_language
        self._default_doc_id = default_doc_id or str(uuid.uuid4())

        # Attempt local OCR import
        self._tesseract_available = False
        try:
            import pytesseract  # noqa: F401
            self._tesseract_available = True
        except ImportError:
            pass

        if use_local_ocr is None:
            use_local_ocr = self._tesseract_available
        self._use_local_ocr = use_local_ocr

    # ── Public API ─────────────────────────────────────────────────────────

    async def parse_file(
        self,
        file_path: str | Path,
        doc_id: str | None = None,
    ) -> list[ASTNode]:
        """Parse a single image file into AST nodes.

        Args:
            file_path: Path to the image file.
            doc_id:    Override the auto-generated document ID.

        Returns:
            A list containing one :class:`ASTNode` with ``node_type=IMAGE``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file is not a supported image format.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        resolved_doc_id = doc_id or self._default_doc_id

        # Read and base64-encode the image
        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect MIME type (use extension-based detection for Python 3.13+ compatibility)
        mime_type = _EXT_MIME.get(ext, "image/png")

        # Build a data URI for downstream use
        data_uri = f"data:{mime_type};base64,{image_b64}"

        # Optional: local OCR to extract text from the image
        ocr_text = ""
        if self._use_local_ocr and self._tesseract_available:
            try:
                import pytesseract
                from PIL import Image as PILImage

                pil_image = PILImage.open(path)
                ocr_text = pytesseract.image_to_string(pil_image, lang=self._ocr_language)
                ocr_text = ocr_text.strip()
                if ocr_text:
                    logger.info(
                        "Local OCR extracted %d chars from %s", len(ocr_text), path.name
                    )
            except Exception as exc:
                logger.warning("Local OCR failed for %s: %s", path.name, exc)

        # Create the AST node
        # Use the image filename as the content "title"
        title = path.stem.replace("_", " ").replace("-", " ").title()

        # Combine OCR text into the content if available
        content_parts = [title]
        if ocr_text:
            content_parts.append("")
            content_parts.append(ocr_text)
        content = "\n".join(content_parts)

        node = ASTNode(
            content=content,
            node_type=NodeType.IMAGE,
            depth=0,
            parent_id=None,
            doc_id=resolved_doc_id,
            source_date=None,
            ingestion_date=datetime.now(timezone.utc),
            image_data=data_uri,
        )

        return [node]

    @property
    def is_tesseract_available(self) -> bool:
        """Whether ``pytesseract`` is installed and importable."""
        return self._tesseract_available

    @property
    def is_local_ocr_enabled(self) -> bool:
        """Whether local OCR is currently enabled."""
        return self._use_local_ocr
