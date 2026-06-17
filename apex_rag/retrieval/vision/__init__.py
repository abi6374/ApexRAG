"""
apex_rag.retrieval.vision — Multi-modal Support (Part 8).

Provides:
  - VisionAdapter:  Unified vision API wrapping any LLMProvider.
  - ImageParser:    OCR-capable document parser for image files.
"""

from __future__ import annotations

from apex_rag.retrieval.vision.provider import VisionAdapter
from apex_rag.retrieval.vision.parser import ImageParser

__all__ = [
    "VisionAdapter",
    "ImageParser",
]
