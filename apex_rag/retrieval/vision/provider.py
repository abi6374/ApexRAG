"""
vision/provider.py — Unified vision API (Part 8 Multi-modal).

``VisionAdapter`` wraps any :class:`LLMProvider` and exposes a clean
vision-specific interface for describing, OCR'ing, and classifying images.

Usage::

    from apex_rag.providers import OpenAIProvider
    from apex_rag.retrieval.vision.provider import VisionAdapter

    adapter = VisionAdapter(OpenAIProvider("gpt-4o"))
    description = await adapter.describe_image(base64_data)
    text        = await adapter.extract_text(base64_data)
    category    = await adapter.classify_image(base64_data)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable


# ═══════════════════════════════════════════════════════════════
# VisionProvider Protocol
# ═══════════════════════════════════════════════════════════════


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for vision-capable providers.

    Implementations must be able to accept base64-encoded image data
    and return text descriptions, extracted text, or image classifications.
    """

    async def describe_image(
        self,
        image_data: str,
        prompt: str | None = None,
    ) -> str:
        """Generate a natural-language description of an image.

        Args:
            image_data: Base64-encoded image bytes.
            prompt:     Optional custom prompt override.

        Returns:
            A text description of the image content.
        """
        ...

    async def extract_text(
        self,
        image_data: str,
    ) -> str:
        """Extract / OCR text from an image using vision capabilities.

        Args:
            image_data: Base64-encoded image bytes.

        Returns:
            The text content found in the image.
        """
        ...

    async def classify_image(
        self,
        image_data: str,
    ) -> str:
        """Classify the type of an image.

        Returns one of: ``chart``, ``diagram``, ``photo``, ``screenshot``,
        ``document``, ``table``, ``other``.

        Args:
            image_data: Base64-encoded image bytes.

        Returns:
            A category string.
        """
        ...

    @property
    def is_vision_capable(self) -> bool:
        """Whether the underlying provider supports image inputs."""
        ...

    async def stream_describe(
        self,
        image_data: str,
        prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a description token-by-token.

        Args:
            image_data: Base64-encoded image bytes.
            prompt:     Optional custom prompt override.

        Yields:
            Description tokens as they arrive.
        """
        ...


# ═══════════════════════════════════════════════════════════════
# VisionAdapter — wraps any LLMProvider
# ═══════════════════════════════════════════════════════════════


_DESCRIBE_PROMPT = "Describe this image in detail. What does it show?"
_EXTRACT_TEXT_PROMPT = (
    "Extract ALL text from this image. Return only the extracted text, "
    "preserving line breaks and structure as much as possible."
)
_CLASSIFY_PROMPT = (
    "Classify this image into exactly one category: chart, diagram, photo, "
    "screenshot, document, table, or other. Return only the single word."
)


class VisionAdapter:
    """Wraps an :class:`LLMProvider` with a clean vision-specific API.

    Args:
        llm: Any provider implementing the ``LLMProvider`` protocol
             (OpenAI, Anthropic, Groq, Ollama, etc.).
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    # ── Public API ─────────────────────────────────────────────────────────

    async def describe_image(
        self,
        image_data: str,
        prompt: str | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int = 300,
    ) -> str:
        """Generate a natural-language description of an image.

        Args:
            image_data: Base64-encoded image bytes.
            prompt:     Optional custom prompt (defaults to *describe* prompt).
            temperature: Sampling temperature.
            max_tokens:  Max tokens in the response.

        Returns:
            A text description of the image content.
        """
        return await self._llm.generate(  # type: ignore[no-any-return]
            prompt=prompt or _DESCRIBE_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            images=[image_data],
        )

    async def extract_text(
        self,
        image_data: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        """Extract / OCR text from an image using vision capabilities.

        Args:
            image_data: Base64-encoded image bytes.
            temperature: Sampling temperature (low for deterministic OCR).
            max_tokens:  Max tokens in the response.

        Returns:
            The text content found in the image.
        """
        return await self._llm.generate(  # type: ignore[no-any-return]
            prompt=_EXTRACT_TEXT_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            images=[image_data],
        )

    async def classify_image(
        self,
        image_data: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 20,
    ) -> str:
        """Classify the type of an image.

        Returns one of: ``chart``, ``diagram``, ``photo``, ``screenshot``,
        ``document``, ``table``, ``other``.

        Args:
            image_data: Base64-encoded image bytes.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens:  Max tokens in the response.

        Returns:
            A category string.
        """
        result = await self._llm.generate(
            prompt=_CLASSIFY_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            images=[image_data],
        )
        result = result.strip().lower()
        valid = {"chart", "diagram", "photo", "screenshot", "document", "table", "other"}
        return result if result in valid else "other"

    async def stream_describe(
        self,
        image_data: str,
        prompt: str | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int = 300,
    ) -> AsyncGenerator[str, None]:
        """Stream a description token-by-token.

        Args:
            image_data: Base64-encoded image bytes.
            prompt:     Optional custom prompt override.
            temperature: Sampling temperature.
            max_tokens:  Max tokens in the response.

        Yields:
            Description tokens as they arrive.
        """
        async for token in self._llm.stream_generate(
            prompt=prompt or _DESCRIBE_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            images=[image_data],
        ):
            yield token

    @property
    def is_vision_capable(self) -> bool:
        """Returns ``True`` — all providers with ``images`` param support is assumed.

        .. note::
            Some Groq models or older Ollama models may not support vision.
            Wrap with a check at runtime if needed.
        """
        return True
