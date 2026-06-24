"""
providers.py — Pluggable LLM interface for ApexRAG.

By default, ApexRAG uses local Ollama models. However, you can plug in ANY model
(OpenAI, Anthropic, vLLM, custom endpoints) by implementing the `AsyncLLM` protocol.

All third-party provider imports (openai, groq, anthropic) are lazy — they are
imported only when that provider class is instantiated. This keeps the core
`pip install apex-rag` dependency-free of those SDKs.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

# ═══════════════════════════════════════════════════════════════
# LLMProvider Protocol (Part 7: Full interface)
# ═══════════════════════════════════════════════════════════════


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for pluggable LLM providers.

    Full-featured interface with:
    - ``generate()`` — text completion
    - ``stream_generate()`` — token-by-token streaming
    - ``embed()`` — text embeddings (optional; raises ``NotImplementedError``)

    For backward compatibility, ``AsyncLLM`` is kept as an alias.
    """

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        """
        Generate text completion from a prompt.

        Args:
            prompt:      The full text prompt.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens:  Maximum tokens to generate.
            images:      Optional list of base64 image strings.

        Returns:
            The raw text string response.
        """
        ...

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text completion token-by-token.

        Args:
            prompt:      The full text prompt.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens:  Maximum tokens to generate.
            images:      Optional list of base64 image strings.

        Yields:
            Individual content tokens as they arrive.
        """
        ...
        # Convenience: default implementation calls generate() and yields once
        # so consumers don't break.
        yield await self.generate(
            prompt, temperature=temperature, max_tokens=max_tokens, images=images
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate text embeddings for a list of strings.

        Providers that do not support embeddings (Groq, Anthropic)
        should raise ``NotImplementedError``.

        Args:
            texts: List of input strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...
        raise NotImplementedError(
            f"{type(self).__name__} does not support embeddings"
        )


AsyncLLM = LLMProvider


class OllamaProvider:
    """Default provider for local Ollama instances.

    Supports ``generate()``, ``stream_generate()``, and ``embed()``
    via the native Ollama /api/embed and /api/chat streaming endpoints.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        embed_model: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        # Lazy import — ollama is a core dependency always available
        self._ollama = importlib.import_module("ollama")
        self.model = model
        self.embed_model = embed_model or model
        self._client = self._ollama.AsyncClient(host=host, timeout=timeout)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        response: dict[str, Any] = await self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": images}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        msg: dict[str, Any] = response["message"]
        return str(msg["content"])

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        async for part in await self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": images}],
            options={"temperature": temperature, "num_predict": max_tokens},
            stream=True,
        ):
            delta: dict[str, Any] = part.get("message", {})
            content: str = delta.get("content", "")
            if content:
                yield content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama's /api/embed endpoint."""
        response: dict[str, Any] = await self._client.embed(
            model=self.embed_model,
            input=texts,
        )
        return response["embeddings"]  # type: ignore[no-any-return]


class OpenAIProvider:
    """
    Provider for OpenAI API (requires ``pip install openai``).

    Usage::

        llm = OpenAIProvider("gpt-4o-mini", api_key="sk-...")
        await ApexIndex.create(model=llm)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        # Lazy import — openai is an optional dependency
        import openai as openai_mod

        self.model = model
        self._client = openai_mod.AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ── OpenAI-specific ──────────────────────────────────────────────

    _embedding_model: str = "text-embedding-3-small"

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        content: str | list[dict[str, Any]] = prompt
        if images:
            content = [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in images
                ],
            ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        content: str | list[dict[str, Any]] = prompt
        if images:
            content = [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in images
                ],
            ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:  # type: ignore[union-attr]
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via OpenAI's embeddings API."""
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        # Preserve input order
        by_index = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in by_index]


class GroqProvider:
    """
    High-speed provider for Groq API (requires ``pip install groq``).

    Groq does **not** provide an embedding endpoint, so ``embed()``
    raises ``NotImplementedError``.

    Usage::

        llm = GroqProvider("llama3-70b-8192", api_key="gsk_...")
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        api_key: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        # Lazy import — groq is an optional dependency
        import groq as groq_mod

        self.model = model
        self._client = groq_mod.AsyncGroq(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        content: str | list[dict[str, Any]] = prompt
        if images:
            content = [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in images
                ],
            ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        content: str | list[dict[str, Any]] = prompt
        if images:
            content = [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in images
                ],
            ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:  # type: ignore[union-attr]
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Groq does not provide an embedding endpoint."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support embeddings"
        )


class AnthropicProvider:
    """
    Provider for Anthropic Claude API (requires ``pip install anthropic``).

    Usage::

        llm = AnthropicProvider("claude-3-5-sonnet-20240620", api_key="sk-ant-...")
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20240620",
        api_key: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        # Lazy import — anthropic is an optional dependency
        import anthropic as anthropic_mod

        self.model = model
        self._client = anthropic_mod.AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            content.extend(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img},
                }
                for img in images
            )
        msg_payload: list[dict[str, Any]] = [{"role": "user", "content": content}]
        response = await self._client.messages.create(
            model=self.model,
            messages=msg_payload,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content[0].text if response.content else ""  # type: ignore[union-attr]

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            content.extend(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img},
                }
                for img in images
            )
        msg_payload: list[dict[str, Any]] = [{"role": "user", "content": content}]
        async with self._client.messages.stream(
            model=self.model,
            messages=msg_payload,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Anthropic does not provide an embedding endpoint."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support embeddings"
        )


class GeminiProvider:
    """
    Provider for Google Gemini API (requires ``pip install google-generativeai``).

    Usage::

        llm = GeminiProvider("gemini-1.5-flash", api_key="...")
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        # Lazy import — google-generativeai is an optional dependency
        import google.generativeai as genai

        self.model_name = model
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> str:
        # For multi-modal Gemini, images and text are passed in a list
        content: list[Any] = [prompt]
        if images:
            import base64

            for img in images:
                content.append({
                    "mime_type": "image/png",
                    "data": base64.b64decode(img)
                })

        response = await self._model.generate_content_async(
            content,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        return response.text

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        content: list[Any] = [prompt]
        if images:
            import base64
            for img in images:
                content.append({
                    "mime_type": "image/png",
                    "data": base64.b64decode(img)
                })

        response = await self._model.generate_content_async(
            content,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Google's embedding-004 model."""
        import google.generativeai as genai
        response = await genai.embed_content_async(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return response["embedding"]


class OpenRouterProvier:
    """
    Provider for OpenRouter (uses OpenAI-compatible SDK).

    Usage::

        llm = OpenRouterProvier("meta-llama/llama-3-70b-instruct", api_key="...")
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-3-70b-instruct",
        api_key: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        import openai as openai_mod
        self.model = model
        # OpenRouter uses the OpenAI SDK but with a different base URL
        self._client = openai_mod.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,  # noqa: ARG002
    ) -> str:
        # OpenRouter supports standard OpenAI format
        messages = [{"role": "user", "content": prompt}]
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
        images: list[str] | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[str, None]:
        messages = [{"role": "user", "content": prompt}]
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream: # type: ignore
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """OpenRouter typically proxies chat, not embeddings directly in a standard way."""
        raise NotImplementedError("OpenRouter provider does not support embeddings.")
