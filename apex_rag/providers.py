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
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AsyncLLM(Protocol):
    """
    Protocol for pluggable LLM generation.
    Any class implementing this `generate` method can be passed into ApexIndex.
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


class OllamaProvider:
    """Default provider for local Ollama instances."""

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        # Lazy import — ollama is a core dependency always available
        self._ollama = importlib.import_module("ollama")
        self.model = model
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
    ) -> None:
        # Lazy import — openai is an optional dependency
        import openai as openai_mod
        self.model = model
        self._client = openai_mod.AsyncOpenAI(api_key=api_key, base_url=base_url)

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
                *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images],
            ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class GroqProvider:
    """
    High-speed provider for Groq API (requires ``pip install groq``).

    Usage::

        llm = GroqProvider("llama3-70b-8192", api_key="gsk_...")
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        api_key: str | None = None,
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
                *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images],
            ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


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
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}
                for img in images
            )
        msg_payload: list[dict[str, Any]] = [{"role": "user", "content": content}]
        response = await self._client.messages.create(
            model=self.model,
            messages=msg_payload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content[0].text if response.content else ""
