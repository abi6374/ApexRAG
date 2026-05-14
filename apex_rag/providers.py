"""
providers.py — Pluggable LLM interface for ApexRAG.

By default, ApexRAG uses local Ollama models. However, you can plug in ANY model
(OpenAI, Anthropic, vLLM, custom endpoints) by implementing the `AsyncLLM` protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
    ) -> str:
        """
        Generate text completion from a prompt.

        Args:
            prompt:      The full text prompt.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens:  Maximum tokens to generate.

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
        timeout: float = 60.0,
    ) -> None:
        import ollama
        self.model = model
        self._client = ollama.AsyncClient(host=host, timeout=timeout)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ) -> str:
        response = await self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]


class OpenAIProvider:
    """
    Example provider for OpenAI API (requires `pip install openai`).
    
    Usage:
        llm = OpenAIProvider("gpt-4o-mini", api_key="sk-...")
        await ApexIndex.create(model=llm)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class GroqProvider:
    """
    High-speed provider for Groq API (requires `pip install groq`).
    
    Usage:
        llm = GroqProvider("llama3-70b-8192", api_key="gsk_...")
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        api_key: str | None = None,
    ) -> None:
        from groq import AsyncGroq
        self.model = model
        self._client = AsyncGroq(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider:
    """
    Provider for Anthropic Claude API (requires `pip install anthropic`).
    
    Usage:
        llm = AnthropicProvider("claude-3-5-sonnet-20240620", api_key="sk-ant-...")
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20240620",
        api_key: str | None = None,
    ) -> None:
        from anthropic import AsyncAnthropic
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ) -> str:
        response = await self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content[0].text if response.content else ""
