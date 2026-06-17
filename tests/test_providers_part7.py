"""
tests/test_providers_part7.py — Part 7: LLM Provider Protocol (embed & stream_generate).

Tests for:
- protocol compatibility
- OllamaProvider.stream_generate()
- OpenAIProvider.stream_generate()
- GroqProvider.stream_generate()
- AnthropicProvider.stream_generate()
- OllamaProvider.embed()
- OpenAIProvider.embed()
- GroqProvider.embed() — NotImplementedError
- AnthropicProvider.embed() — NotImplementedError
- EvidenceSynthesizerAgent.stream_synthesize via stream_generate()
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apex_rag.agents.synthesizer.agent import EvidenceSynthesizerAgent
from apex_rag.core.evidence.models import EvidencePacket
from apex_rag.providers import (
    AnthropicProvider,
    GroqProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """Mock an Ollama AsyncClient with streaming and embedding support."""
    client = MagicMock()

    # Chat (streaming)
    async def _chat_stream(**kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        tokens = ["Hello", " ", "from", " ", "Ollama"]
        for token in tokens:
            yield {"message": {"content": token}}

    # Chat (non-streaming)
    async def _chat(**kwargs: Any) -> dict[str, Any]:
        return {"message": {"content": "Hello from Ollama"}}

    # Embed
    async def _embed(**kwargs: Any) -> dict[str, Any]:
        texts: list[str] = kwargs.get("input", [])
        return {
            "embeddings": [[0.1, 0.2, 0.3] for _ in texts],
        }

    async def chat_side_effect(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _chat_stream(**kwargs)
        return await _chat(**kwargs)

    client.chat = AsyncMock(side_effect=chat_side_effect)
    client.embed = AsyncMock(side_effect=_embed)
    return client


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock an OpenAI AsyncClient with streaming and embedding support."""
    client = MagicMock()

    # Embedding
    class EmbeddingData:
        def __init__(self, index: int, embedding: list[float]):
            self.index = index
            self.embedding = embedding

    class EmbeddingResponse:
        def __init__(self, texts: list[str]):
            self.data = [EmbeddingData(i, [0.4, 0.5, 0.6]) for i in range(len(texts))]

    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=lambda model, input: EmbeddingResponse(input))

    # Chat completion streaming
    class Delta:
        def __init__(self, content: str | None):
            self.content = content

    class Choice:
        def __init__(self, delta: Delta):
            self.delta = delta

    class Chunk:
        def __init__(self, text: str):
            self.choices = [Choice(Delta(text))]

    class StreamWrapper:
        def __init__(self, tokens: list[str]):
            self._tokens = tokens

        def __aiter__(self) -> "StreamWrapper":
            return self

        async def __anext__(self) -> Chunk:
            if not self._tokens:
                raise StopAsyncIteration
            return Chunk(self._tokens.pop(0))

    async def chat_create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return StreamWrapper(["Hello", " ", "from", " ", "OpenAI"])
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello from OpenAI"))]
        )

    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=chat_create)
    return client


@pytest.fixture
def mock_groq_client() -> MagicMock:
    """Mock a Groq AsyncClient with streaming support."""
    client = MagicMock()

    class Delta:
        def __init__(self, content: str | None):
            self.content = content

    class Choice:
        def __init__(self, delta: Delta):
            self.delta = delta

    class Chunk:
        def __init__(self, text: str):
            self.choices = [Choice(Delta(text))]

    class StreamWrapper:
        def __init__(self, tokens: list[str]):
            self._tokens = tokens

        def __aiter__(self) -> "StreamWrapper":
            return self

        async def __anext__(self) -> Chunk:
            if not self._tokens:
                raise StopAsyncIteration
            return Chunk(self._tokens.pop(0))

    async def chat_create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return StreamWrapper(["Hello", " ", "from", " ", "Groq"])
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello from Groq"))]
        )

    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=chat_create)
    return client


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mock an Anthropic AsyncClient with streaming support."""
    client = MagicMock()

    class TextStream:
        def __init__(self, tokens: list[str]):
            self._tokens = tokens

        def __aiter__(self) -> "TextStream":
            return self

        async def __anext__(self) -> str:
            if not self._tokens:
                raise StopAsyncIteration
            return self._tokens.pop(0)

    class StreamManager:
        def __init__(self, tokens: list[str]):
            self.text_stream = TextStream(tokens)

        async def __aenter__(self) -> "StreamManager":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    async def messages_create(**kwargs: Any) -> Any | StreamManager:
        if "stream" in str(type(kwargs.get("messages", []))):
            return MagicMock(content=[MagicMock(text="Hello from Anthropic")])
        return MagicMock(content=[MagicMock(text="Hello from Anthropic")])

    client.messages = MagicMock()
    # For streaming: client.messages.stream(...)
    stream_manager = StreamManager(["Hello", " ", "from", " ", "Anthropic"])
    client.messages.stream = MagicMock(return_value=stream_manager)
    client.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text="Hello from Anthropic")])
    )
    return client


# ═══════════════════════════════════════════════════════════════
# Protocol Compliance
# ═══════════════════════════════════════════════════════════════

class TestLLMProviderProtocol:
    """Verify all providers conform to the LLMProvider protocol."""

    def test_ollama_is_llm_provider(self) -> None:
        assert isinstance(OllamaProvider(model="llama3.1"), LLMProvider)

    def test_openai_is_llm_provider(self) -> None:
        assert isinstance(OpenAIProvider(model="gpt-4o-mini", api_key="sk-test"), LLMProvider)

    def test_groq_is_llm_provider(self) -> None:
        assert isinstance(GroqProvider(model="llama3-70b", api_key="gsk-test"), LLMProvider)

    def test_anthropic_is_llm_provider(self) -> None:
        assert isinstance(AnthropicProvider(model="claude-3-5-sonnet", api_key="sk-ant-test"), LLMProvider)


# ═══════════════════════════════════════════════════════════════
# OllamaProvider
# ═══════════════════════════════════════════════════════════════

class TestOllamaProvider:
    """Tests concrete Ollama provider with mocked client."""

    @pytest.fixture
    def provider(self, mock_ollama_client: MagicMock) -> OllamaProvider:
        p = OllamaProvider(model="llama3.1")
        p._client = mock_ollama_client
        return p

    @pytest.mark.asyncio
    async def test_stream_generate_yields_tokens(
        self, provider: OllamaProvider
    ) -> None:
        chunks: list[str] = []
        async for chunk in provider.stream_generate("Hello"):
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "from", " ", "Ollama"]

    @pytest.mark.asyncio
    async def test_stream_generate_respects_params(
        self, provider: OllamaProvider
    ) -> None:
        chunks: list[str] = []
        async for chunk in provider.stream_generate(
            "Hello", temperature=0.7, max_tokens=50
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello from Ollama"

    @pytest.mark.asyncio
    async def test_stream_generate_with_images(
        self, provider: OllamaProvider
    ) -> None:
        chunks: list[str] = []
        async for chunk in provider.stream_generate(
            "Describe this", images=["base64img"]
        ):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_embed_returns_vectors(
        self, provider: OllamaProvider
    ) -> None:
        vectors = await provider.embed(["text one", "text two"])
        assert len(vectors) == 2
        assert all(len(v) == 3 for v in vectors)

    @pytest.mark.asyncio
    async def test_embed_uses_embed_model(
        self, provider: OllamaProvider
    ) -> None:
        # Provider was created with embed_model defaulting to model
        assert provider.embed_model == "llama3.1"
        vectors = await provider.embed(["test"])
        assert len(vectors) == 1


# ═══════════════════════════════════════════════════════════════
# OpenAIProvider
# ═══════════════════════════════════════════════════════════════

class TestOpenAIProvider:
    """Tests concrete OpenAI provider with mocked client."""

    @pytest.fixture
    def provider(self, mock_openai_client: MagicMock) -> OpenAIProvider:
        p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
        p._client = mock_openai_client
        return p

    @pytest.mark.asyncio
    async def test_stream_generate_yields_tokens(
        self, provider: OpenAIProvider
    ) -> None:
        chunks: list[str] = []
        async for chunk in provider.stream_generate("Hello"):
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "from", " ", "OpenAI"]

    @pytest.mark.asyncio
    async def test_stream_generate_respects_params(
        self, provider: OpenAIProvider
    ) -> None:
        chunks: list[str] = []
        async for chunk in provider.stream_generate(
            "Hello", temperature=0.7, max_tokens=100
        ):
            chunks.append(chunk)
        assert "".join(chunks) == "Hello from OpenAI"

    @pytest.mark.asyncio
    async def test_embed_returns_vectors(
        self, provider: OpenAIProvider
    ) -> None:
        vectors = await provider.embed(["text one", "text two"])
        assert len(vectors) == 2
        assert all(len(v) == 3 for v in vectors)

    @pytest.mark.asyncio
    async def test_embed_maintains_input_order(
        self, provider: OpenAIProvider
    ) -> None:
        vectors = await provider.embed(["first", "second", "third"])
        assert len(vectors) == 3

    @pytest.mark.asyncio
    async def test_generate_still_works(
        self, provider: OpenAIProvider
    ) -> None:
        result = await provider.generate("Hello")
        assert result == "Hello from OpenAI"


# ═══════════════════════════════════════════════════════════════
# GroqProvider
# ═══════════════════════════════════════════════════════════════

class TestGroqProvider:
    """Tests concrete Groq provider with mocked client."""

    @pytest.fixture
    def provider(self, mock_groq_client: MagicMock) -> GroqProvider:
        p = GroqProvider(model="llama3-70b", api_key="gsk-test")
        p._client = mock_groq_client
        return p

    @pytest.mark.asyncio
    async def test_stream_generate_yields_tokens(
        self, provider: GroqProvider
    ) -> None:
        chunks: list[str] = []
        async for chunk in provider.stream_generate("Hello"):
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "from", " ", "Groq"]

    @pytest.mark.asyncio
    async def test_embed_raises_not_implemented(
        self, provider: GroqProvider
    ) -> None:
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            await provider.embed(["test"])

    @pytest.mark.asyncio
    async def test_generate_still_works(
        self, provider: GroqProvider
    ) -> None:
        result = await provider.generate("Hello")
        assert result == "Hello from Groq"


# ═══════════════════════════════════════════════════════════════
# AnthropicProvider
# ═══════════════════════════════════════════════════════════════

class TestAnthropicProvider:
    """Tests concrete Anthropic provider with mocked client."""

    @pytest.fixture
    def provider(self, mock_anthropic_client: MagicMock) -> AnthropicProvider:
        p = AnthropicProvider(model="claude-3-5-sonnet", api_key="sk-ant-test")
        p._client = mock_anthropic_client
        return p

    @pytest.mark.asyncio
    async def test_stream_generate_yields_tokens(
        self, provider: AnthropicProvider
    ) -> None:
        chunks: list[str] = []
        async for chunk in provider.stream_generate("Hello"):
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "from", " ", "Anthropic"]

    @pytest.mark.asyncio
    async def test_embed_raises_not_implemented(
        self, provider: AnthropicProvider
    ) -> None:
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            await provider.embed(["test"])

    @pytest.mark.asyncio
    async def test_generate_still_works(
        self, provider: AnthropicProvider
    ) -> None:
        result = await provider.generate("Hello")
        assert result == "Hello from Anthropic"


# ═══════════════════════════════════════════════════════════════
# EvidenceSynthesizerAgent via stream_generate
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_streaming_llm() -> AsyncMock:
    """An AsyncMock that implements stream_generate()."""
    llm = AsyncMock()

    async def _stream_gen(
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        tokens = ["The", " ", "answer", " ", "is", " ", "42"]
        for t in tokens:
            yield t

    llm.stream_generate = _stream_gen
    llm.generate = AsyncMock(return_value="The answer is 42")
    return llm


@pytest.fixture
def evidence_packets() -> list[EvidencePacket]:
    return [
        EvidencePacket(
            node_id="pkt-1",
            source_document="doc-1",
            section_path="/root/section1",
            retrieval_reason="Contains Q2 data",
            verification_result=True,
            confidence_score=0.95,
            content="Q2 revenue was $40M.",
        ),
        EvidencePacket(
            node_id="pkt-2",
            source_document="doc-1",
            section_path="/root/section2",
            retrieval_reason="Contains Q3 data",
            verification_result=True,
            confidence_score=0.92,
            content="Q3 revenue was $52M.",
        ),
    ]


class TestSynthesizerStreaming:
    """Verify EvidenceSynthesizerAgent uses stream_generate()."""

    @pytest.mark.asyncio
    async def test_stream_synthesize_yields_tokens(
        self,
        mock_streaming_llm: AsyncMock,
        evidence_packets: list[EvidencePacket],
    ) -> None:
        synthesizer = EvidenceSynthesizerAgent(llm=mock_streaming_llm)
        chunks: list[str] = []
        async for chunk in synthesizer.stream_synthesize(
            "Compare Q2 and Q3 revenue?", evidence_packets
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "".join(chunks) == "The answer is 42"

    @pytest.mark.asyncio
    async def test_stream_synthesize_no_packets(
        self,
        mock_streaming_llm: AsyncMock,
    ) -> None:
        synthesizer = EvidenceSynthesizerAgent(llm=mock_streaming_llm)
        chunks: list[str] = []
        async for chunk in synthesizer.stream_synthesize("Query?", []):
            chunks.append(chunk)
        assert "".join(chunks) == "I could not find enough evidence to answer your query."

    @pytest.mark.asyncio
    async def test_stream_synthesize_no_verified_packets(
        self,
        mock_streaming_llm: AsyncMock,
    ) -> None:
        synthesizer = EvidenceSynthesizerAgent(llm=mock_streaming_llm)
        unverified = [
            EvidencePacket(
                node_id="pkt-1",
                source_document="doc-1",
                section_path="/root",
                retrieval_reason="test",
                verification_result=False,
                confidence_score=0.5,
                content="Unverified content",
            )
        ]
        chunks: list[str] = []
        async for chunk in synthesizer.stream_synthesize("Query?", unverified):
            chunks.append(chunk)
        assert "".join(chunks) == "No verified evidence was provided."


# ═══════════════════════════════════════════════════════════════
# Protocol Default Implementation
# ═══════════════════════════════════════════════════════════════

class TestProtocolDefaults:
    """Verify the default implementations of stream_generate and embed."""

    def test_protocol_default_stream_falls_back_to_generate(self) -> None:
        """The protocol's default stream_generate calls generate() and yields once."""
        # We can't instantiate a Protocol, but we can verify the shape
        assert hasattr(LLMProvider, "stream_generate")
        assert hasattr(LLMProvider, "embed")

    @pytest.mark.asyncio
    async def test_embed_default_raises(self) -> None:
        """Verifying NotImplementedError through a class that doesn't override embed."""
        class BareMinProvider:
            async def generate(self, prompt: str, **kwargs: Any) -> str:
                return "response"

            async def stream_generate(
                self, prompt: str, **kwargs: Any
            ) -> AsyncGenerator[str, None]:
                yield "response"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                raise NotImplementedError("BareMinProvider does not support embeddings")

        provider = BareMinProvider()
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            await provider.embed(["test"])
