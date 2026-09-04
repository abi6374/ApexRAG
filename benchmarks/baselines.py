"""
benchmarks/baselines.py — Unified adapters for industry RAG baselines.

Provides Baseline A (LangChain) and Baseline B (LlamaIndex) implementations
to compare against ApexRAG on HotpotQA and other datasets.

LangChain 1.x compat: uses ``langchain_classic`` for chain API and
``langchain_community`` / ``langchain_openai`` / ``langchain_google_genai``
for vector stores and LLMs.

Both baselines accept a ``provider`` of ``"openai"`` or ``"gemini"`` and
fall back to a clearly-labeled mock only when no matching API key is set
(see :func:`_has_key`) or the required package isn't installed -- they
never silently produce a mock result while claiming to be a real run.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from dataclasses import dataclass
from typing import Protocol

# ── RAGAS import patch (must run before any ragas import) ──────────────
_PATCHED = False


def _patch_ragas_imports() -> None:
    """Inject the missing ``langchain_community.chat_models.vertexai``
    module that RAGAS 0.4.x expects but is absent from modern
    ``langchain-community`` builds."""
    global _PATCHED
    if _PATCHED:
        return
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        m = types.ModuleType("langchain_community.chat_models.vertexai")
        m.__dict__["ChatVertexAI"] = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = m
    _PATCHED = True


# Apply immediately so that subsequent imports of ragas work
_patch_ragas_imports()


logger = logging.getLogger("apex_rag.benchmarks.baselines")

_GEMINI_MODEL = "gemini-2.0-flash"
_GEMINI_EMBED_MODEL = "models/embedding-001"


@dataclass
class BaselineResult:
    answer: str
    contexts: list[str]
    mocked: bool = False


class RAGBaseline(Protocol):
    async def query(self, question: str, text_context: str) -> BaselineResult: ...


def _has_key(provider: str) -> bool:
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    raise ValueError(f"Unknown provider: {provider}")


def _mock_result(system: str, question: str, text_context: str) -> BaselineResult:
    return BaselineResult(
        answer=f"Mocked {system} answer for: {question[:30]}...",
        contexts=[text_context[:200]],
        mocked=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Baseline A: LangChain (Naive Recursive Chunker)
# ═══════════════════════════════════════════════════════════════════════


class LangChainBaseline:
    """Baseline using standard LangChain components.

    Uses ``langchain_classic.chains.RetrievalQA`` for a standard
    ``Retrieve -> Synthesize`` pipeline with a ``FAISS`` vector store.
    LLM/embeddings backend is selected by ``provider`` ("openai" or
    "gemini").
    """

    def __init__(self, provider: str = "openai", model_name: str | None = None):
        self.provider = provider
        self.model_name = model_name or ("gpt-4o-mini" if provider == "openai" else _GEMINI_MODEL)

    def _build_llm_and_embeddings(self):
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

            llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0)
            embeddings = GoogleGenerativeAIEmbeddings(model=_GEMINI_EMBED_MODEL)
            return llm, embeddings
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model=self.model_name, temperature=0)
        embeddings = OpenAIEmbeddings()
        return llm, embeddings

    async def query(self, question: str, text_context: str) -> BaselineResult:
        # Early exit: if no matching API key, return a clearly-labeled mock
        # result immediately, without triggering any client initialisation.
        if not _has_key(self.provider):
            return _mock_result("LangChain", question, text_context)

        try:
            from langchain_classic.chains import RetrievalQA
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            llm, embeddings = self._build_llm_and_embeddings()
        except (ImportError, RuntimeError) as exc:
            logger.warning("LangChain baseline unavailable (%s). Using mock.", exc)
            return _mock_result("LangChain", question, text_context)

        # 1. Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = [Document(page_content=x) for x in splitter.split_text(text_context)]

        # 2. Embed & Index
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 3. Retrieve & Synthesize
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        res = await qa_chain.ainvoke({"query": question})

        return BaselineResult(
            answer=res["result"],
            contexts=[doc.page_content for doc in res["source_documents"]],
        )


# ═══════════════════════════════════════════════════════════════════════
# Baseline B: LlamaIndex (Sentence Window)
# ═══════════════════════════════════════════════════════════════════════


class LlamaIndexBaseline:
    """Baseline using LlamaIndex Sentence Window retrieval.

    LLM/embeddings backend is selected by ``provider`` ("openai" or
    "gemini"). Note: ``llama-index-llms-openai`` / ``llama-index-embeddings-openai``
    and ``llama-index-llms-gemini`` / ``llama-index-embeddings-gemini`` must
    be installed for the corresponding provider -- otherwise this falls
    back to a clearly-labeled mock rather than silently mis-scoring.
    """

    def __init__(self, provider: str = "openai", model_name: str | None = None):
        self.provider = provider
        self.model_name = model_name or ("gpt-4o-mini" if provider == "openai" else _GEMINI_MODEL)

    def _build_llm_and_embed_model(self):
        if self.provider == "gemini":
            from llama_index.embeddings.gemini import GeminiEmbedding
            from llama_index.llms.gemini import Gemini

            llm = Gemini(model=f"models/{self.model_name}", temperature=0)
            embed_model = GeminiEmbedding(model_name=_GEMINI_EMBED_MODEL)
            return llm, embed_model
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        llm = OpenAI(model=self.model_name, temperature=0)
        embed_model = OpenAIEmbedding()
        return llm, embed_model

    async def query(self, question: str, text_context: str) -> BaselineResult:
        # Early exit: if no matching API key, return a clearly-labeled mock.
        if not _has_key(self.provider):
            return _mock_result("LlamaIndex", question, text_context)

        try:
            from llama_index.core import Document, Settings, VectorStoreIndex
            from llama_index.core.node_parser import SentenceWindowNodeParser
            from llama_index.core.postprocessor import (
                MetadataReplacementPostProcessor,
            )

            llm, embed_model = self._build_llm_and_embed_model()
        except (ImportError, RuntimeError) as exc:
            logger.warning("LlamaIndex baseline unavailable (%s). Using mock.", exc)
            return _mock_result("LlamaIndex", question, text_context)

        Settings.embed_model = embed_model

        # 1. Parse with Sentence Window
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        doc = Document(text=text_context)
        index = VectorStoreIndex.from_documents([doc], transformations=[node_parser])

        # 2. Query with Metadata Replacement
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
            llm=llm,
        )

        response = query_engine.query(question)

        return BaselineResult(
            answer=str(response),
            contexts=[n.node.get_content() for n in response.source_nodes],
        )
