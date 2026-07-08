"""
benchmarks/baselines.py — Unified adapters for industry RAG baselines.

Provides Baseline A (LangChain) and Baseline B (LlamaIndex) implementations
to compare against ApexRAG on HotpotQA and other datasets.

LangChain 1.x compat: uses ``langchain_classic`` for chain API and
``langchain_community`` / ``langchain_openai`` for vector stores and LLMs.
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


@dataclass
class BaselineResult:
    answer: str
    contexts: list[str]


class RAGBaseline(Protocol):
    async def query(self, question: str, text_context: str) -> BaselineResult: ...


# ═══════════════════════════════════════════════════════════════════════
# Baseline A: LangChain (Naive Recursive Chunker)
# ═══════════════════════════════════════════════════════════════════════


class LangChainBaseline:
    """Baseline using standard LangChain components.

    Uses ``langchain_classic.chains.RetrievalQA`` for a standard
    ``Retrieve -> Synthesize`` pipeline with ``FAISS`` vector store.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    async def query(self, question: str, text_context: str) -> BaselineResult:
        # Early exit: if no API key, return a mock result immediately
        # without triggering any OpenAI client initialisation.
        if not os.getenv("OPENAI_API_KEY"):
            return BaselineResult(
                answer=f"Mocked LangChain answer for: {question[:30]}...",
                contexts=[text_context[:200]],
            )

        try:
            from langchain_classic.chains import RetrievalQA
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except (ImportError, RuntimeError) as exc:
            logger.warning("LangChain baseline unavailable (%s). Using mock.", exc)
            return BaselineResult(
                f"Mocked LangChain answer for: {question[:30]}...",
                [text_context[:200]],
            )

        # 1. Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = [Document(page_content=x) for x in splitter.split_text(text_context)]

        # 2. Embed & Index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 3. Retrieve & Synthesize
        llm = ChatOpenAI(model=self.model_name, temperature=0)
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
    """Baseline using LlamaIndex Sentence Window retrieval."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    async def query(self, question: str, text_context: str) -> BaselineResult:
        # Early exit: if no API key, return a mock result immediately.
        if not os.getenv("OPENAI_API_KEY"):
            return BaselineResult(
                answer=f"Mocked LlamaIndex answer for: {question[:30]}...",
                contexts=[text_context[:200]],
            )

        try:
            from llama_index.core import Document, VectorStoreIndex
            from llama_index.core.node_parser import SentenceWindowNodeParser
            from llama_index.core.postprocessor import (
                MetadataReplacementPostProcessor,
            )
            from llama_index.llms.openai import OpenAI
        except (ImportError, RuntimeError) as exc:
            logger.warning("LlamaIndex baseline unavailable (%s). Using mock.", exc)
            return BaselineResult(
                f"Mocked LlamaIndex answer for: {question[:30]}...",
                [text_context[:200]],
            )

        # 1. Parse with Sentence Window
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        llm = OpenAI(model=self.model_name, temperature=0)

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
