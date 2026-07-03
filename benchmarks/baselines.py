"""
benchmarks/baselines.py — Unified adapters for industry RAG baselines.

Provides Baseline A (LangChain) and Baseline B (LlamaIndex) implementations
to compare against ApexRAG.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


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
    """Baseline using standard LangChain components."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    async def query(self, question: str, text_context: str) -> BaselineResult:
        try:
            from langchain.chains import RetrievalQA
            from langchain.docstore.document import Document
            from langchain_community.vectorstores import FAISS
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except (ImportError, RuntimeError) as e:
            print(f"  [!] LangChain baseline skipped due to environment error: {e}")
            return BaselineResult(
                f"Mocked LangChain answer for: {question[:30]}...", [text_context[:200]]
            )

        # 1. Chunk
        # ... (rest of logic)
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = [Document(page_content=x) for x in splitter.split_text(text_context)]

        # 2. Embed & Index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 3. Retrieve & Synthesize
        if not os.getenv("OPENAI_API_KEY"):
            return BaselineResult(
                answer=f"Mocked LangChain answer for: {question[:30]}...",
                contexts=[docs[0].page_content] if docs else [],
            )

        llm = ChatOpenAI(model=self.model_name, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        res = await qa_chain.ainvoke({"query": question})

        return BaselineResult(
            answer=res["result"], contexts=[doc.page_content for doc in res["source_documents"]]
        )


# ═══════════════════════════════════════════════════════════════════════
# Baseline B: LlamaIndex (Sentence Window)
# ═══════════════════════════════════════════════════════════════════════


class LlamaIndexBaseline:
    """Baseline using LlamaIndex Sentence Window retrieval."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    async def query(self, question: str, text_context: str) -> BaselineResult:
        try:
            from llama_index.core import Document, VectorStoreIndex
            from llama_index.core.node_parser import SentenceWindowNodeParser
            from llama_index.core.postprocessor import MetadataReplacementPostProcessor
            from llama_index.llms.openai import OpenAI
        except (ImportError, RuntimeError) as e:
            print(f"  [!] LlamaIndex baseline skipped due to environment error: {e}")
            return BaselineResult(
                f"Mocked LlamaIndex answer for: {question[:30]}...", [text_context[:200]]
            )

        # 1. Parse with Sentence Window
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        llm = OpenAI(model=self.model_name, temperature=0)

        # ServiceContext is deprecated in newer versions, use Settings or explicit args
        # But for compatibility with older envs:
        doc = Document(text=text_context)

        index = VectorStoreIndex.from_documents([doc], transformations=[node_parser])

        # 2. Query with Metadata Replacement
        if not os.getenv("OPENAI_API_KEY"):
            return BaselineResult(
                answer=f"Mocked LlamaIndex answer for: {question[:30]}...",
                contexts=[doc.text[:200]],
            )

        query_engine = index.as_query_engine(
            similarity_top_k=2,
            node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
            llm=llm,
        )

        response = query_engine.query(question)

        return BaselineResult(
            answer=str(response), contexts=[n.node.get_content() for n in response.source_nodes]
        )
