from pydantic import ConfigDict
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from apex_rag.client import ApexIndex


class ApexRAGRetriever(BaseRetriever):
    """
    LangChain integration for ApexRAG.

    This retriever uses ApexRAG's Multi-Agent Orchestrator to find structurally
    verified nodes that answer the query.

    Example:
        ```python
        from apex_rag import ApexIndex
        from apex_rag.integrations.langchain import ApexRAGRetriever

        index = await ApexIndex.create()
        retriever = ApexRAGRetriever(index=index, doc_id="my-document-id")

        docs = retriever.invoke("What is the growth rate?")
        ```
    """

    index: ApexIndex
    doc_id: str
    tenant_id: str = "default"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Sync version of retrieval. Not recommended for ApexRAG."""
        raise NotImplementedError("ApexRAG is natively async. Use ainvoke() or ._aget_relevant_documents()")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Uses ApexRAG's agentic loop to find verified EvidencePackets and
        convert them to LangChain Documents.
        """
        packets = await self.index.retrieve_verified_nodes(
            question=query,
            doc_id=self.doc_id,
            tenant_id=self.tenant_id
        )

        if not packets:
            return []

        lc_docs = []
        for packet in packets:
            lc_docs.append(
                Document(
                    page_content=packet.content,
                    metadata={
                        "node_id": packet.node_id,
                        "source": packet.source_document,
                        "page": packet.page_number,
                        "path": packet.section_path,
                        "confidence": packet.confidence_score,
                    }
                )
            )

        return lc_docs
