from apex_rag.core.ast.models import ASTNode
from apex_rag.core.protocols.interfaces import VerificationEngine
from apex_rag.providers import AsyncLLM


class NumericalVerifier(VerificationEngine):
    """
    A specialized verification engine that ensures the node contains
    the specific numerical quantities or metrics requested in the query.
    """
    def __init__(self, llm: AsyncLLM):
        self.llm = llm
        self.system_prompt = (
            "You are a specialized numerical verification engine. Your task is to identify numerical values "
            "or metrics requested in the query, and strictly verify if the provided document text contains "
            "those exact quantities or metrics. "
            "Respond ONLY with 'TRUE' if the numerical data directly answers the query, or 'FALSE' if it does not. "
            "Do not explain your reasoning. Do not hallucinate."
        )

    async def verify(self, query: str, node: ASTNode) -> bool:
        prompt = (
            f"SYSTEM: {self.system_prompt}\n\n"
            f"Query: {query}\n"
            f"Document Text: {node.content}\n"
            f"Does the Document Text contain the numerical quantities to answer the Query? (Reply TRUE or FALSE)"
        )
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        return "TRUE" in response.strip().upper()
