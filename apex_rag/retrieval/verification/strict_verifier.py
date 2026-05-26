from apex_rag.core.ast.models import ASTNode
from apex_rag.core.protocols.interfaces import VerificationEngine
from apex_rag.providers import AsyncLLM


class StrictLeafVerifier(VerificationEngine):
    """
    A verification engine that uses an LLM to strictly determine if a node's content
    answers the given query.
    """
    def __init__(self, llm: AsyncLLM):
        self.llm = llm
        self.system_prompt = (
            "You are a strict verification engine. Your ONLY job is to determine if the provided "
            "document text contains the answer to the user's query. "
            "Respond ONLY with 'TRUE' if the exact answer is present, or 'FALSE' if it is not. "
            "Do not explain your reasoning. Do not hallucinate external knowledge."
        )

    async def verify(self, query: str, node: ASTNode) -> bool:
        prompt = (
            f"SYSTEM: {self.system_prompt}\n\n"
            f"Query: {query}\n"
            f"Document Text: {node.content}\n"
            f"Does the Document Text answer the Query? (Reply TRUE or FALSE)"
        )

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.0 # Deterministic evaluation
        )
        # Clean the response to ensure robustness
        clean_resp = response.strip().upper()
        return "TRUE" in clean_resp
