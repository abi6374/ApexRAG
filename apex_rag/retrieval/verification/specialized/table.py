from apex_rag.core.ast.models import ASTNode
from apex_rag.core.protocols.interfaces import VerificationEngine
from apex_rag.providers import AsyncLLM


class TableVerifier(VerificationEngine):
    """
    A specialized verification engine that ensures the node is a Table
    and answers the query based on column/row intersection.
    """

    def __init__(self, llm: AsyncLLM):
        self.llm = llm
        self.system_prompt = (
            "You are a specialized table verification engine. Your task is to determine if the provided "
            "table data contains the answer to the user's query by checking column and row intersections. "
            "Respond ONLY with 'TRUE' if the table contains the specific data requested, or 'FALSE' if it does not. "
            "Do not explain your reasoning. Do not hallucinate."
        )

    async def verify(self, query: str, node: ASTNode) -> bool:
        if node.node_type.lower() != "table":
            return False

        prompt = (
            f"SYSTEM: {self.system_prompt}\n\n"
            f"Query: {query}\n"
            f"Table Content:\n{node.content}\n\n"
            f"Does this Table answer the Query? (Reply TRUE or FALSE)"
        )
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        return "TRUE" in response.strip().upper()
