from typing import Any, Protocol

from apex_rag.core.ast.models import ASTNode


class DocumentParser(Protocol):
    """
    Protocol for Document Parsers.
    Any parser (Docling, MarkItDown, PyMuPDF) must implement this interface.
    It takes a file path and returns the root ASTNode of the parsed document.
    """
    async def parse(self, file_path: str, **kwargs: Any) -> ASTNode:
        ...

class DeterministicRetriever(Protocol):
    """
    Protocol for Deterministic Pre-Filtering.
    Takes a query and a starting ASTNode (usually the root) and returns
    a list of high-scoring candidate ASTNodes based on non-LLM methods 
    (like FTS5, BM25, heading overlap).
    """
    async def retrieve(self, query: str, root_node: ASTNode, top_k: int = 5) -> list[ASTNode]:
        ...

class VerificationEngine(Protocol):
    """
    Protocol for the strict leaf verifier.
    Asks the LLM if a specific node strictly answers the query.
    """
    async def verify(self, query: str, node: ASTNode) -> bool:
        ...

class QueryPlanner(Protocol):
    """
    Protocol for the Query Planning Agent.
    Breaks down a complex query into sub-queries.
    """
    async def plan(self, query: str) -> list[str]:
        ...

class CriticAgent(Protocol):
    """
    Protocol for the Critic Agent.
    Evaluates retrieved nodes against the planned sub-queries.
    """
    async def evaluate(self, sub_queries: list[str], nodes: list[ASTNode]) -> bool:
        ...
