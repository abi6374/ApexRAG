from apex_rag.core.ast.models import ASTNode
from apex_rag.core.cache import VerificationCache
from apex_rag.core.protocols.interfaces import VerificationEngine
from apex_rag.observability.metrics_service import metrics_service
from apex_rag.providers import AsyncLLM


class StrictLeafVerifier(VerificationEngine):
    """
    A verification engine that uses an LLM to strictly determine if a node's content
    answers the given query.

    Uses :class:`VerificationCache` to avoid redundant LLM calls for the same
    (query, leaf_id) combination.  Cache hits skip the LLM entirely.
    """

    def __init__(self, llm: AsyncLLM):
        self.llm = llm
        self._verify_cache = VerificationCache()
        self.system_prompt = (
            "You are a strict verification engine. Your ONLY job is to determine if the provided "
            "document text contains the answer to the user's query. "
            "Respond ONLY with 'TRUE' if the exact answer is present, or 'FALSE' if it is not. "
            "Do not explain your reasoning. Do not hallucinate external knowledge."
        )

    async def verify(self, query: str, node: ASTNode) -> bool:
        # Check VerificationCache first
        leaf_id = getattr(node, "node_id", "") or str(id(node))
        cached = await self._verify_cache.get(query, leaf_id)
        if cached is not None:
            metrics_service.record_cache_hit()
            return cached

        metrics_service.record_cache_miss()
        prompt = (
            f"SYSTEM: {self.system_prompt}\n\n"
            f"Query: {query}\n"
            f"Document Text: {node.content}\n"
            f"Does the Document Text answer the Query? (Reply TRUE or FALSE)"
        )

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.0,  # Deterministic evaluation
        )
        metrics_service.increment_llm_calls()
        # Clean the response to ensure robustness
        clean_resp = response.strip().upper()
        result = "TRUE" in clean_resp

        # Cache the result
        await self._verify_cache.set(query, leaf_id, result)
        return result
