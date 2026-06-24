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

    All cache keys are tenant-aware (Principle 19) when ``tenant_context``
    is provided.
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

    async def verify(self, query: str, node: ASTNode, tenant_context: str | None = None) -> bool:
        """Verify whether a node contains the answer to a query.

        Args:
            query:           The user's query.
            node:            The AST node to verify.
            tenant_context:  Optional tenant ID for tenant-aware caching.

        Returns:
            True if the node contains the answer.
        """
        leaf_id = getattr(node, "node_id", "") or str(id(node))
        verify_tenant_id = tenant_context or "default"

        # Check VerificationCache first (tenant-aware key — Principle 19)
        cached = await self._verify_cache.get(query, leaf_id, tenant_id=verify_tenant_id)
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

        # Cache the result (tenant-aware key — Principle 19)
        await self._verify_cache.set(query, leaf_id, result, tenant_id=verify_tenant_id)
        return result
