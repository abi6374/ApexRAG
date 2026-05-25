from apex_rag.core.ast.models import ASTNode
from apex_rag.core.protocols.interfaces import CriticAgent, QueryPlanner
from apex_rag.retrieval.agentic.navigator import ASTNavigationAgent
from apex_rag.utils import ReasoningTrace, logger


class Orchestrator:
    """
    Coordinates the Planner, Navigator, and Critic for multi-hop graph reasoning.
    """
    def __init__(
        self,
        planner: QueryPlanner,
        navigator: ASTNavigationAgent,
        critic: CriticAgent,
        trace: ReasoningTrace | None = None
    ):
        self.planner = planner
        self.navigator = navigator
        self.critic = critic
        self.trace = trace or ReasoningTrace(enabled=True)

    async def execute_query(self, query: str, doc_id: str) -> str | None:
        """
        Executes the full reasoning loop: Plan -> Navigate -> Critic -> Synthesize.
        """
        # 1. Plan
        logger.info(f"[PLANNING] Breaking down query: '{query}'")
        sub_queries = await self.planner.plan(query)
        logger.info(f"[PLAN] Sub-queries generated: {sub_queries}")

        retrieved_nodes: list[ASTNode] = []

        # 2. Navigate (fetch context for each sub-query)
        for sq in sub_queries:
            logger.info(f"[NAVIGATING] Resolving sub-query: '{sq}'")
            # Reuse the navigator agent from Phase 1, which now acts as our graph walker
            nav_result = await self.navigator.find(query=sq, doc_id=doc_id)

            if nav_result and nav_result.verified:
                logger.info(f"[RETRIEVED] Node {nav_result.node_id} answers '{sq}'")
                # Reconstruct an ASTNode for the Critic
                node = ASTNode(
                    id=nav_result.node_id,
                    node_type="RetrievedContent",
                    content=nav_result.content
                )
                retrieved_nodes.append(node)
            else:
                logger.info(f"[FAILED] Could not resolve sub-query: '{sq}'")

        if not retrieved_nodes:
            return None

        # 3. Critic (evaluate completeness)
        logger.info("[CRITIC] Evaluating retrieved context against all sub-queries...")
        passes = await self.critic.evaluate(sub_queries, retrieved_nodes)

        if not passes:
            logger.info("[CRITIC REJECTED] Retrieved context is insufficient.")
            return None

        logger.info("[CRITIC APPROVED] Context is sufficient.")
        # 4. Synthesize (In a full implementation, this would be another LLM call)
        # For now, we return the combined context as the result.
        final_answer = "\n\n".join(f"--- Context --- \n{n.content}" for n in retrieved_nodes)
        return final_answer
