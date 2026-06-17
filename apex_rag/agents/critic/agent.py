import json
import re

from apex_rag.core.ast.models import ASTNode
from apex_rag.core.protocols.interfaces import CriticAgent
from apex_rag.providers import AsyncLLM
from apex_rag.utils import async_retry

_CRITIC_PROMPT = """\
You are an expert Critic Agent for an AI retrieval system. Your job is to evaluate whether a set of retrieved document sections provides enough information to answer ALL the required sub-queries.

Sub-Queries to Answer:
{sub_queries_text}

Retrieved Context:
{context_text}

Task:
1. Review the Context.
2. Determine if every single sub-query is fully answered by the Context.
3. If yes, the evaluation passes. If no, the evaluation fails (meaning more retrieval is needed).

Respond ONLY with valid JSON in the following format:
{{
  "passes_evaluation": <true or false>,
  "reason": "<explain which sub-queries are answered and which are missing>"
}}
"""


class EvaluationCriticAgent(CriticAgent):
    """
    LLM-backed agent that evaluates retrieved nodes against planned sub-queries.
    """

    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    @async_retry(max_attempts=3, backoff_base=2.0)
    async def evaluate(self, sub_queries: list[str], nodes: list[ASTNode]) -> bool:
        if not sub_queries:
            return True

        sub_queries_text = "\n".join(f"- {sq}" for sq in sub_queries)

        context_text = ""
        for node in nodes:
            # Handle both CoreASTNode and UnifiedASTNode
            nid = getattr(node, "node_id", getattr(node, "id", "unknown"))
            context_text += f"--- [{nid}] ---\n{node.content}\n"

        prompt = _CRITIC_PROMPT.format(sub_queries_text=sub_queries_text, context_text=context_text)

        raw = await self.llm.generate(prompt=prompt, temperature=0.0)

        try:
            match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(raw.strip())
            return bool(data.get("passes_evaluation", False))
        except Exception:
            # On failure, be conservative and reject
            return False
