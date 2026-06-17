import json
import re
from typing import Any

from apex_rag.core.protocols.interfaces import QueryPlanner
from apex_rag.providers import AsyncLLM
from apex_rag.utils import async_retry

_PLANNER_PROMPT = """\
You are an expert Query Planning Agent. Your task is to analyze a user query and:
1. Classify it into exactly ONE of the following categories:
   - FACTUAL: Simple lookups or fact retrieval.
   - COMPARATIVE: Comparing entities, metrics, or time periods.
   - TEMPORAL: Queries with time context, histories, or chronological order.
   - CAUSAL: Understanding causes, effects, or impact dynamics.
   - LEGAL: Law citations, regulatory rules, or policy conditions.
   - FINANCIAL: Corporate statements, balance sheets, or numerical metric analysis.
   - TECHNICAL: Systems, specifications, or architectural documentation.
   - CODE: Source code references, function/class interactions, or call structures.
   - MULTI_DOCUMENT: Synthesis across multiple distinct files/documents.

2. Decompose the query into a logical sequence of sub-queries.

User Query: "{query}"

Respond ONLY with valid JSON in this format:
{{
  "query_type": "<CATEGORY>",
  "sub_queries": [
    "Sub-query 1",
    "Sub-query 2"
  ],
  "reasoning": "Reasoning for classification and sub-query plan"
}}
"""


class QueryPlannerAgent(QueryPlanner):
    """
    LLM-backed agent that classifies query types and decomposes them into sub-queries.
    """

    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    @async_retry(max_attempts=3, backoff_base=2.0)
    async def plan(self, query: str) -> list[str]:
        # Backward compatibility return (list of subqueries)
        res = await self.plan_query(query)
        return res.get("sub_queries", [query])

    async def plan_query(self, query: str) -> dict[str, Any]:
        """
        Runs the full planning step, returning query_type, sub_queries, and reasoning.
        """
        prompt = _PLANNER_PROMPT.format(query=query)
        raw = await self.llm.generate(prompt=prompt, temperature=0.0)

        fallback = {
            "query_type": "FACTUAL",
            "sub_queries": [query],
            "reasoning": "Fallback query planning applied."
        }

        try:
            match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(raw.strip())

            # Ensure correct format
            if not isinstance(data, dict):
                return fallback

            sub_queries = data.get("sub_queries", [])
            if not isinstance(sub_queries, list) or not sub_queries:
                data["sub_queries"] = [query]

            query_type = data.get("query_type", "FACTUAL").upper()
            valid_types = {
                "FACTUAL", "COMPARATIVE", "TEMPORAL", "CAUSAL", "LEGAL",
                "FINANCIAL", "TECHNICAL", "CODE", "MULTI_DOCUMENT"
            }
            if query_type not in valid_types:
                data["query_type"] = "FACTUAL"

            return data
        except Exception:
            return fallback
