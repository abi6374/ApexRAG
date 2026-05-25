import json
import re

from apex_rag.core.protocols.interfaces import QueryPlanner
from apex_rag.providers import AsyncLLM
from apex_rag.utils import async_retry

_PLANNER_PROMPT = """\
You are an expert Query Planning Agent. Your task is to analyze a complex user query and break it down into a logical sequence of sub-queries that need to be resolved to answer the full question.

User Query: "{query}"

Analyze the query:
1. Is it a simple factual lookup? (1 step)
2. Is it a multi-hop or comparison question? (Break it down)
3. Does it require looking up definitions before fetching metrics?

Respond ONLY with valid JSON in the following format:
{{
  "sub_queries": [
    "Step 1 query",
    "Step 2 query"
  ]
}}
"""

class QueryPlannerAgent(QueryPlanner):
    """
    LLM-backed agent that decomposes complex questions into a list of simpler sub-queries.
    """
    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    @async_retry(max_attempts=3, backoff_base=2.0)
    async def plan(self, query: str) -> list[str]:
        prompt = _PLANNER_PROMPT.format(query=query)
        raw = await self.llm.generate(prompt=prompt, temperature=0.0)

        try:
            match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(raw.strip())
            sub_queries = data.get("sub_queries", [])
            if not isinstance(sub_queries, list) or not sub_queries:
                return [query] # Fallback to original query
            return sub_queries
        except Exception:
            return [query] # Fallback on error
