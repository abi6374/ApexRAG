import json
import re
from typing import Any

from apex_rag.models.unified_models import ASTNode, NodeType
from apex_rag.providers import AsyncLLM


class BaseVerifier:
    """Base specialized verifier interface for ApexRAG V3."""

    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        """
        Verify the node against the query.
        Returns a dict: {"verified": bool, "confidence": float, "reason": str}
        """
        raise NotImplementedError()

    def _parse_llm_response(self, raw: str) -> dict[str, Any]:
        fallback = {
            "verified": False,
            "confidence": 0.0,
            "reason": "Failed to parse verifier response.",
        }
        try:
            match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(raw.strip())

            verified = bool(data.get("verified", False))
            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", ""))

            return {
                "verified": verified,
                "confidence": min(max(confidence, 0.0), 1.0),
                "reason": reason,
            }
        except Exception:
            raw_upper = raw.strip().upper()
            if "TRUE" in raw_upper:
                return {"verified": True, "confidence": 1.0, "reason": "Parsed raw string TRUE."}
            elif "FALSE" in raw_upper:
                return {"verified": False, "confidence": 1.0, "reason": "Parsed raw string FALSE."}
            return fallback


class FactualVerifier(BaseVerifier):
    """Verifies general factual truth and alignment with the query."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        prompt = f"""You are a specialized Factual Verification Engine.
Query: "{query}"
Node Content: "{node.content}"

Determine if the content contains the direct fact matching the query.
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        return self._parse_llm_response(raw)


class NumericalVerifier(BaseVerifier):
    """Verifies numerical calculations, metrics, or quantities."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        prompt = f"""You are a specialized Numerical Verification Engine.
Query: "{query}"
Node Content: "{node.content}"

Determine if the content matches and mathematically answers the numeric request or metrics.
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        return self._parse_llm_response(raw)


class TableVerifier(BaseVerifier):
    """Verifies tabulations, tables, columns and row alignments."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        if (
            node.node_type != NodeType.TABLE
            and "||" not in node.content
            and "\t" not in node.content
        ):
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": "Node is not a structured table.",
            }

        prompt = f"""You are a specialized Table Verification Engine.
Query: "{query}"
Table Content: "{node.content}"

Verify if the table rows and column intersections answer the query.
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        return self._parse_llm_response(raw)


class CitationVerifier(BaseVerifier):
    """Verifies exact matches and cited reference quotes."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        prompt = f"""You are a specialized Citation Verification Engine.
Query: "{query}"
Node Content: "{node.content}"

Does this text offer a direct quote or clear citation matching the query claims?
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        return self._parse_llm_response(raw)


class CodeVerifier(BaseVerifier):
    """Verifies source code syntax, dependency symbols, and calls."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        prompt = f"""You are a specialized Code Verification Engine.
Query: "{query}"
Code Content: "{node.content}"

Does the code syntax, symbol definitions, or call relationships answer the logic query?
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        return self._parse_llm_response(raw)


class TemporalVerifier(BaseVerifier):
    """Verifies chronological timelines, version dates, or histories."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        prompt = f"""You are a specialized Temporal Verification Engine.
Query: "{query}"
Node Content: "{node.content}"

Verify if the text matches the chronological sequence, timestamps, or date constraints of the query.
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        return self._parse_llm_response(raw)


class ConsistencyVerifier(BaseVerifier):
    """Verifies internal logical consistency and lack of contradictions."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        prompt = f"""You are a specialized Consistency Verification Engine.
Query: "{query}"
Node Content: "{node.content}"

Does the content contain any self-contradictory claims or logically inconsistent facts relative to the query?
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation of consistency"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        # Note: verified should be true if it IS consistent, false if inconsistent
        return self._parse_llm_response(raw)


class GraphVerifier(BaseVerifier):
    """Verifies nodes based on relationship logic and path contexts."""

    async def verify(self, query: str, node: ASTNode) -> dict[str, Any]:
        prompt = f"""You are a specialized Graph Path Verification Engine.
Query: "{query}"
Node Content: "{node.content}"

Check if this node answers query relations, dependency hierarchies, or path logics.
Respond ONLY with a valid JSON:
{{
  "verified": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "Brief explanation"
}}
"""
        raw = await self.llm.generate(prompt, temperature=0.0)
        return self._parse_llm_response(raw)


class VerifierRegistry:
    """Manages routing of queries to specialized verification engines."""

    def __init__(self, llm: AsyncLLM):
        self.verifiers: dict[str, BaseVerifier] = {
            "factual": FactualVerifier(llm),
            "numerical": NumericalVerifier(llm),
            "table": TableVerifier(llm),
            "citation": CitationVerifier(llm),
            "code": CodeVerifier(llm),
            "temporal": TemporalVerifier(llm),
            "consistency": ConsistencyVerifier(llm),
            "graph": GraphVerifier(llm),
        }

    def get_verifier(self, name: str) -> BaseVerifier:
        return self.verifiers.get(name.lower(), self.verifiers["factual"])
