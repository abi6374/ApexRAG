import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from apex_rag.models.unified_models import ASTNode as UnifiedASTNode
from apex_rag.core.protocols.interfaces import DeterministicRetriever, VerificationEngine
from apex_rag.providers import AsyncLLM
from apex_rag.ingestion.apex_storage import ApexStorage, ASTNodeRow
from apex_rag.utils import ReasoningTrace

logger = logging.getLogger("apex_rag.retrieval.agentic.navigator")


@dataclass
class ASTNavigationResult:
    """Result of traversing the Universal Document AST."""

    node: UnifiedASTNode
    path: str
    title: str
    trace: list[tuple[str, str]]
    verified: bool
    confidence: float

    @property
    def content(self) -> str:
        return self.node.content

    @property
    def node_id(self) -> str:
        return self.node.node_id


_NAVIGATE_PROMPT = """\
You are a precise document navigator for a search engine.

User Query: "{query}"

Below are the sub-sections available. Each entry shows the section ID, its type, and its summary:

{children_text}

Task:
Identify the section(s) most likely to contain the answer to the query.

Rules:
- Return the BEST matching section ID as "chosen_id".
- Optionally return a second-best "fallback_id" if uncertain.
- If NO section is relevant, set both to null.

Respond ONLY with valid JSON.
{{"chosen_id": "<string or null>", "fallback_id": "<string or null>", "reason": "<why>"}}
"""


class ASTNavigationAgent:
    """
    Phase 1 Universal AST Navigation Agent.
    Uses Deterministic pre-filtering before invoking LLM logic.
    """

    def __init__(
        self,
        storage: ApexStorage,
        model: AsyncLLM,
        retriever: DeterministicRetriever,
        verifier: VerificationEngine,
        trace: ReasoningTrace | None = None,
    ):
        self._storage = storage
        self._model = model
        self._retriever = retriever
        self._verifier = verifier
        self._trace = trace or ReasoningTrace(enabled=True)

    async def find(
        self, query: str, doc_id: str, root_node_id: str | None = None
    ) -> ASTNavigationResult | None:
        async with self._storage.session() as session:
            # 1. Fetch root nodes from DB
            db_nodes = await self._storage.get_ast_children(
                session, parent_id=root_node_id, doc_id=doc_id
            )
            if not db_nodes:
                return None

            # Convert DB roots to UnifiedASTNode objects
            ast_roots = [await self._db_to_ast(session, node) for node in db_nodes]

            traversal_trace: list[tuple[str, str]] = []
            visited: set[str] = set()

            for root in ast_roots:
                result = await self._navigate(
                    query=query,
                    session=session,
                    current_node=root,
                    traversal_trace=traversal_trace,
                    visited=visited,
                )
                if result:
                    return result

            return None

    async def _navigate(
        self,
        query: str,
        session: Any,
        current_node: UnifiedASTNode,
        traversal_trace: list[tuple[str, str]],
        visited: set[str],
    ) -> ASTNavigationResult | None:
        if current_node.node_id in visited:
            return None
        visited.add(current_node.node_id)

        # Title is usually the first line or first 100 chars
        title = current_node.content[:100].split("\n")[0]
        traversal_trace.append((current_node.node_id, title))

        # Determine if Leaf (no children)
        if not current_node.children:
            # Phase 1 Verification
            is_verified = await self._verifier.verify(query, current_node)
            if not is_verified:
                return None

            return ASTNavigationResult(
                node=current_node,
                path="",  # Path building to be added
                title=title,
                trace=list(traversal_trace),
                verified=True,
                confidence=1.0,
            )

        # It's an internal node. Pre-filter candidates deterministically.
        candidates = await self._retriever.retrieve(query, current_node, top_k=5)

        if not candidates:
            logger.debug("[NAVIGATE] No candidates found for node %s", current_node.node_id)
            return None

        # Load candidate summaries (signposts are now in content)
        candidate_texts = ""
        for cand in candidates:
            # Signpost is usually the first 150 chars/prepended block
            summary = cand.content[:150].replace("\n", " ")
            candidate_texts += f"[{cand.node_id}] {cand.node_type}: {summary}\n"

        logger.debug("[NAVIGATE] Node %s has %d candidates", current_node.node_id, len(candidates))
        prompt = _NAVIGATE_PROMPT.format(query=query, children_text=candidate_texts)
        raw = await self._model.generate(prompt=prompt, temperature=0.0, max_tokens=150)

        chosen_id, fallback_id = self._parse_json_ids(raw)
        logger.debug("[NAVIGATE] LLM chose: %s (fallback: %s)", chosen_id, fallback_id)

        # Execute navigation
        for cid in [chosen_id, fallback_id]:
            if not cid:
                continue

            # Find the actual candidate node
            child_node = next((c for c in candidates if c.node_id == cid), None)
            if child_node:
                res = await self._navigate(query, session, child_node, traversal_trace, visited)
                if res:
                    return res

        return None

    def _parse_json_ids(self, raw: str) -> tuple[str | None, str | None]:
        try:
            match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(raw.strip())
            return data.get("chosen_id"), data.get("fallback_id")
        except Exception:
            return None, None

    async def _db_to_ast(self, session: Any, db_node: ASTNodeRow) -> UnifiedASTNode:
        """Convert a DB ASTNodeRow (and its children) into a UnifiedASTNode."""
        # For full retrieval, we recursively load.
        children_db = await self._storage.get_ast_children(session, db_node.node_id, db_node.doc_id)
        children_ast = [await self._db_to_ast(session, c) for c in children_db]

        return UnifiedASTNode(
            node_id=db_node.node_id,
            node_type=db_node.node_type,
            content=db_node.content,
            parent_id=db_node.parent_id,
            children=children_ast,  # Store full node objects for navigation
            doc_id=db_node.doc_id,
            depth=db_node.depth or 0,
        )
