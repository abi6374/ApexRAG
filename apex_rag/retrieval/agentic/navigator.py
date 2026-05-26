import json
import re
from dataclasses import dataclass
from typing import Any

from apex_rag.core.ast.models import ASTNode
from apex_rag.core.protocols.interfaces import DeterministicRetriever, VerificationEngine
from apex_rag.providers import AsyncLLM
from apex_rag.storage import NodeData, StorageEngine
from apex_rag.utils import ReasoningTrace


@dataclass
class ASTNavigationResult:
    """Result of traversing the Universal Document AST."""
    content: str
    node_id: str
    path: str
    title: str
    trace: list[tuple[str, str]]
    verified: bool
    confidence: float

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
        storage: StorageEngine,
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
        self,
        query: str,
        doc_id: str,
        root_node_id: str | None = None
    ) -> ASTNavigationResult | None:
        async with self._storage.session() as session:
            # 1. Fetch root nodes from DB
            db_nodes = await self._storage.get_ast_children(session, parent_id=root_node_id, doc_id=doc_id)
            if not db_nodes:
                return None

            # Convert DB roots to ASTNode objects for the Retriever
            ast_roots = [await self._db_to_ast(session, node) for node in db_nodes]

            # In this architecture, instead of assuming a single strict root,
            # we can pass the children as a pseudo-root to the retriever,
            # or pre-filter candidates dynamically.
            # To simplify, we will navigate each root sequentially, but we'll use
            # deterministic filtering within _navigate.

            traversal_trace: list[tuple[str, str]] = []
            visited: set[str] = set()

            for root in ast_roots:
                result = await self._navigate(
                    query=query,
                    session=session,
                    current_node=root,
                    traversal_trace=traversal_trace,
                    visited=visited
                )
                if result:
                    return result

            return None

    async def _navigate(
        self,
        query: str,
        session: Any,
        current_node: ASTNode,
        traversal_trace: list[tuple[str, str]],
        visited: set[str]
    ) -> ASTNavigationResult | None:
        if current_node.id in visited:
            return None
        visited.add(current_node.id)

        title = current_node.content[:50] if current_node.node_type != "Section" else current_node.content
        traversal_trace.append((current_node.id, title))

        # Determine if Leaf (no children)
        if not current_node.children:
            # Phase 1 Verification
            is_verified = await self._verifier.verify(query, current_node)
            if not is_verified:
                return None

            return ASTNavigationResult(
                content=current_node.content,
                node_id=current_node.id,
                path="", # To be expanded later
                title=title,
                trace=list(traversal_trace),
                verified=True,
                confidence=1.0
            )

        # It's an internal node. Pre-filter candidates deterministically.
        candidates = await self._retriever.retrieve(query, current_node, top_k=5)

        if not candidates:
            return None

        # If the retriever is very confident, we could skip LLM.
        # But Phase 1 dictates we use LLM navigation over the candidate Semantic Models.

        # Load Semantic Models for candidates
        candidate_texts = ""
        for cand in candidates:
            sm = await self._storage.get_semantic_model(session, cand.id)
            summary = sm.concise_summary if sm else cand.content[:100]
            candidate_texts += f"[{cand.id}] {cand.node_type}: {summary}\n"

        prompt = _NAVIGATE_PROMPT.format(query=query, children_text=candidate_texts)
        raw = await self._model.generate(prompt=prompt, temperature=0.0, max_tokens=150)

        chosen_id, fallback_id = self._parse_json_ids(raw)

        # Execute navigation
        for cid in [chosen_id, fallback_id]:
            if not cid:
                continue

            # Find the actual candidate node
            child_node = next((c for c in candidates if c.id == cid), None)
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

    async def _db_to_ast(self, session: Any, db_node: NodeData) -> ASTNode:
        """Convert a DB NodeData (and its children) into an ASTNode."""
        # For full retrieval, we recursive load. In production this uses joinedload.
        children_db = await self._storage.get_ast_children(session, db_node.id)
        children_ast = [await self._db_to_ast(session, c) for c in children_db]

        return ASTNode(
            id=db_node.id,
            node_type=db_node.node_type,
            content=db_node.content,
            parent_id=db_node.parent_id,
            children=children_ast
        )
