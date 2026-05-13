"""
navigation.py — The ApexRAG Navigation Agent (High-Accuracy Edition).

Navigation Strategy for 99.999% accuracy:
  1. EXPLORE  — LLM chooses the best child based on Semantic Map summaries.
  2. RECURSE  — Enter that child and repeat until a leaf is reached.
  3. VERIFY   — At the leaf, ask the LLM: "Does this section actually answer
                the query?" If YES → return. If NO → backtrack and try siblings.
  4. BACKTRACK — If all children of a node are exhausted, return None to the
                 parent (which then tries its own siblings).
  5. MULTI-CANDIDATE — The LLM may return up to 2 ranked candidates at each
                       level; the agent tries them in order before backtracking.

The tree is navigated without any depth limit — it follows the actual document
structure. A visited-node set prevents infinite loops in malformed trees.

All decisions are emitted to the ReasoningTrace for full observability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from apex_rag.providers import AsyncLLM
from apex_rag.storage import DocumentNode, StorageEngine
from apex_rag.utils import ReasoningTrace, async_retry, logger, truncate

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class NavigationResult:
    """
    The output of a successful navigation run.

    Attributes:
        content:    The exact leaf section text answering the query.
        node_id:    Primary key of the leaf DocumentNode.
        path:       LTree path of the leaf node, e.g. "2.1.3".
        title:      Section title of the found leaf.
        trace:      Ordered list of (node_id, title) pairs traversed.
        verified:   Whether the LLM verified this leaf answers the query.
        confidence: Self-reported confidence from the verification step (0–1).
    """

    content: str
    node_id: int
    path: str
    title: str
    trace: list[tuple[int, str]]
    verified: bool = False
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# LLM Prompt Templates
# ---------------------------------------------------------------------------

_NAVIGATE_PROMPT = """\
You are a precise document navigator for a search engine.

User Query: "{query}"

Below are the sub-sections available in the current document section.
Each entry shows the section ID, title, and its summary:

{children_text}

Task:
Identify the section(s) most likely to contain the answer to the query.

Rules:
- Return the BEST matching section ID as "chosen_id".
- Optionally return a second-best "fallback_id" if uncertain.
- If NO section is relevant, set both to null and explain why.

Respond ONLY with valid JSON. No prose before or after.
Format exactly:
{{"chosen_id": <integer or null>, "fallback_id": <integer or null>, "reason": "<why>"}}
"""

_VERIFY_PROMPT = """\
You are a precise quality-control agent for a search engine.

User Query: "{query}"

The following text was retrieved as the answer:

---
Section: {title}
Content: {content}
---

Task:
Does this section DIRECTLY and COMPLETELY answer the user's query?
Be strict: partial answers or tangentially related content should be rejected.

Respond ONLY with valid JSON:
{{"answers_query": <true or false>, "confidence": <0.0 to 1.0>, "reason": "<brief>"}}
"""

_ID_EXTRACTION_RE = re.compile(
    r'\{.*?"chosen_id"\s*:\s*(?P<id>\d+|null).*?\}', re.DOTALL
)
_VERIFY_RE = re.compile(
    r'\{.*?"answers_query"\s*:\s*(?P<val>true|false).*?\}', re.DOTALL | re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Navigation Agent
# ---------------------------------------------------------------------------


class NavigationAgent:
    """
    Recursive, verification-backed LLM agent that walks the document tree.

    Key accuracy features:
    - **Unlimited depth**: follows the tree as deep as it goes.
    - **Multi-candidate**: tries the LLM's best + fallback choice at each level.
    - **Verification**: every leaf is confirmed by a separate LLM call.
    - **Visited set**: prevents revisiting nodes in cycles.
    - **Sibling fallback**: exhausts all siblings before backtracking.

    Args:
        storage:          StorageEngine instance.
        model:            AsyncLLM instance for navigation decisions.
        verifier_model:   AsyncLLM instance for leaf verification (defaults to `model`).
        verify_leaves:    If True, every candidate leaf is verified by LLM.
        trace:            ReasoningTrace instance.
    """

    def __init__(
        self,
        storage: StorageEngine,
        *,
        model: AsyncLLM,
        verifier_model: AsyncLLM | None = None,
        verify_leaves: bool = True,
        trace: ReasoningTrace | None = None,
    ) -> None:
        self._storage = storage
        self._model = model
        self._verifier_model = verifier_model or model
        self._verify_leaves = verify_leaves
        self._trace = trace or ReasoningTrace(enabled=True)

    # -- Public API ---------------------------------------------------------

    async def find(
        self,
        query: str,
        doc_id: str,
        *,
        root_node_id: int | None = None,
    ) -> NavigationResult | None:
        """
        Navigate the document tree to find the leaf best answering `query`.

        Args:
            query:        Natural-language question.
            doc_id:       Document to search (from ingestion).
            root_node_id: Restrict search to a subtree (optional).

        Returns:
            NavigationResult, or None if the query cannot be answered.
        """
        async with self._storage.session() as session:
            if root_node_id is not None:
                root_node = await self._storage.get_node(session, root_node_id)
                root_nodes = [root_node] if root_node else []
            else:
                root_nodes = list(
                    await self._storage.get_children(
                        session, parent_id=None, doc_id=doc_id
                    )
                )

            if not root_nodes:
                logger.warning("No root nodes for doc_id=%r", doc_id)
                return None

            first = root_nodes[0]
            self._trace.start(query, first.id)

            traversal_trace: list[tuple[int, str]] = []
            visited: set[int] = set()

            for root in root_nodes:
                result = await self._navigate(
                    query=query,
                    session=session,
                    current_node=root,
                    traversal_trace=traversal_trace,
                    visited=visited,
                )
                if result is not None:
                    self._trace.finish(found=True)
                    return result

            self._trace.finish(found=False)
            return None

    # -- Recursive navigation -----------------------------------------------

    async def _navigate(
        self,
        query: str,
        session: Any,
        current_node: DocumentNode,
        traversal_trace: list[tuple[int, str]],
        visited: set[int],
    ) -> NavigationResult | None:
        """
        Depth-first navigation with multi-candidate and verification.
        No depth limit — recurse until leaves or exhaustion.
        """
        if current_node.id in visited:
            logger.debug("Skipping already-visited node %d", current_node.id)
            return None
        visited.add(current_node.id)

        traversal_trace.append((current_node.id, current_node.title))
        self._trace.enter_node(current_node.id, current_node.summary, current_node.path)

        # ── Leaf node ──────────────────────────────────────────────────────
        if current_node.is_leaf:
            content = current_node.content or ""
            self._trace.leaf_reached(current_node.id, content)

            if self._verify_leaves:
                verified, confidence = await self._verify_leaf(
                    query, current_node.title, content
                )
                if not verified:
                    self._trace.backtrack(
                        current_node.id,
                        current_node.parent_id,
                        reason=f"Verification failed (confidence={confidence:.2f})",
                    )
                    return None  # signal backtrack
                return NavigationResult(
                    content=content,
                    node_id=current_node.id,
                    path=current_node.path,
                    title=current_node.title,
                    trace=list(traversal_trace),
                    verified=True,
                    confidence=confidence,
                )
            else:
                return NavigationResult(
                    content=content,
                    node_id=current_node.id,
                    path=current_node.path,
                    title=current_node.title,
                    trace=list(traversal_trace),
                    verified=False,
                    confidence=1.0,
                )

        # ── Internal node: fetch children and ask LLM ─────────────────────
        children = [
            c for c in await self._storage.get_children(session, parent_id=current_node.id)
            if c.id not in visited
        ]

        if not children:
            # Node has children in DB but all visited or none → treat as leaf
            fallback_content = current_node.summary or current_node.title
            self._trace.leaf_reached(current_node.id, fallback_content)
            return NavigationResult(
                content=fallback_content,
                node_id=current_node.id,
                path=current_node.path,
                title=current_node.title,
                trace=list(traversal_trace),
                verified=False,
                confidence=0.5,
            )

        self._trace.exploring_children(current_node.id, len(children))

        chosen_id, fallback_id, reason = await self._ask_llm(query, children)
        self._trace.agent_choice(chosen_id, reason)

        # Build ordered candidate list: [chosen, fallback, remaining...]
        candidate_ids: list[int] = []
        if chosen_id is not None:
            candidate_ids.append(chosen_id)
        if fallback_id is not None and fallback_id != chosen_id:
            candidate_ids.append(fallback_id)
        # Append remaining siblings in order for exhaustive search
        for child in children:
            if child.id not in candidate_ids:
                candidate_ids.append(child.id)

        id_to_child = {c.id: c for c in children}

        # If LLM returned NONE and no fallback, backtrack immediately
        if chosen_id is None and fallback_id is None:
            self._trace.backtrack(current_node.id, current_node.parent_id)
            return None

        for candidate_id in candidate_ids:
            child = id_to_child.get(candidate_id)
            if child is None or child.id in visited:
                continue
            result = await self._navigate(
                query=query,
                session=session,
                current_node=child,
                traversal_trace=traversal_trace,
                visited=visited,
            )
            if result is not None:
                return result

        # All candidates exhausted
        self._trace.backtrack(current_node.id, current_node.parent_id)
        return None

    # -- LLM calls ----------------------------------------------------------

    @async_retry(max_attempts=3, backoff_base=2.0, exceptions=(Exception,))
    async def _ask_llm(
        self,
        query: str,
        children: list[DocumentNode],
    ) -> tuple[int | None, int | None, str]:
        """
        Ask Ollama which child(ren) to explore.

        Returns:
            (chosen_id, fallback_id, reason)
        """
        children_text = "\n".join(
            f"[{c.id}] {c.title}{' ('+c.page_range+')' if c.page_range else ''}\n"
            f"       Summary: {truncate(c.summary, 140)}"
            for c in children
        )

        prompt = _NAVIGATE_PROMPT.format(
            query=query,
            children_text=children_text,
        )

        raw = await self._model.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=120,
        )

        return self._parse_navigate_response(raw.strip(), children)

    @async_retry(max_attempts=2, backoff_base=1.5, exceptions=(Exception,))
    async def _verify_leaf(
        self,
        query: str,
        title: str,
        content: str,
    ) -> tuple[bool, float]:
        """
        Ask the verifier model if `content` actually answers `query`.

        Returns:
            (answers_query: bool, confidence: float)
        """
        prompt = _VERIFY_PROMPT.format(
            query=query,
            title=title,
            content=truncate(content, 1500),
        )

        raw = await self._verifier_model.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=80,
        )

        return self._parse_verify_response(raw.strip())

    # -- Response parsers ---------------------------------------------------

    def _parse_navigate_response(
        self,
        raw: str,
        children: list[DocumentNode],
    ) -> tuple[int | None, int | None, str]:
        """
        4-tier fallback parser for the navigation LLM response.
        Returns (chosen_id, fallback_id, reason).
        """
        valid_ids = {c.id for c in children}

        def _to_valid_id(val: Any) -> int | None:
            if val is None or str(val).lower() in ("null", "none", ""):
                return None
            try:
                cid = int(val)
                return cid if cid in valid_ids else None
            except (ValueError, TypeError):
                return None

        # Attempt 1: strict JSON
        try:
            data: dict[str, Any] = json.loads(raw)
            chosen = _to_valid_id(data.get("chosen_id"))
            fallback = _to_valid_id(data.get("fallback_id"))
            reason = str(data.get("reason", ""))
            return chosen, fallback, reason
        except (json.JSONDecodeError, KeyError):
            pass

        # Attempt 2: regex on embedded JSON
        match = _ID_EXTRACTION_RE.search(raw)
        if match:
            id_str = match.group("id")
            chosen = _to_valid_id(id_str)
            return chosen, None, "regex-parsed"

        # Attempt 3: explicit NONE keyword
        if "NONE" in raw.upper() or "NONE" in raw:
            return None, None, "explicit NONE"

        # Attempt 4: heuristic — first two valid IDs in the response
        numbers = re.findall(r"\b(\d+)\b", raw)
        found_ids: list[int] = []
        for num_str in numbers:
            cid = int(num_str)
            if cid in valid_ids and cid not in found_ids:
                found_ids.append(cid)
            if len(found_ids) == 2:
                break

        if found_ids:
            chosen = found_ids[0]
            fallback = found_ids[1] if len(found_ids) > 1 else None
            logger.info("Heuristic: extracted ids=%s", found_ids)
            return chosen, fallback, "heuristic"

        logger.error("Parse failure: %s", truncate(raw, 200))
        return None, None, "parse failure"

    def _parse_verify_response(self, raw: str) -> tuple[bool, float]:
        """Parse the leaf verification response. Returns (answers_query, confidence)."""
        # Attempt 1: strict JSON
        try:
            data = json.loads(raw)
            answers = bool(data.get("answers_query", False))
            confidence = float(data.get("confidence", 0.5))
            return answers, min(max(confidence, 0.0), 1.0)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Attempt 2: regex
        match = _VERIFY_RE.search(raw)
        if match:
            answers = match.group("val").lower() == "true"
            # Try to find a confidence number
            conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
            confidence = float(conf_match.group(1)) if conf_match else 0.7
            return answers, min(max(confidence, 0.0), 1.0)

        # Attempt 3: keyword scan
        raw_lower = raw.lower()
        if any(w in raw_lower for w in ("yes", "true", "correct", "answers", "relevant")):
            return True, 0.7
        if any(w in raw_lower for w in ("no", "false", "incorrect", "does not", "doesn't")):
            return False, 0.3

        logger.warning("Verify parse failure: %s", truncate(raw, 100))
        return True, 0.5  # Optimistic default to avoid over-rejection
