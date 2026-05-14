"""
test_search.py — Unit tests for the NavigationAgent.

Uses a fully mocked document tree and a stubbed LLM to verify the
navigation logic without any Ollama dependency.

The mock tree structure:
    Root (id=1)
    ├── Chapter 1: Physics (id=2)
    │   ├── Section 1.1: Mechanics (id=4) [LEAF]
    │   └── Section 1.2: Thermodynamics (id=5) [LEAF]
    └── Chapter 2: Chemistry (id=3)
        ├── Section 2.1: Organic (id=6) [LEAF]  ← contains "benzene ring"
        └── Section 2.2: Inorganic (id=7) [LEAF]
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apex_rag.navigation import NavigationAgent, NavigationResult
from apex_rag.providers import AsyncLLM
from apex_rag.storage import DocumentNode, StorageEngine
from apex_rag.utils import ReasoningTrace

# ---------------------------------------------------------------------------
# Mock Tree Builder
# ---------------------------------------------------------------------------

DOC_ID = "mock-doc-001"


def _make_node(
    node_id: int,
    parent_id: int | None,
    title: str,
    summary: str,
    path: str,
    depth: int,
    position: int,
    content: str | None = None,
) -> DocumentNode:
    node = DocumentNode(
        doc_id=DOC_ID,
        parent_id=parent_id,
        path=path,
        title=title,
        summary=summary,
        content=content,
        depth=depth,
        position=position,
        page_start=0,
        page_end=0,
    )
    node.id = node_id
    node.children = []  # populated manually below
    return node


def _build_mock_tree() -> dict[int, DocumentNode]:
    nodes: dict[int, DocumentNode] = {}

    root = _make_node(1, None, "Document Root", "Entire science textbook", "1", 0, 1)
    ch1  = _make_node(2, 1,    "Chapter 1: Physics",   "Covers mechanics, thermodynamics", "1.1", 1, 1)
    ch2  = _make_node(3, 1,    "Chapter 2: Chemistry", "Covers organic and inorganic chem", "1.2", 1, 2)
    sec11 = _make_node(4, 2,   "Section 1.1: Mechanics",
                       "Newton's laws and kinematics", "1.1.1", 2, 1,
                       content="F=ma. Objects in motion stay in motion.")
    sec12 = _make_node(5, 2,   "Section 1.2: Thermodynamics",
                       "Heat, entropy, and energy transfer", "1.1.2", 2, 2,
                       content="Heat flows from hot to cold bodies.")
    sec21 = _make_node(6, 3,   "Section 2.1: Organic Chemistry",
                       "Carbon compounds, benzene ring structures, polymers", "1.2.1", 2, 1,
                       content="Benzene (C6H6) is the archetypal aromatic compound with a ring structure.")
    sec22 = _make_node(7, 3,   "Section 2.2: Inorganic Chemistry",
                       "Metals, salts, and ionic compounds", "1.2.2", 2, 2,
                       content="Sodium chloride (NaCl) is a common ionic compound.")

    # Wire up children lists (mirrors what DB queries would return)
    root.children = [ch1, ch2]
    ch1.children  = [sec11, sec12]
    ch2.children  = [sec21, sec22]

    for n in [root, ch1, ch2, sec11, sec12, sec21, sec22]:
        nodes[n.id] = n

    return nodes


TREE = _build_mock_tree()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def silent_trace() -> ReasoningTrace:
    """A ReasoningTrace with output suppressed for clean test output."""
    return ReasoningTrace(enabled=False)


@pytest.fixture
def mock_storage(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Return a MagicMock StorageEngine pre-wired with the mock tree.
    get_node, get_children, and list_documents are async-patched.
    """
    storage = MagicMock(spec=StorageEngine)
    storage._engine = MagicMock()
    storage._engine.url = "sqlite+aiosqlite:///mock.db"

    async def fake_get_node(session: object, node_id: int) -> DocumentNode | None:
        return TREE.get(node_id)

    async def fake_get_children(
        session: object, parent_id: int | None, doc_id: str | None = None
    ) -> list[DocumentNode]:
        children = [n for n in TREE.values() if n.parent_id == parent_id]
        return sorted(children, key=lambda n: n.position)

    storage.get_node = AsyncMock(side_effect=fake_get_node)
    storage.get_children = AsyncMock(side_effect=fake_get_children)
    storage.list_documents = AsyncMock(return_value=[DOC_ID])
    storage.get_cached_query = AsyncMock(return_value=None)
    storage.insert_cache_entry = AsyncMock()
    storage.search_children = AsyncMock(return_value=[])

    # session() must return an async context manager
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    session_cm.__aexit__ = AsyncMock(return_value=False)
    storage.session = MagicMock(return_value=session_cm)

    return storage


class DummyLLM:
    """Dummy AsyncLLM for testing."""
    async def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 150) -> str:
        return ""

# ---------------------------------------------------------------------------
# Helper: patch the LLM with a scripted decision sequence
# ---------------------------------------------------------------------------


def make_llm_responses(decisions: list[tuple[int | None, str]]):
    """
    Return a patched _ask_llm that replays `decisions` in order.
    Each decision is (chosen_id, reason); fallback_id is always None.
    """
    call_index = {"i": 0}

    async def patched_ask_llm(
        self: NavigationAgent,
        query: str,
        children: list[DocumentNode],
        session: any = None,
        **kwargs: any,
    ) -> tuple[int | None, int | None, str]:
        idx = call_index["i"]
        call_index["i"] += 1
        if idx < len(decisions):
            chosen, reason = decisions[idx]
            return chosen, None, reason
        return None, None, "exhausted"

    return patched_ask_llm


# ---------------------------------------------------------------------------
# Navigation Tests
# ---------------------------------------------------------------------------


class TestNavigationAgent:

    @pytest.mark.asyncio
    async def test_finds_correct_leaf(
        self, mock_storage: MagicMock, silent_trace: ReasoningTrace
    ) -> None:
        """
        Query about benzene should navigate: Root → Ch2 → Section 2.1 (leaf).
        LLM decisions: [root→ch2 (id=3), ch2→sec21 (id=6)]
        """
        agent = NavigationAgent(mock_storage, model=DummyLLM(), trace=silent_trace, verify_leaves=False)

        decisions = [(3, "Chemistry chapter covers benzene"), (6, "Organic chem has benzene ring")]
        with patch.object(NavigationAgent, "_ask_llm", make_llm_responses(decisions)):
            result = await agent.find("What is the benzene ring structure?", DOC_ID, root_node_id=1)

        assert result is not None
        assert result.node_id == 6
        assert "Benzene" in result.content
        assert result.path == "1.2.1"

    @pytest.mark.asyncio
    async def test_returns_none_when_llm_says_none(
        self, mock_storage: MagicMock, silent_trace: ReasoningTrace
    ) -> None:
        """
        If LLM returns NONE at the root level, result should be None.
        """
        agent = NavigationAgent(mock_storage, model=DummyLLM(), trace=silent_trace)

        decisions = [(None, "No section matches this query")]
        with patch.object(NavigationAgent, "_ask_llm", make_llm_responses(decisions)):
            result = await agent.find("What is the population of Mars?", DOC_ID, root_node_id=1)

        assert result is None

    @pytest.mark.asyncio
    async def test_backtrack_and_find_in_sibling(
        self, mock_storage: MagicMock, silent_trace: ReasoningTrace
    ) -> None:
        """
        Navigate into wrong child first (Physics), backtrack, then find in Chemistry.

        Actual agent call sequence:
          call#0 → root's children → choose ch1 (id=2)
          call#1 → ch1's children → returns NONE  (backtrack)
          [ch2 is now tried directly as a remaining sibling — no extra LLM call at root]
          call#2 → ch2's children → choose sec22 (id=7)
        """
        agent = NavigationAgent(mock_storage, model=DummyLLM(), trace=silent_trace, verify_leaves=False)

        decisions = [
            (2, "Physics seems relevant"),   # call#0: root → ch1 (wrong path)
            (None, "Not in physics"),         # call#1: ch1 → NONE (backtrack)
            (7, "Inorganic covers salts"),    # call#2: ch2 → sec22
        ]
        with patch.object(NavigationAgent, "_ask_llm", make_llm_responses(decisions)):
            result = await agent.find("Tell me about sodium chloride.", DOC_ID, root_node_id=1)

        assert result is not None
        assert result.node_id == 7
        assert "NaCl" in result.content

    @pytest.mark.asyncio
    async def test_traversal_trace_populated(
        self, mock_storage: MagicMock, silent_trace: ReasoningTrace
    ) -> None:
        """The trace should record the path of visited nodes."""
        agent = NavigationAgent(mock_storage, model=DummyLLM(), trace=silent_trace, verify_leaves=False)

        decisions = [(2, "Physics"), (4, "Mechanics has Newton's laws")]
        with patch.object(NavigationAgent, "_ask_llm", make_llm_responses(decisions)):
            result = await agent.find("What is Newton's second law?", DOC_ID, root_node_id=1)

        assert result is not None
        visited_ids = [nid for nid, _ in result.trace]
        assert 1 in visited_ids  # root
        assert 2 in visited_ids  # ch1
        assert 4 in visited_ids  # sec11 (leaf)

    @pytest.mark.asyncio
    async def test_direct_leaf_node(
        self, mock_storage: MagicMock, silent_trace: ReasoningTrace
    ) -> None:
        """If root_node_id points directly to a leaf, return it immediately."""
        agent = NavigationAgent(mock_storage, model=DummyLLM(), trace=silent_trace, verify_leaves=False)
        result = await agent.find("F=ma", DOC_ID, root_node_id=4)

        assert result is not None
        assert result.node_id == 4
        assert "F=ma" in result.content


# ---------------------------------------------------------------------------
# LLM Response Parser Tests
# ---------------------------------------------------------------------------


class TestResponseParser:

    @pytest.fixture
    def agent(self, mock_storage: MagicMock, silent_trace: ReasoningTrace) -> NavigationAgent:
        return NavigationAgent(mock_storage, model=DummyLLM(), trace=silent_trace, verify_leaves=False)

    def test_valid_json_parsed(self, agent: NavigationAgent) -> None:
        children = [TREE[2], TREE[3]]
        raw = '{"chosen_id": 3, "reason": "Chemistry covers this topic"}'
        chosen, fallback, reason = agent._parse_navigate_response(raw, children)
        assert chosen == 3
        assert "Chemistry" in reason

    def test_null_id_returns_none(self, agent: NavigationAgent) -> None:
        children = [TREE[2], TREE[3]]
        raw = '{"chosen_id": null, "reason": "No relevant section"}'
        chosen, fallback, reason = agent._parse_navigate_response(raw, children)
        assert chosen is None

    def test_none_keyword_returns_none(self, agent: NavigationAgent) -> None:
        children = [TREE[2], TREE[3]]
        chosen, fallback, reason = agent._parse_navigate_response("NONE", children)
        assert chosen is None

    def test_invalid_id_returns_none(self, agent: NavigationAgent) -> None:
        children = [TREE[2], TREE[3]]
        raw = '{"chosen_id": 999, "reason": "Some reason"}'
        chosen, _, _r = agent._parse_navigate_response(raw, children)
        assert chosen is None  # 999 not in valid_ids

    def test_json_embedded_in_prose(self, agent: NavigationAgent) -> None:
        children = [TREE[2], TREE[3]]
        raw = 'Based on my analysis: {"chosen_id": 2, "reason": "Physics"} is the answer.'
        chosen, _, _r = agent._parse_navigate_response(raw, children)
        assert chosen == 2

    def test_heuristic_number_extraction(self, agent: NavigationAgent) -> None:
        children = [TREE[2], TREE[3]]
        raw = "I think section 3 is most relevant."
        chosen, _, _r = agent._parse_navigate_response(raw, children)
        assert chosen == 3

    def test_malformed_response_returns_none(self, agent: NavigationAgent) -> None:
        children = [TREE[2], TREE[3]]
        chosen, _, _r = agent._parse_navigate_response("The answer is definitely here!", children)
        assert chosen is None
