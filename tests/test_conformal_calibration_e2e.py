"""
test_conformal_calibration_e2e.py — End-to-end test for real conformal calibration.

Proves the behavior change from Phase 2 of the benchmark-honesty plan:

    - **Before** ``index.enterprise.calibrate_conformal(...)`` is ever
      called, ``ConformalWrapperAgent`` uses its uncalibrated default
      threshold (0.0): every retrieved packet passes through unfiltered
      and ``answer.coverage_guarantee`` reports ``0.0``. This is today's
      real, if undocumented, behavior -- not a bug this test is fixing,
      just documenting.

    - **After** calibration with a held-out labeled set,
      ``ConformalWrapperAgent`` keeps the calibrated threshold as instance
      state on the orchestrator, so every *subsequent* ``index.query()``
      call automatically reflects it -- no other API changes needed.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from apex_rag.client import ApexIndex
from apex_rag.providers import AsyncLLM

# ── Mock LLM (mirrors tests/test_benchmarks.py::mock_llm) ──────────────────


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=AsyncLLM)

    async def mock_generate(prompt: str, **_kwargs: Any) -> str:
        prompt_lower = prompt.lower()
        # StrictLeafVerifier (apex_rag/retrieval/verification/strict_verifier.py)
        # expects a bare "TRUE"/"FALSE" string, not JSON -- and its prompt
        # contains "strict verification engine", not the word "verify".
        if "strict verification engine" in prompt_lower:
            return "TRUE"
        if "decomposition" in prompt_lower or "plan" in prompt_lower:
            # Omit sub_queries so QueryPlannerAgent falls back to [original
            # query] (apex_rag/agents/planner/agent.py) instead of a fake
            # placeholder that wouldn't match any real document content.
            return "{}"
        if "chosen_id" in prompt_lower:
            ids = re.findall(r"\[([a-f0-9\-]+)\]", prompt)
            chosen = ids[0] if ids else "1"
            return f'{{"chosen_id": "{chosen}", "fallback_id": null, "reason": "Mock selection"}}'
        elif "evaluate" in prompt_lower or "provides enough information" in prompt_lower:
            return '{"passes_evaluation": true, "reason": "Verified"}'
        elif "cite each claim" in prompt_lower:
            return "Citing claim. [Node ID: mock-1]"
        else:
            return "Mock summary about the topic."

    llm.generate = AsyncMock(side_effect=mock_generate)

    async def mock_embed(texts: list[str], **_kwargs: Any) -> list[list[float]]:
        import random

        return [[random.uniform(-1.0, 1.0) for _ in range(384)] for _ in texts]

    llm.embed = AsyncMock(side_effect=mock_embed)

    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[str, None]:
        yield "Mocked "
        yield "response"

    llm.stream_generate = _stream
    return llm


# ── Fixtures ─────────────────────────────────────────────────────────────

# 10 tiny single-fact documents -- one candidate node each, so the mock
# navigator (which always picks the first/only candidate) reliably lands on
# the node containing the gold answer. Used as the held-out calibration set.
_CALIBRATION_FACTS = [
    ("cal-1", "Revenue", "Revenue in Q1 was $10M.", "What was Q1 revenue?", "$10M"),
    ("cal-2", "Headcount", "Headcount at year end was 500.", "What was year-end headcount?", "500"),
    ("cal-3", "CEO", "The CEO is Jane Smith.", "Who is the CEO?", "Jane Smith"),
    ("cal-4", "Margin", "Net margin improved to 12%.", "What was net margin?", "12%"),
    ("cal-5", "Launch", "The product launched on March 3rd.", "When did the product launch?", "March 3rd"),
    ("cal-6", "HQ", "Headquarters moved to Austin.", "Where is the headquarters?", "Austin"),
    ("cal-7", "Funding", "The company raised $25M in Series B.", "How much was raised in Series B?", "$25M"),
    ("cal-8", "Churn", "Customer churn dropped to 3%.", "What was customer churn?", "3%"),
    ("cal-9", "Partner", "The main partner is Acme Corp.", "Who is the main partner?", "Acme Corp"),
    ("cal-10", "Uptime", "System uptime was 99.99%.", "What was system uptime?", "99.99%"),
]


@pytest.fixture
async def calibrated_index(mock_llm: MagicMock) -> ApexIndex:
    """An index with the 10 calibration documents ingested (not yet calibrated)."""
    index = await ApexIndex.create(
        db_url="sqlite+aiosqlite:///:memory:",
        provider=mock_llm,
        trace_enabled=False,
    )
    for doc_id, heading, fact, _question, _gold in _CALIBRATION_FACTS:
        await index.ingest_text(f"# {heading}\n{fact}", doc_id=doc_id)
    return index


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_uncalibrated_query_reports_zero_coverage(calibrated_index: ApexIndex) -> None:
    """Documents today's real (undocumented) default behavior: with no
    calibration ever performed, coverage_guarantee is 0.0 and every
    retrieved packet passes through unfiltered."""
    answer = await calibrated_index.query("What was Q1 revenue?", "cal-1")

    assert answer.coverage_guarantee == 0.0
    assert len(answer.evidence_packets) > 0


@pytest.mark.asyncio
async def test_calibrate_conformal_changes_future_queries(calibrated_index: ApexIndex) -> None:
    """After calibrating with a held-out labeled set, subsequent
    index.query() calls automatically use the calibrated threshold --
    no signature change to query() needed."""
    enterprise = calibrated_index.enterprise

    calibration_examples = [
        (question, doc_id, gold) for doc_id, _heading, _fact, question, gold in _CALIBRATION_FACTS
    ]
    summary = await enterprise.calibrate_conformal(calibration_examples)

    assert summary["n_examples"] == 10
    assert summary["n_retrieval_hits"] == 10  # every fact was actually retrieved
    assert summary["calibrated"] is True
    assert summary["threshold"] > 0.0

    # A fresh query on one of the calibrated documents should now report a
    # real (non-zero) coverage guarantee, reflecting the calibrated wrapper
    # -- with no change to how query() is called.
    answer = await calibrated_index.query("What was Q1 revenue?", "cal-1")
    assert answer.coverage_guarantee == pytest.approx(
        calibrated_index.enterprise._orchestrator.conformal_wrapper.calibrator.coverage_level
    )
    assert answer.coverage_guarantee > 0.0


@pytest.mark.asyncio
async def test_calibrate_conformal_too_few_examples_stays_conservative(
    calibrated_index: ApexIndex,
) -> None:
    """Fewer than min_calibration_size (10) examples falls back to the
    same conservative 0.0 threshold as no calibration at all -- it should
    never silently claim a guarantee it can't back up."""
    enterprise = calibrated_index.enterprise

    summary = await enterprise.calibrate_conformal(
        [(q, d, g) for d, _h, _f, q, g in _CALIBRATION_FACTS[:3]]
    )

    assert summary["n_examples"] == 3
    assert summary["calibrated"] is False
    assert summary["threshold"] == 0.0
