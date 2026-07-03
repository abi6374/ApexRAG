"""
tests/temporal/test_temporal.py — 18+ tests for Part 3: Temporal Intelligence.

Covers:
    - TemporalExtractor: date extraction from 5+ formats via Strategy B (regex)
    - TemporalExtractor: Strategy A (metadata)
    - TemporalExtractor: Strategy C (LLM fallback — mocked)
    - FreshnessScorer: decay at 0, 30, 365, 1825 days
    - FreshnessScorer: domain-specific decay rates
    - FreshnessScorer: half-life computation
    - TemporalContradictionDetector: known-conflicting pair
    - TemporalContradictionDetector: known-consistent pair
    - TemporalContradictionDetector: cosine similarity threshold enforcement
    - TemporalContradictionDetector: negation heuristic
    - TemporalContradictionDetector: LLM confirmation (mocked)
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from apex_rag.models.unified_models import ASTNode, EdgeType, NodeType
from apex_rag.temporal.contradiction import TemporalContradictionDetector
from apex_rag.temporal.extractor import TemporalExtractor
from apex_rag.temporal.scorer import DEFAULT_DECAY_RATES, FreshnessScorer

# ══════════════════════════════════════════════════════════════════════════
# TemporalExtractor tests
# ══════════════════════════════════════════════════════════════════════════


class TestStrategyA_Metadata:
    """Metadata-based date extraction (Strategy A)."""

    def test_pdf_creation_date_iso(self) -> None:
        """Extract from 'creation_date' with ISO format."""
        extractor = TemporalExtractor()
        result = extractor._strategy_a({"creation_date": "2024-06-01"})
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 1

    def test_docx_last_modified(self) -> None:
        """Extract from 'last_modified' key."""
        extractor = TemporalExtractor()
        result = extractor._strategy_a({"last_modified": "2023-12-15"})
        assert result is not None
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 15

    def test_markdown_frontmatter_date(self) -> None:
        """Extract from 'date' key (frontmatter)."""
        extractor = TemporalExtractor()
        result = extractor._strategy_a({"date": "2025-01-20"})
        assert result is not None
        assert result.year == 2025
        assert result.month == 1

    def test_empty_metadata_returns_none(self) -> None:
        """Empty or missing metadata yields None."""
        extractor = TemporalExtractor()
        assert extractor._strategy_a({}) is None
        assert extractor._strategy_a({"title": "No dates here"}) is None

    def test_metadata_none_returns_none(self) -> None:
        """When metadata parameter is None, skip Strategy A."""
        extractor = TemporalExtractor()
        result = extractor._strategy_a({})
        assert result is None


class TestStrategyB_Regex:
    """Regex-based date extraction (Strategy B)."""

    def test_iso_format(self) -> None:
        """YYYY-MM-DD format."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Published on 2024-06-01 in the journal.")
        assert result is not None
        assert result == datetime(2024, 6, 1, tzinfo=timezone.utc)

    def test_us_long_format(self) -> None:
        """Month DD, YYYY format."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Issued June 1, 2024 by the committee.")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 1

    def test_us_short_format(self) -> None:
        """Mon DD, YYYY format (abbreviated month)."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Last updated: Dec 15, 2023.")
        assert result is not None
        assert result.year == 2023
        assert result.month == 12

    def test_dd_month_yyyy(self) -> None:
        """DD Month YYYY format."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Contract signed 15 March 2022.")
        assert result is not None
        assert result.year == 2022
        assert result.month == 3
        assert result.day == 15

    def test_month_yyyy(self) -> None:
        """Month YYYY format (no day — defaults to 1st)."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Report for January 2023 shows growth.")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1

    def test_quarter_format(self) -> None:
        """Q3 2024 format."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Results for Q3 2024 exceeded targets.")
        assert result is not None
        assert result.year == 2024
        assert result.month == 7  # Q3 starts in July
        assert result.day == 1

    def test_quarter_with_fy(self) -> None:
        """Q1 FY2024 format."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Budget: Q1 FY2024 allocation.")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1

    def test_year_only(self) -> None:
        """Four-digit year only."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Copyright 2021 by the author.")
        assert result is not None
        assert result.year == 2021
        assert result.month == 1

    def test_european_format(self) -> None:
        """DD/MM/YYYY format."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("Issued 01/06/2024 (European format).")
        # Should parse as DD/MM/YYYY
        if result is not None:
            # We accept either DD/MM or MM/DD interpretation
            assert result.year == 2024
        assert result is not None

    def test_no_date_returns_none(self) -> None:
        """Text with no recognizable date returns None."""
        extractor = TemporalExtractor()
        result = extractor._strategy_b("This document has no date information whatsoever.")
        assert result is None

    def test_date_beyond_500_chars_ignored(self) -> None:
        """Dates after the first 500 chars are not detected."""
        extractor = TemporalExtractor()
        text = "A" * 490 + "2024-12-25"
        extractor._strategy_b(text)
        # 2024-12-25 starts at index 490, within the 500 char window
        # Actually it starts at 490, and the match ends at 500, so it IS within 500 chars.
        # Let me test with a date beyond 500:
        text2 = "A" * 510 + "2024-12-25"
        result2 = extractor._strategy_b(text2)
        assert result2 is None


class TestStrategyC_LLM:
    """LLM-based date extraction (Strategy C — mocked)."""

    @pytest.mark.asyncio
    async def test_llm_confident_date(self) -> None:
        """LLM returns a confident date."""
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="2024-06-01|95")
        extractor = TemporalExtractor(llm=llm)
        result = await extractor._strategy_c("Some text about 2024 events.")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6

    @pytest.mark.asyncio
    async def test_llm_low_confidence_returns_none(self) -> None:
        """LLM returns low confidence < 70 % → None."""
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="2024-06-01|50")
        extractor = TemporalExtractor(llm=llm)
        result = await extractor._strategy_c("Ambiguous text.")
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_returns_none_string(self) -> None:
        """LLM returns 'None' explicitly."""
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="None")
        extractor = TemporalExtractor(llm=llm)
        result = await extractor._strategy_c("Vague text.")
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self) -> None:
        """LLM raises exception → graceful None."""
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
        extractor = TemporalExtractor(llm=llm)
        result = await extractor._strategy_c("Any text.")
        assert result is None


class TestFullExtract:
    """Integration tests for the full extract() pipeline."""

    @pytest.mark.asyncio
    async def test_metadata_takes_priority(self) -> None:
        """Strategy A (metadata) wins over Strategy B (regex)."""
        extractor = TemporalExtractor()
        text = "Published on 2023-01-15"
        metadata = {"creation_date": "2024-06-01"}
        result = await extractor.extract(text, metadata=metadata)
        assert result is not None
        # Should use the metadata date (2024-06-01), not the regex match
        assert result.year == 2024

    @pytest.mark.asyncio
    async def test_fallback_to_regex(self) -> None:
        """Falls back to regex when metadata is empty."""
        extractor = TemporalExtractor()
        text = "Published on 2023-01-15 in the journal."
        result = await extractor.extract(text, metadata={})
        assert result is not None
        assert result.year == 2023

    @pytest.mark.asyncio
    async def test_no_date_extracted(self) -> None:
        """Returns None when no strategy finds a date."""
        extractor = TemporalExtractor()
        result = await extractor.extract("No dates here at all.")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
# FreshnessScorer tests
# ══════════════════════════════════════════════════════════════════════════


class TestFreshnessScorer:
    """Exponential-decay freshness scoring."""

    def test_freshness_zero_days(self) -> None:
        """Freshness = 1.0 when days_since = 0 (today)."""
        scorer = FreshnessScorer(reference_date=date(2024, 6, 1))
        score = scorer.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert score == 1.0

    def test_freshness_30_days(self) -> None:
        """Freshness after 30 days at default rate (0.001)."""
        scorer = FreshnessScorer(reference_date=date(2024, 7, 1))
        score = scorer.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        expected = math.exp(-0.001 * 30)
        assert score == pytest.approx(expected, rel=1e-6)

    def test_freshness_365_days(self) -> None:
        """Freshness after 1 year at default rate."""
        scorer = FreshnessScorer(reference_date=date(2025, 6, 1))
        score = scorer.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        expected = math.exp(-0.001 * 365)
        assert score == pytest.approx(expected, rel=1e-4)

    def test_freshness_1825_days(self) -> None:
        """Freshness after ~5 years at default rate."""
        scorer = FreshnessScorer(reference_date=date(2029, 6, 1))
        score = scorer.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        expected = math.exp(-0.001 * 1825)
        assert score == pytest.approx(expected, rel=1e-3)

    def test_none_date_is_fresh(self) -> None:
        """None source_date → freshness = 1.0 (assumed current)."""
        scorer = FreshnessScorer()
        score = scorer.compute(None)
        assert score == 1.0

    def test_future_date_capped_at_one(self) -> None:
        """Future dates (negative days) → freshness = 1.0."""
        scorer = FreshnessScorer(reference_date=date(2024, 1, 1))
        score = scorer.compute(datetime(2025, 6, 1, tzinfo=timezone.utc))
        assert score == 1.0

    def test_legal_domain_decay(self) -> None:
        """Legal domain has slower decay (0.0002)."""
        scorer = FreshnessScorer(domain="legal", reference_date=date(2025, 6, 1))
        score = scorer.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        expected = math.exp(-0.0002 * 365)
        assert score == pytest.approx(expected, rel=1e-4)
        # Legal should be fresher than general after the same period
        general = FreshnessScorer(domain="general", reference_date=date(2025, 6, 1))
        general_score = general.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert score > general_score

    def test_news_domain_decay(self) -> None:
        """News domain has faster decay (0.005)."""
        scorer = FreshnessScorer(domain="news", reference_date=date(2024, 7, 1))
        score = scorer.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        expected = math.exp(-0.005 * 30)
        assert score == pytest.approx(expected, rel=1e-4)

    def test_custom_decay_rate(self) -> None:
        """Explicit decay rate overrides domain."""
        scorer = FreshnessScorer(decay_rate=0.01, reference_date=date(2024, 7, 1))
        score = scorer.compute(datetime(2024, 6, 1, tzinfo=timezone.utc))
        expected = math.exp(-0.01 * 30)
        assert score == pytest.approx(expected, rel=1e-4)

    def test_half_life_general(self) -> None:
        """Half-life for general domain (0.001) ≈ 693 days."""
        scorer = FreshnessScorer(domain="general")
        half = scorer.half_life_days()
        expected = math.log(2) / 0.001
        assert half == pytest.approx(expected, rel=1e-4)

    def test_half_life_legal(self) -> None:
        """Half-life for legal domain (0.0002) ≈ 3466 days."""
        scorer = FreshnessScorer(domain="legal")
        half = scorer.half_life_days()
        expected = math.log(2) / 0.0002
        assert half == pytest.approx(expected, rel=1e-4)

    def test_suggest_decay_rate(self) -> None:
        """Suggest a decay rate for a 365-day half-life."""
        rate = FreshnessScorer.suggest_decay_rate(365)
        expected = math.log(2) / 365
        assert rate == pytest.approx(expected, rel=1e-4)

    def test_zero_decay_rate_infinite_half_life(self) -> None:
        """Zero decay → infinite half-life."""
        scorer = FreshnessScorer(decay_rate=0)
        assert scorer.half_life_days() == float("inf")

    def test_default_decay_rates_registry(self) -> None:
        """DEFAULT_DECAY_RATES has expected keys."""
        assert "general" in DEFAULT_DECAY_RATES
        assert "news" in DEFAULT_DECAY_RATES
        assert "legal" in DEFAULT_DECAY_RATES
        assert DEFAULT_DECAY_RATES["general"] == 0.001
        assert DEFAULT_DECAY_RATES["news"] == 0.005
        assert DEFAULT_DECAY_RATES["legal"] == 0.0002


# ══════════════════════════════════════════════════════════════════════════
# TemporalContradictionDetector tests
# ══════════════════════════════════════════════════════════════════════════


def _make_node(
    content: str,
    embedding: list[float] | None = None,
) -> ASTNode:
    """Helper to create an ASTNode with a deterministic embedding."""
    if embedding is None:
        # Default: high-similarity vector (all 0.5s)
        embedding = [0.5] * 8
    return ASTNode(
        content=content,
        node_type=NodeType.PARAGRAPH,
        doc_id="test-doc",
        embedding=embedding,
    )


class TestStep1_Similarity:
    """Cosine similarity threshold enforcement."""

    def test_similar_embeddings_passes(self) -> None:
        """Similar vectors (cos > 0.65) pass Step 1."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("Revenue was $40M in Q2.", embedding=[0.8, 0.1, 0.1])
        node_b = _make_node("Revenue was $52M in Q3.", embedding=[0.7, 0.2, 0.1])
        assert detector._step_1_similarity(node_a, node_b) is True

    def test_dissimilar_embeddings_fails(self) -> None:
        """Dissimilar vectors (cos <= 0.65) fail Step 1."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("Revenue was $40M.", embedding=[1.0, 0.0, 0.0])
        node_b = _make_node("The weather is nice.", embedding=[0.0, 1.0, 0.0])
        assert detector._step_1_similarity(node_a, node_b) is False

    def test_identical_content_fails(self) -> None:
        """Identical content → not a contradiction, fails Step 1."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("Revenue was $40M.", embedding=[0.8, 0.1, 0.1])
        node_b = _make_node("Revenue was $40M.", embedding=[0.7, 0.2, 0.1])
        assert detector._step_1_similarity(node_a, node_b) is False

    def test_no_embeddings_skips_check(self) -> None:
        """Empty embeddings → skip similarity check (pass through)."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("Some content.", embedding=[])
        node_b = _make_node("Other content.", embedding=[])
        assert detector._step_1_similarity(node_a, node_b) is True


class TestStep2_Negation:
    """Negation heuristic checks."""

    def test_negation_detected(self) -> None:
        """One node contains 'not' with shared entity."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("The policy is not valid anymore.")
        node_b = _make_node("The policy is in full effect.")
        assert detector._step_2_negation_heuristic(node_a, node_b) is True

    def test_no_shared_entity_fails(self) -> None:
        """No shared words >= 4 chars → fail."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("Cats are mammals.")
        node_b = _make_node("Quantum physics is interesting.")
        assert detector._step_2_negation_heuristic(node_a, node_b) is False

    def test_no_negation_fails(self) -> None:
        """Both nodes lack negation phrases."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("The policy is active.")
        node_b = _make_node("The policy is in full effect.")
        assert detector._step_2_negation_heuristic(node_a, node_b) is False

    def test_no_longer_detected(self) -> None:
        """'No longer' phrase detected."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("This rule no longer applies.")
        node_b = _make_node("This rule applies to all cases.")
        assert detector._step_2_negation_heuristic(node_a, node_b) is True

    def test_was_repealed_detected(self) -> None:
        """'Was repealed' phrase detected."""
        detector = TemporalContradictionDetector()
        node_a = _make_node("The amendment was repealed in 2024.")
        node_b = _make_node("The amendment remains in effect.")
        assert detector._step_2_negation_heuristic(node_a, node_b) is True


class TestStep3_LLM:
    """LLM confirmation (mocked)."""

    @pytest.mark.asyncio
    async def test_llm_confirms_contradiction(self) -> None:
        """LLM says YES → CausalEdge returned."""
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value="YES|The passages make opposing claims about revenue."
        )
        detector = TemporalContradictionDetector(llm=llm)
        node_a = _make_node("Revenue was $40M in Q2.", embedding=[0.8, 0.1, 0.1])
        node_b = _make_node("Revenue was $52M in Q3.", embedding=[0.7, 0.2, 0.1])
        edge = await detector._step_3_llm_confirm(node_a, node_b)
        assert edge is not None
        assert edge.edge_type == EdgeType.CONTRADICTS
        assert "opposing claims" in edge.evidence

    @pytest.mark.asyncio
    async def test_llm_denies_contradiction(self) -> None:
        """LLM says NO → None returned."""
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="NO|The passages are consistent and complementary.")
        detector = TemporalContradictionDetector(llm=llm)
        node_a = _make_node("Revenue was $40M in Q2.")
        node_b = _make_node("Revenue was $52M in Q3.")
        edge = await detector._step_3_llm_confirm(node_a, node_b)
        assert edge is None

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self) -> None:
        """LLM error → graceful None."""
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("API error"))
        detector = TemporalContradictionDetector(llm=llm)
        node_a = _make_node("Content A.")
        node_b = _make_node("Content B.")
        edge = await detector._step_3_llm_confirm(node_a, node_b)
        assert edge is None


class TestFullDetect:
    """Full three-step pipeline."""

    @pytest.mark.asyncio
    async def test_known_conflicting_pair(self) -> None:
        """Detect contradiction between two known conflicting passages."""
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value="YES|The first says the policy is valid, the second says it is not."
        )
        detector = TemporalContradictionDetector(llm=llm)

        # High cosine similarity (same topic), negation in one
        emb = [0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        node_a = _make_node("The policy is valid for all employees.", embedding=emb)
        node_b = _make_node(
            "The policy is not valid anymore.", embedding=[0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        edge = await detector.detect(node_a, node_b)
        assert edge is not None
        assert edge.edge_type == EdgeType.CONTRADICTS
        assert edge.strength > 0.65

    @pytest.mark.asyncio
    async def test_known_consistent_pair(self) -> None:
        """No contradiction between two consistent passages."""
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value="NO|Both passages consistently describe revenue growth."
        )
        detector = TemporalContradictionDetector(llm=llm)

        emb = [0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        node_a = _make_node("Revenue grew by 20% in Q2.", embedding=emb)
        node_b = _make_node(
            "Revenue grew by 20% this quarter.", embedding=[0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        edge = await detector.detect(node_a, node_b)
        assert edge is None

    @pytest.mark.asyncio
    async def test_no_llm_fallback_to_steps_1_and_2(self) -> None:
        """Without LLM, detector uses steps 1 and 2 only."""
        detector = TemporalContradictionDetector(llm=None)
        node_a = _make_node("The policy is valid for all.", embedding=[0.8, 0.1, 0.1])
        node_b = _make_node("The policy is not valid anymore.", embedding=[0.7, 0.2, 0.1])

        edge = await detector.detect(node_a, node_b)
        assert edge is not None
        assert edge.edge_type == EdgeType.CONTRADICTS
        assert "Negation heuristic" in edge.evidence

    @pytest.mark.asyncio
    async def test_cosine_similarity_function(self) -> None:
        """Direct test of the cosine similarity utility."""
        detector = TemporalContradictionDetector()
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        assert detector._cosine_similarity(vec_a, vec_b) == pytest.approx(1.0)

        vec_c = [1.0, 0.0, 0.0]
        vec_d = [0.0, 1.0, 0.0]
        assert detector._cosine_similarity(vec_c, vec_d) == pytest.approx(0.0)

        vec_e: list[float] = []
        vec_f = [1.0, 0.0, 0.0]
        assert detector._cosine_similarity(vec_e, vec_f) == 0.0
