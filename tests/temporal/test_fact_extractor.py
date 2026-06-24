"""
tests/temporal/test_fact_extractor.py — Tests for fact_extractor.py.

Covers all 9 extraction strategies:
  1. Metrics (currency, percentage, numeric)
  2. Policies (subject + modal + condition)
  3. Organizations
  4. People
  5. Products
  6. Events
  7. Contract terms
  8. Compliance/regulations
  9. Date ranges

Also tests empty content, provenance fields, and extract_from_text.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from apex_rag.models.unified_models import ASTNode, NodeType
from apex_rag.temporal.fact_extractor import FactExtractor
from apex_rag.temporal.fact_store import TemporalFact


# ── Helpers ──────────────────────────────────────────────────────────────


def make_node(content: str, node_id: str | None = None) -> ASTNode:
    """Create a simple PARAGRAPH ASTNode."""
    return ASTNode(
        node_id=node_id or str(uuid.uuid4()),
        content=content,
        node_type=NodeType.PARAGRAPH,
        doc_id="test-doc",
        source_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
    )


def make_empty_node() -> ASTNode:
    """Create an ASTNode with empty content."""
    return ASTNode(
        node_id=str(uuid.uuid4()),
        content="",
        node_type=NodeType.PARAGRAPH,
        doc_id="test-doc",
    )


# ── Tests ────────────────────────────────────────────────────────────────


class TestFactExtractorEmptyContent:
    """Extracting from empty/blank content should return empty list."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_empty(self) -> None:
        extractor = FactExtractor()
        node = make_empty_node()
        facts = await extractor.extract_from_node(node)
        assert facts == []

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_empty(self) -> None:
        extractor = FactExtractor()
        node = make_node("   \n  \t  ")
        facts = await extractor.extract_from_node(node)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_from_text_empty(self) -> None:
        extractor = FactExtractor()
        facts = await extractor.extract_from_text("", doc_id="test-doc")
        assert facts == []


class TestMetricsExtraction:
    """Strategy 1: Metrics (currency, percentage, numeric values)."""

    @pytest.mark.asyncio
    async def test_revenue_usd(self) -> None:
        extractor = FactExtractor()
        node = make_node("Revenue was $120,000 in Q3")
        facts = await extractor.extract_from_node(node)
        metric_facts = [f for f in facts if f.metadata.get("fact_type") == "metric"]
        assert len(metric_facts) >= 1
        rev_fact = metric_facts[0]
        assert "Revenue" in rev_fact.subject
        assert rev_fact.predicate == "was"

    @pytest.mark.asyncio
    async def test_revenue_with_currency_symbol(self) -> None:
        extractor = FactExtractor()
        node = make_node("Revenue: $120,000.00")
        facts = await extractor.extract_from_node(node)
        metric_facts = [f for f in facts if f.metadata.get("fact_type") == "metric"]
        assert len(metric_facts) >= 1

    @pytest.mark.asyncio
    async def test_percentage_value(self) -> None:
        extractor = FactExtractor()
        node = make_node("Growth rate is 15% year over year")
        facts = await extractor.extract_from_node(node)
        metric_facts = [f for f in facts if f.metadata.get("fact_type") == "metric"]
        assert len(metric_facts) >= 1

    @pytest.mark.asyncio
    async def test_multiple_metrics(self) -> None:
        extractor = FactExtractor()
        node = make_node("Revenue was $40M. Profit was $10M. Headcount is 500.")
        facts = await extractor.extract_from_node(node)
        metric_facts = [f for f in facts if f.metadata.get("fact_type") == "metric"]
        assert len(metric_facts) >= 3

    @pytest.mark.asyncio
    async def test_metric_has_provenance(self) -> None:
        extractor = FactExtractor()
        node = make_node("Revenue was $120,000")
        facts = await extractor.extract_from_node(node, doc_id="doc-x", tenant_id="tenant-a")
        assert all(f.source_document_id == "doc-x" for f in facts)
        assert all(f.tenant_id == "tenant-a" for f in facts)
        assert all(f.extraction_method == "regex" for f in facts)


class TestPolicyExtraction:
    """Strategy 2: Policy statements."""

    @pytest.mark.asyncio
    async def test_policy_shall(self) -> None:
        extractor = FactExtractor()
        node = make_node("All employees shall submit expense reports within 30 days.")
        facts = await extractor.extract_from_node(node)
        policy_facts = [f for f in facts if f.metadata.get("fact_type") == "policy"]
        assert len(policy_facts) >= 1
        assert "employee" in policy_facts[0].subject.lower()
        assert policy_facts[0].predicate == "shall"

    @pytest.mark.asyncio
    async def test_policy_must(self) -> None:
        extractor = FactExtractor()
        node = make_node("The manager must approve all overtime requests in writing.")
        facts = await extractor.extract_from_node(node)
        policy_facts = [f for f in facts if f.metadata.get("fact_type") == "policy"]
        assert len(policy_facts) >= 1

    @pytest.mark.asyncio
    async def test_policy_is_required(self) -> None:
        extractor = FactExtractor()
        node = make_node("Each department is required to submit a quarterly budget.")
        facts = await extractor.extract_from_node(node)
        policy_facts = [f for f in facts if f.metadata.get("fact_type") == "policy"]
        assert len(policy_facts) >= 1


class TestOrganizationExtraction:
    """Strategy 3: Organization detection."""

    @pytest.mark.asyncio
    async def test_corporation_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("Acme Corporation announced record profits.")
        facts = await extractor.extract_from_node(node)
        org_facts = [f for f in facts if f.metadata.get("fact_type") == "organization"]
        assert len(org_facts) >= 1
        assert "Acme" in org_facts[0].subject

    @pytest.mark.asyncio
    async def test_inc_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("GlobalTech Inc. is a leading provider.")
        facts = await extractor.extract_from_node(node)
        org_facts = [f for f in facts if f.metadata.get("fact_type") == "organization"]
        assert len(org_facts) >= 1
        assert "GlobalTech" in org_facts[0].subject

    @pytest.mark.asyncio
    async def test_llc_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("Partnership with BlueSky LLC was finalized.")
        facts = await extractor.extract_from_node(node)
        org_facts = [f for f in facts if f.metadata.get("fact_type") == "organization"]
        assert len(org_facts) >= 1
        assert "BlueSky" in org_facts[0].subject

    @pytest.mark.asyncio
    async def test_no_duplicate_organizations(self) -> None:
        extractor = FactExtractor()
        node = make_node("Acme Corp and Acme Corp are the same company. Acme Corp leads.")
        facts = await extractor.extract_from_node(node)
        org_facts = [f for f in facts if f.metadata.get("fact_type") == "organization"]
        # Should only have one Acme Corp fact despite 3 mentions
        acme_facts = [f for f in org_facts if "Acme" in f.subject]
        assert len(acme_facts) == 1


class TestPersonExtraction:
    """Strategy 4: Person detection."""

    @pytest.mark.asyncio
    async def test_person_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("John Doe was appointed as CEO.")
        facts = await extractor.extract_from_node(node)
        person_facts = [f for f in facts if f.metadata.get("fact_type") == "person"]
        assert len(person_facts) >= 1
        assert "John" in person_facts[0].subject

    @pytest.mark.asyncio
    async def test_person_with_title(self) -> None:
        extractor = FactExtractor()
        node = make_node("Dr. Jane Smith presented the findings.")
        facts = await extractor.extract_from_node(node)
        person_facts = [f for f in facts if f.metadata.get("fact_type") == "person"]
        assert len(person_facts) >= 1
        combined = " ".join(f.subject for f in person_facts)
        assert "Jane" in combined or "Smith" in combined

    @pytest.mark.asyncio
    async def test_no_duplicate_people(self) -> None:
        extractor = FactExtractor()
        node = make_node("John Doe spoke. John Doe also attended.")
        facts = await extractor.extract_from_node(node)
        person_facts = [f for f in facts if f.metadata.get("fact_type") == "person"]
        john_facts = [f for f in person_facts if "John" in f.subject]
        assert len(john_facts) == 1


class TestProductExtraction:
    """Strategy 5: Product detection."""

    @pytest.mark.asyncio
    async def test_pascal_case_product(self) -> None:
        extractor = FactExtractor()
        node = make_node("The new SuperWidget is now available.")
        facts = await extractor.extract_from_node(node)
        product_facts = [f for f in facts if f.metadata.get("fact_type") == "product"]
        assert len(product_facts) >= 1

    @pytest.mark.asyncio
    async def test_product_suffix(self) -> None:
        extractor = FactExtractor()
        node = make_node("We launched the Analytics Platform v2.")
        facts = await extractor.extract_from_node(node)
        product_facts = [f for f in facts if f.metadata.get("fact_type") == "product"]
        assert len(product_facts) >= 1


class TestEventExtraction:
    """Strategy 6: Event detection."""

    @pytest.mark.asyncio
    async def test_event_with_date(self) -> None:
        extractor = FactExtractor()
        node = make_node("The conference was held on 2025-06-15 in Chicago.")
        facts = await extractor.extract_from_node(node)
        event_facts = [f for f in facts if f.metadata.get("fact_type") == "event"]
        assert len(event_facts) >= 1
        assert event_facts[0].predicate == "occurred_on"

    @pytest.mark.asyncio
    async def test_event_with_named_month(self) -> None:
        extractor = FactExtractor()
        node = make_node("On January 15, 2025 the board meeting took place.")
        facts = await extractor.extract_from_node(node)
        event_facts = [f for f in facts if f.metadata.get("fact_type") == "event"]
        assert len(event_facts) >= 1


class TestContractTermExtraction:
    """Strategy 7: Contract terms."""

    @pytest.mark.asyncio
    async def test_term_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("Term: 12 months from the effective date.")
        facts = await extractor.extract_from_node(node)
        contract_facts = [f for f in facts if f.metadata.get("fact_type") == "contract_term"]
        assert len(contract_facts) >= 1

    @pytest.mark.asyncio
    async def test_payment_term(self) -> None:
        extractor = FactExtractor()
        node = make_node("Payment: Net 30 days after invoice.")
        facts = await extractor.extract_from_node(node)
        contract_facts = [f for f in facts if f.metadata.get("fact_type") == "contract_term"]
        assert len(contract_facts) >= 1


class TestRegulationExtraction:
    """Strategy 8: Compliance/regulations."""

    @pytest.mark.asyncio
    async def test_gdpr_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("All data processing must comply with GDPR requirements.")
        facts = await extractor.extract_from_node(node)
        reg_facts = [f for f in facts if f.metadata.get("fact_type") == "regulation"]
        assert len(reg_facts) >= 1
        assert "GDPR" in reg_facts[0].subject

    @pytest.mark.asyncio
    async def test_hipaa_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("Patient data is protected under HIPAA regulations.")
        facts = await extractor.extract_from_node(node)
        reg_facts = [f for f in facts if f.metadata.get("fact_type") == "regulation"]
        assert len(reg_facts) >= 1
        assert "HIPAA" in reg_facts[0].subject

    @pytest.mark.asyncio
    async def test_sox_detected(self) -> None:
        extractor = FactExtractor()
        node = make_node("Financial reporting follows SOX guidelines.")
        facts = await extractor.extract_from_node(node)
        reg_facts = [f for f in facts if f.metadata.get("fact_type") == "regulation"]
        assert len(reg_facts) >= 1
        assert "SOX" in reg_facts[0].subject


class TestDateRangeExtraction:
    """Strategy 9: Date ranges."""

    @pytest.mark.asyncio
    async def test_from_to_date_range(self) -> None:
        extractor = FactExtractor()
        node = make_node("The contract covers from 2024 to 2026.")
        facts = await extractor.extract_from_node(node)
        range_facts = [f for f in facts if f.metadata.get("fact_type") == "date_range"]
        assert len(range_facts) >= 1
        assert range_facts[0].predicate == "covers"

    @pytest.mark.asyncio
    async def test_between_and_date_range(self) -> None:
        extractor = FactExtractor()
        node = make_node("The policy applies between 2023 and 2025.")
        facts = await extractor.extract_from_node(node)
        range_facts = [f for f in facts if f.metadata.get("fact_type") == "date_range"]
        assert len(range_facts) >= 1


class TestFullExtractionPipeline:
    """Integration-style tests running all strategies together."""

    @pytest.mark.asyncio
    async def test_extract_from_rich_content(self) -> None:
        """Extract multiple fact types from a single rich paragraph."""
        extractor = FactExtractor()
        text = (
            "Acme Corp reported Revenue was $50M in Q2 2025. "
            "All employees shall follow GDPR guidelines. "
            "Dr. Jane Smith was appointed as CFO."
        )
        node = make_node(text)
        facts = await extractor.extract_from_node(node)
        fact_types = {f.metadata.get("fact_type") for f in facts}
        assert "metric" in fact_types
        assert "regulation" in fact_types
        assert len(facts) >= 3

    @pytest.mark.asyncio
    async def test_provenance_fields(self) -> None:
        """Every extracted fact has provenance (Principle 17)."""
        extractor = FactExtractor()
        node = make_node("Revenue was $40M. Policy must be followed. Acme Corp leads.")
        facts = await extractor.extract_from_node(
            node, doc_id="doc-provenance", tenant_id="tenant-p",
        )
        for fact in facts:
            assert fact.source_document_id == "doc-provenance"
            assert fact.tenant_id == "tenant-p"
            assert fact.source_node_id == node.node_id
            assert fact.extraction_method == "regex"
            assert fact.created_at is not None

    @pytest.mark.asyncio
    async def test_extract_from_text_convenience(self) -> None:
        """extract_from_text wraps content in a temporary node."""
        extractor = FactExtractor()
        facts = await extractor.extract_from_text(
            "Revenue was $40M", doc_id="doc-txt",
        )
        assert len(facts) >= 1
        assert all(f.source_document_id == "doc-txt" for f in facts)
