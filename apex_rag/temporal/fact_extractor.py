"""
temporal/fact_extractor.py — Multi-strategy fact extraction from AST nodes.

Extracts structured facts from document content using regex-based strategies
for metrics, dates, policies, contracts, organizations, people, products,
compliance rules, regulations, and events.

PRINCIPLE 2 — Non-Blocking Fact Extraction.
  Extraction runs asynchronously and is designed to be invoked from a
  background worker (FactPipeline), not synchronously during ingestion.

PRINCIPLE 17 — Provenance Required.
  Every extracted fact includes source_node_id, document_id, tenant_id,
  confidence, extraction_method, and created_at.

Reuses existing parsers:
  - MetricValueParser (core/metrics/parser.py) for currency/percentage/numeric extraction
  - TemporalExtractor (temporal/extractor.py) for date extraction

Usage:
    extractor = FactExtractor()
    facts = await extractor.extract_from_node(node, doc_id="doc-123", tenant_id="tenant-a")
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from apex_rag.core.metrics.parser import MetricValueParser, UnitType
from apex_rag.models.unified_models import ASTNode, NodeType
from apex_rag.temporal.fact_store import TemporalFact

logger = logging.getLogger("apex_rag.temporal.fact_extractor")


# ═══════════════════════════════════════════════════════════════
# Regex Patterns
# ═══════════════════════════════════════════════════════════════

# Policy / Compliance keywords: subject + modal + condition
_RE_POLICY = re.compile(
    r"(?P<subject>[A-Z][A-Za-z\s]{2,50}?)"
    r"\s+(shall|must|will|should|may|is\s+required\s+to|is\s+responsible\s+for)"
    r"\s+(?P<condition>.{5,200}?)(?:\.|;|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Organization detection: "Acme Corp", "Acme Corporation", "GlobalTech Inc."
# Negative lookahead prevents the name group from consuming suffix words.
_RE_ORGANIZATION = re.compile(
    r"""
    \b
    (                                         # Group 1: company name
        [A-Z][a-zA-Z]+                        # First word (may be camelCase)
        (?:                                    # Optional additional words
            \s
            (?!                                # Negative lookahead: NOT a suffix
                (?:Corporation|Corp|Inc|LLC|Ltd|PLC|GmbH|SA|NV|AG|Co|Group)\b
            )
            [A-Z][a-zA-Z]+                     # Another word
        )*
    )
    \s+
    (Corporation|Corp|Inc|LLC|Ltd|PLC|GmbH|SA|NV|AG|Co|Group)  # Group 2: suffix
    \b
    """,
    re.VERBOSE,
)

# People detection: "John Doe", "Dr. Jane Smith"
_RE_PERSON = re.compile(
    r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)?\s*([A-Z][a-z]+)\s([A-Z][a-z]{1,20})\b"
)

# Product detection: "Product X", "SuperWidget 3000", "Analytics Platform"
_RE_PRODUCT = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:[A-Z][a-z]+)+)\b"  # PascalCase
    r"|\b(\w*(?:Widget|Tool|System|Platform|Suite|Engine))\b"  # Suffix-based
)

# Event detection: date + event pattern
_RE_EVENT = re.compile(
    r"(?:on|in|during)\s+"
    r"(?P<date>(?:\d{4}-\d{2}-\d{2}|Q[1-4]\s+\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}))\s*,?\s*"
    r"(?P<event>.{5,150}?)(?:\.|;|$)",
    re.IGNORECASE,
)

# Contract terms: "Term: 12 months", "Payment: Net 30"
_RE_CONTRACT_TERM = re.compile(
    r"(?:Term|Duration|Period|Payment|Notice|Cancellation)"
    r"\s*[:=]?\s*(.{5,100}?)(?:\.|;|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Revenue / financial metrics: "$X", "₹Y", "X% growth"
_RE_METRIC_LINE = re.compile(
    r"(?P<metric>[A-Za-z\s]{2,40}?)\s*(?:is|was|were|:|=)\s*"
    r"(?P<value>[₹$€£¥]?\s*[\d,]+\.?\d*\s*[kKmMbBtT%]?)",
    re.IGNORECASE | re.MULTILINE,
)

# Date range patterns: "from YYYY to YYYY", "between YYYY and YYYY"
_RE_DATE_RANGE = re.compile(
    r"(?:from|between)\s+(?P<start>\d{4}(?:-\d{2}-\d{2})?)"
    r"\s+(?:to|and|through)\s+(?P<end>\d{4}(?:-\d{2}-\d{2})?)",
    re.IGNORECASE,
)

# Compliance/regulation references
_RE_REGULATION = re.compile(
    r"\b(?:GDPR|HIPAA|SOX|PCI|DSS|ISO\s*\d+|FASB|IFRS|GAAP|"
    r"ESG|KYC|AML|CCPA|ADA|OSHA|EPA)\b",
    re.IGNORECASE,
)


class FactExtractor:
    """Extracts structured :class:`TemporalFact` objects from AST node content.

    Strategies are run in priority order:
      1. Metrics (currency, percentage, numeric values)
      2. Policies (subject + modal + condition)
      3. Dates and events
      4. Organizations
      5. People
      6. Products
      7. Contract terms
      8. Compliance/regulations
      9. Date ranges

    Each extracted fact includes full provenance tracking (Principle 17).
    """

    def __init__(self) -> None:
        self._metric_parser = MetricValueParser()

    # ── Public API ─────────────────────────────────────────────────────

    async def extract_from_node(
        self,
        node: ASTNode,
        *,
        doc_id: str | None = None,
        tenant_id: str = "default",
    ) -> list[TemporalFact]:
        """Extract all facts from a single AST node.

        Args:
            node:      The AST node to extract facts from.
            doc_id:    Override document ID (defaults to node.doc_id).
            tenant_id: Tenant isolation boundary.

        Returns:
            A list of :class:`TemporalFact` objects, one per extracted fact.
        """
        facts: list[TemporalFact] = []
        content = node.content
        resolved_doc_id = doc_id or node.doc_id
        now = datetime.now(timezone.utc)
        source_date = node.source_date or now

        if not content or not content.strip():
            return facts

        # 1. Extract metrics (revenue, financial values)
        facts.extend(self._extract_metrics(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 2. Extract policies
        facts.extend(self._extract_policies(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 3. Extract organizations
        facts.extend(self._extract_organizations(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 4. Extract people
        facts.extend(self._extract_people(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 5. Extract products
        facts.extend(self._extract_products(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 6. Extract events
        facts.extend(self._extract_events(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 7. Extract contract terms
        facts.extend(self._extract_contract_terms(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 8. Extract compliance/regulations
        facts.extend(self._extract_regulations(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        # 9. Extract date ranges
        facts.extend(self._extract_date_ranges(content, resolved_doc_id, node.node_id, tenant_id, source_date, now))

        logger.debug("Extracted %d facts from node %s", len(facts), node.node_id)
        return facts

    async def extract_from_text(
        self,
        text: str,
        *,
        doc_id: str,
        node_id: str | None = None,
        tenant_id: str = "default",
        source_date: datetime | None = None,
    ) -> list[TemporalFact]:
        """Extract facts from raw text content.

        Args:
            text:        The raw text content.
            doc_id:      The document ID.
            node_id:     The source node ID (optional — auto-generated if None).
            tenant_id:   Tenant isolation boundary.
            source_date: When the source was authored.

        Returns:
            Extracted facts.
        """
        node = ASTNode(
            content=text,
            node_type=NodeType.PARAGRAPH,
            doc_id=doc_id,
            node_id=node_id or str(uuid.uuid4()),
            source_date=source_date,
        )
        return await self.extract_from_node(
            node, doc_id=doc_id, tenant_id=tenant_id,
        )

    # ── Strategy: Metrics ──────────────────────────────────────────────

    def _extract_metrics(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        for match in _RE_METRIC_LINE.finditer(content):
            metric_name = match.group("metric").strip()
            raw_value = match.group("value").strip()

            parsed = self._metric_parser.parse(raw_value)
            if parsed.unit_type == UnitType.UNKNOWN:
                # Still record as a text fact with lower confidence
                fact = TemporalFact(
                    subject=metric_name,
                    predicate="was",
                    object=raw_value,
                    confidence=0.5,
                    source_document_id=doc_id,
                    source_node_id=node_id,
                    tenant_id=tenant_id,
                    valid_from=source_date,
                    created_at=now,
                    extraction_method="regex",
                    metadata={
                        "fact_type": "metric",
                        "raw_value": raw_value,
                        "parse_confidence": parsed.confidence,
                        "unit_type": parsed.unit_type.value,
                    },
                )
            else:
                fact = TemporalFact(
                    subject=metric_name,
                    predicate="was",
                    object=str(parsed.numeric_value) if parsed.unit_type == UnitType.NUMERIC
                           else f"{parsed.numeric_value} ({raw_value})",
                    confidence=parsed.confidence,
                    source_document_id=doc_id,
                    source_node_id=node_id,
                    tenant_id=tenant_id,
                    valid_from=source_date,
                    created_at=now,
                    extraction_method="regex",
                    metadata={
                        "fact_type": "metric",
                        "raw_value": raw_value,
                        "numeric_value": parsed.numeric_value,
                        "unit_type": parsed.unit_type.value,
                    },
                )
            facts.append(fact)
        return facts

    # ── Strategy: Policies ─────────────────────────────────────────────

    def _extract_policies(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        for match in _RE_POLICY.finditer(content):
            fact = TemporalFact(
                subject=match.group("subject").strip(),
                predicate="shall",
                object=match.group("condition").strip(),
                confidence=0.8,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={"fact_type": "policy"},
            )
            facts.append(fact)
        return facts

    # ── Strategy: Organizations ────────────────────────────────────────

    def _extract_organizations(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        seen: set[str] = set()
        for match in _RE_ORGANIZATION.finditer(content):
            org_name = match.group(0).strip()
            if org_name.lower() in seen:
                continue
            seen.add(org_name.lower())
            fact = TemporalFact(
                subject=org_name,
                predicate="is",
                object="organization",
                confidence=0.7,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={"fact_type": "organization"},
            )
            facts.append(fact)
        return facts

    # ── Strategy: People ───────────────────────────────────────────────

    def _extract_people(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        seen: set[str] = set()
        for match in _RE_PERSON.finditer(content):
            full_name = f"{match.group(1)} {match.group(2)}"
            if full_name.lower() in seen:
                continue
            seen.add(full_name.lower())
            fact = TemporalFact(
                subject=full_name,
                predicate="is",
                object="person",
                confidence=0.6,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={"fact_type": "person"},
            )
            facts.append(fact)
        return facts

    # ── Strategy: Products ─────────────────────────────────────────────

    def _extract_products(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        seen: set[str] = set()
        for match in _RE_PRODUCT.finditer(content):
            product_name = (match.group(1) or match.group(2) or "").strip()
            if not product_name or product_name.lower() in seen:
                continue
            seen.add(product_name.lower())
            fact = TemporalFact(
                subject=product_name,
                predicate="is",
                object="product",
                confidence=0.5,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={"fact_type": "product"},
            )
            facts.append(fact)
        return facts

    # ── Strategy: Events ───────────────────────────────────────────────

    def _extract_events(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        for match in _RE_EVENT.finditer(content):
            event_date = match.group("date")
            event_desc = match.group("event").strip()
            fact = TemporalFact(
                subject=event_desc,
                predicate="occurred_on",
                object=event_date,
                confidence=0.7,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={
                    "fact_type": "event",
                    "event_date": event_date,
                },
            )
            facts.append(fact)
        return facts

    # ── Strategy: Contract Terms ───────────────────────────────────────

    def _extract_contract_terms(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        for match in _RE_CONTRACT_TERM.finditer(content):
            term_value = match.group(1).strip()
            # Determine term type from context
            text_before = content[:match.start()]
            term_type = "contract_term"
            if "term" in text_before.lower():
                term_type = "term"
            elif "payment" in text_before.lower():
                term_type = "payment"
            elif "notice" in text_before.lower():
                term_type = "notice"
            elif "duration" in text_before.lower() or "period" in text_before.lower():
                term_type = "duration"

            fact = TemporalFact(
                subject=term_type,
                predicate="is",
                object=term_value,
                confidence=0.6,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={
                    "fact_type": "contract_term",
                    "term_type": term_type,
                },
            )
            facts.append(fact)
        return facts

    # ── Strategy: Compliance/Regulations ───────────────────────────────

    def _extract_regulations(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        seen: set[str] = set()
        for match in _RE_REGULATION.finditer(content):
            reg = match.group(0).upper().strip()
            if reg in seen:
                continue
            seen.add(reg)
            fact = TemporalFact(
                subject=reg,
                predicate="applies",
                object="regulation",
                confidence=0.8,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={"fact_type": "regulation"},
            )
            facts.append(fact)
        return facts

    # ── Strategy: Date Ranges ──────────────────────────────────────────

    def _extract_date_ranges(
        self,
        content: str,
        doc_id: str,
        node_id: str,
        tenant_id: str,
        source_date: datetime,
        now: datetime,
    ) -> list[TemporalFact]:
        facts: list[TemporalFact] = []
        for match in _RE_DATE_RANGE.finditer(content):
            start = match.group("start")
            end = match.group("end")
            fact = TemporalFact(
                subject="date_range",
                predicate="covers",
                object=f"{start} to {end}",
                confidence=0.7,
                source_document_id=doc_id,
                source_node_id=node_id,
                tenant_id=tenant_id,
                valid_from=source_date,
                created_at=now,
                extraction_method="regex",
                metadata={
                    "fact_type": "date_range",
                    "range_start": start,
                    "range_end": end,
                },
            )
            facts.append(fact)
        return facts
