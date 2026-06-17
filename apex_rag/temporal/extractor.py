"""
temporal/extractor.py — Multi-strategy date extraction from documents.

Extracts ``source_date`` from a document using three strategies in
priority order:

    **Strategy A** — Explicit metadata:
        PDF creation date (via PyMuPDF metadata)
        DOCX last-modified (via python-docx core_properties)
        Markdown frontmatter ``date:`` field

    **Strategy B** — Regex extraction:
        Scans the first 500 characters for date patterns:
        YYYY-MM-DD, DD Month YYYY, Month DD, YYYY, Q3 2024, etc.

    **Strategy C** — LLM fallback:
        Prompts the model to estimate the document's authorship date.
        Returns ``None`` if the model cannot determine with >70 % confidence.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger("apex_rag.temporal.extractor")


# ═══════════════════════════════════════════════════════════════
# Regex patterns for Strategy B
# ═══════════════════════════════════════════════════════════════

_DATE_PATTERNS_B: list[tuple[str, re.Pattern[str], str | None]] = [
    # ISO: 2024-06-01
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "%Y-%m-%d"),
    # US long: June 1, 2024  or  June 01, 2024
    ("us_long", re.compile(r"\b(January|February|March|April|May|June|July|"
                            r"August|September|October|November|December)\s+"
                            r"(\d{1,2}),\s*(\d{4})\b", re.IGNORECASE), None),
    # Short US: Jun 1, 2024  or  Jun 01, 2024
    ("us_short", re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
                            r"(\d{1,2}),\s*(\d{4})\b", re.IGNORECASE), None),
    # DD Month YYYY: 01 June 2024
    ("dd_month_yyyy", re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|"
                                  r"August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE), None),
    # Month YYYY: June 2024
    ("month_yyyy", re.compile(r"\b(January|February|March|April|May|June|July|"
                               r"August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE), None),
    # Quarter: Q3 2024, Q1 2024, Q4 FY2024
    ("quarter", re.compile(r"\bQ([1-4])\s*(?:FY)?(\d{4})\b", re.IGNORECASE), None),
    # Year only: 2024 (4-digit year, 1900-2099)
    ("year_only", re.compile(r"\b(19\d{2}|20\d{2})\b"), None),
    # European: 01/06/2024 or 01-06-2024 (DD/MM/YYYY or DD-MM-YYYY)
    ("european", re.compile(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b"), None),
]

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ── LLMProvider import ────────────────────────────────────────────────

from apex_rag.core.protocols.interfaces import LLMProvider  # noqa: TC001

# ═══════════════════════════════════════════════════════════════
# TemporalExtractor
# ═══════════════════════════════════════════════════════════════


class TemporalExtractor:
    """Extracts the ``source_date`` from a document using three strategies.

    Strategies are tried in priority order:

        1. **Explicit metadata** — PDF creation date, DOCX last-modified,
           Markdown frontmatter ``date:`` field.
        2. **Regex extraction** — common date formats in the first 500
           characters of the text.
        3. **LLM fallback** — prompts the model to estimate the date
           from content; returns ``None`` if confidence is < 70 %.

    Usage::

        extractor = TemporalExtractor()
        date = await extractor.extract(
            text_content="# Q3 Report\\n\\n...",
            file_metadata={"creation_date": "2024-10-15"},
        )
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    # ── Public API ─────────────────────────────────────────────────────────

    async def extract(
        self,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> datetime | None:
        """Run all three strategies in order, returning the first match.

        Args:
            text:     The document's raw text content.
            metadata: Optional dict with keys like ``creation_date``,
                      ``last_modified``, ``date`` (frontmatter).

        Returns:
            A timezone-aware ``datetime``, or ``None`` if no strategy
            could determine a date.
        """
        # Strategy A — explicit metadata
        if metadata:
            result = self._strategy_a(metadata)
            if result is not None:
                logger.info("Strategy A (metadata) → %s", result.date())
                return result

        # Strategy B — regex
        result = self._strategy_b(text)
        if result is not None:
            logger.info("Strategy B (regex) → %s", result.date())
            return result

        # Strategy C — LLM fallback
        if self._llm is not None:
            result = await self._strategy_c(text)
            if result is not None:
                logger.info("Strategy C (LLM) → %s", result.date())
                return result

        logger.info("No date could be extracted.")
        return None

    # ── Strategy A: Explicit metadata ──────────────────────────────────────

    def _strategy_a(self, metadata: dict[str, str]) -> datetime | None:
        """Try to extract the date from explicit metadata fields."""
        # Ordered by reliability
        for key in ("creation_date", "created", "last_modified", "modified", "date"):
            value = metadata.get(key)
            if not value:
                continue

            dt = self._parse_iso_like(value)
            if dt is not None:
                return dt

        return None

    # ── Strategy B: Regex extraction ───────────────────────────────────────

    def _strategy_b(self, text: str, chars: int = 500) -> datetime | None:
        """Scan the first *chars* characters with regex patterns."""
        window = text[:chars]

        for name, pattern, fmt in _DATE_PATTERNS_B:
            match = pattern.search(window)
            if not match:
                continue

            dt = self._regex_to_datetime(match, name, fmt)
            if dt is not None:
                return dt

        return None

    # ── Strategy C: LLM fallback ───────────────────────────────────────────

    async def _strategy_c(self, text: str) -> datetime | None:
        """Use the LLM to estimate the document's authorship date."""
        if self._llm is None:
            return None

        preview = text[:2000]

        prompt = f"""You are a document dating assistant.  Given the following document
text, estimate the date when it was most likely authored or published.

Respond with ONLY a date in YYYY-MM-DD format and a confidence percentage
between 0 and 100 separated by a pipe.  If you cannot determine with >70 %
confidence, respond with "None".

Examples:
  2024-06-01|95
  2023-12-15|80
  None

Document text:
---START---
{preview}
---END---"""

        try:
            response = await self._llm.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=30,
            )
        except Exception as exc:
            logger.warning("LLM date extraction failed: %s", exc)
            return None

        response = response.strip()

        # Parse "YYYY-MM-DD|confidence" or "None"
        if response.lower().startswith("none"):
            return None

        llm_match = re.match(r"(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)", response)
        if not llm_match:
            return None

        confidence = int(llm_match.group(2))
        if confidence < 70:
            return None

        try:
            return datetime.strptime(llm_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_iso_like(value: str) -> datetime | None:
        """Try to parse a date string as ISO-like or common formats."""
        # ISO: 2024-06-01 or 2024-06-01T12:00:00
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ):
            try:
                dt = datetime.strptime(value[:19], fmt)
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except ValueError:
                continue
        return None

    @staticmethod
    def _regex_to_datetime(
        match: re.Match[str],
        name: str,
        fmt: str | None,
    ) -> datetime | None:
        """Convert a regex match to a timezone-aware datetime."""
        groups = match.groups()

        # Try standard format first
        if fmt is not None:
            try:
                dt = datetime.strptime(match.group(0), fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # ── Dispatch by pattern name ────────────────────────────────────────

        if name == "iso":
            try:
                dt = datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return None

        if name == "european":
            # Try DD/MM/YYYY first (European)
            try:
                dt = datetime(int(groups[2]), int(groups[1]), int(groups[0]))
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
            # Try MM/DD/YYYY
            try:
                dt = datetime(int(groups[2]), int(groups[0]), int(groups[1]))
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return None

        if name == "quarter":
            try:
                quarter = int(groups[0])
                year = int(groups[1])
                month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
                dt = datetime(year, month, 1)
                return dt.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError, IndexError):
                return None

        if name == "year_only":
            try:
                dt = datetime(int(groups[0]), 1, 1)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return None

        if name in ("us_long", "us_short", "dd_month_yyyy", "month_yyyy"):
            # Month DD, YYYY  or  DD Month YYYY  or  Month YYYY
            month_val = None
            day_val = None
            year_val = None
            for g in groups:
                g_lower = str(g).lower() if g else ""
                if g_lower in _MONTH_NAMES:
                    month_val = _MONTH_NAMES[g_lower]
                elif g and len(str(g)) == 4 and 1900 <= int(g) <= 2099:
                    year_val = int(g)
                elif g and str(g).isdigit() and 1 <= int(g) <= 31:
                    day_val = int(g)

            if month_val is not None and year_val is not None:
                if day_val is not None:
                    try:
                        dt = datetime(year_val, month_val, day_val)
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass
                dt = datetime(year_val, month_val, 1)
                return dt.replace(tzinfo=timezone.utc)

        return None
