"""
tests/test_metric_parser.py — Tests for the MetricValueParser (Phase 3).
"""

from __future__ import annotations

from apex_rag.core.metrics.parser import MetricValueParser, UnitType


class TestMetricValueParser:
    """Tests for the MetricValueParser."""

    def setup_method(self) -> None:
        self.parser = MetricValueParser()

    def test_parse_currency_dollar(self) -> None:
        result = self.parser.parse("$100")
        assert result.numeric_value == 100.0
        assert result.unit_type == UnitType.CURRENCY
        assert result.confidence == 1.0

    def test_parse_currency_dollar_k(self) -> None:
        result = self.parser.parse("$100k")
        assert result.numeric_value == 100_000.0
        assert result.unit_type == UnitType.CURRENCY
        assert result.confidence == 1.0

    def test_parse_currency_dollar_million(self) -> None:
        result = self.parser.parse("$2.5M")
        assert result.numeric_value == 2_500_000.0
        assert result.unit_type == UnitType.CURRENCY
        assert result.confidence == 1.0

    def test_parse_currency_inr_lakh(self) -> None:
        result = self.parser.parse("₹10L")
        assert result.numeric_value == 1_000_000.0  # 10 * 100,000
        assert result.unit_type == UnitType.CURRENCY
        assert result.confidence == 1.0

    def test_parse_currency_inr_crore(self) -> None:
        result = self.parser.parse("₹2Cr")
        assert result.numeric_value == 20_000_000.0  # 2 * 10,000,000
        assert result.unit_type == UnitType.CURRENCY
        assert result.confidence == 1.0

    def test_parse_percentage(self) -> None:
        result = self.parser.parse("35%")
        assert result.numeric_value == 0.35
        assert result.unit_type == UnitType.PERCENTAGE
        assert result.confidence == 1.0

    def test_parse_negative_percentage(self) -> None:
        result = self.parser.parse("-5%")
        assert result.numeric_value == -0.05
        assert result.unit_type == UnitType.PERCENTAGE
        assert result.confidence == 1.0

    def test_parse_unknown_na(self) -> None:
        result = self.parser.parse("N/A")
        assert result.numeric_value == 0.0
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_unknown_text(self) -> None:
        result = self.parser.parse("Unknown")
        assert result.numeric_value == 0.0
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_not_available(self) -> None:
        result = self.parser.parse("Not Available")
        assert result.numeric_value == 0.0
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_text_million(self) -> None:
        result = self.parser.parse("5 million")
        assert result.numeric_value == 5_000_000.0
        assert result.unit_type == UnitType.NUMERIC
        assert result.confidence == 1.0

    def test_parse_text_billion(self) -> None:
        result = self.parser.parse("2 billion")
        assert result.numeric_value == 2_000_000_000.0
        assert result.unit_type == UnitType.NUMERIC

    def test_parse_text_lakh(self) -> None:
        result = self.parser.parse("10 lakh")
        assert result.numeric_value == 1_000_000.0
        assert result.unit_type == UnitType.NUMERIC

    def test_parse_plain_numeric(self) -> None:
        result = self.parser.parse("42")
        assert result.numeric_value == 42.0
        assert result.unit_type == UnitType.NUMERIC
        assert result.confidence == 1.0

    def test_parse_plain_negative_numeric(self) -> None:
        result = self.parser.parse("-15")
        assert result.numeric_value == -15.0
        assert result.unit_type == UnitType.NUMERIC

    def test_parse_empty_string(self) -> None:
        result = self.parser.parse("")
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_gibberish(self) -> None:
        result = self.parser.parse("foo bar baz")
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_none_str(self) -> None:
        result = self.parser.parse("None")
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_null_str(self) -> None:
        result = self.parser.parse("null")
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_tbd(self) -> None:
        result = self.parser.parse("TBD")
        assert result.unit_type == UnitType.UNKNOWN
        assert result.confidence == 0.0
