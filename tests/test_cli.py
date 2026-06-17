"""
test_cli.py — Unit tests for the ApexRAG CLI module (apex_rag/cli.py).

Tests cover:
  - Banner and help output
  - Error formatting (ApexRAGError, generic Exception)
  - Progress bar and spinner helpers
  - Streaming display generator
  - Doctor check function
  - System info output
  - Formatting helpers (format_success, format_info, format_warning, format_table)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from apex_rag.cli import (
    ProgressBar,
    _check_package,
    _print_system_info,
    display_streaming,
    doctor_check,
    format_error,
    format_info,
    format_success,
    format_table,
    format_warning,
    print_banner,
    print_help,
    repl_loop,
    spinner_context,
)
from apex_rag.exceptions import ApexRAGError, DocumentNotFoundError


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def rich_console() -> Console:
    """A Rich Console that writes to a StringIO for assertion."""
    return Console(file=StringIO(), width=120)


@pytest.fixture
def mock_index() -> MagicMock:
    """A mocked ApexIndex instance for REPL testing."""
    idx = MagicMock()
    idx.list_documents = AsyncMock(return_value=["doc1", "doc2"])
    idx.get_stats = AsyncMock(
        return_value={"total_nodes": 10, "leaf_count": 5, "max_depth": 3}
    )
    idx.stream_query = AsyncMock(
        return_value=_mock_stream(["This ", "is ", "the ", "answer."])
    )
    return idx


async def _mock_stream(chunks: list[str]) -> AsyncGenerator[str, None]:
    for chunk in chunks:
        yield chunk


# ── Banner & Help ─────────────────────────────────────────────────────────


class TestBanner:
    def test_print_banner_contains_version(self) -> None:
        """Banner should include the version string and subtitle."""
        from apex_rag import __version__

        buf = StringIO()
        console = Console(file=buf, width=120)
        with patch("apex_rag.cli.console", console):
            print_banner()
        output = buf.getvalue()
        assert __version__ in output
        assert "Structural AI Retrieval" in output

    def test_print_banner_with_subtitle(self) -> None:
        """Subtitle should appear in banner output."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        with patch("apex_rag.cli.console", console):
            print_banner(subtitle="Custom Mode")
        output = buf.getvalue()
        assert "Custom Mode" in output

    def test_print_help_contains_commands(self) -> None:
        """Help output should mention all CLI commands."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        with patch("apex_rag.cli.console", console):
            print_help()
        output = buf.getvalue()
        for cmd in ["serve", "ingest", "query", "stream", "repl", "doctor", "list", "info"]:
            assert cmd in output

    def test_print_help_shown_when_no_command(self) -> None:
        """Running with no command should print help."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        with patch("apex_rag.cli.console", console):
            from apex_rag.cli import COMMAND_HELP
            console.print(COMMAND_HELP)
        output = buf.getvalue()
        assert "serve" in output
        assert "repl" in output


# ── Error Formatting ──────────────────────────────────────────────────────


class TestErrorFormatting:
    def test_format_apex_rag_error_shows_code_and_hint(self) -> None:
        """ApexRAGError should display code and hint."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        exc = DocumentNotFoundError(
            message="Document 'abc' not found.",
            hint="Use list_documents() to see available documents.",
        )
        with patch("apex_rag.cli.console", console):
            format_error(exc)
        output = buf.getvalue()
        assert "APEX_100" in output
        assert "not found" in output
        assert "list_documents" in output

    def test_format_generic_exception_shows_traceback(self) -> None:
        """Generic exceptions should show a traceback."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        exc = ValueError("Something went wrong")
        with patch("apex_rag.cli.console", console):
            format_error(exc)
        output = buf.getvalue()
        assert "Unexpected Error" in output
        assert "ValueError" in output

    def test_format_success_prints_green_check(self) -> None:
        """Success message should include a checkmark."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        with patch("apex_rag.cli.console", console):
            format_success("All good!")
        output = buf.getvalue()
        assert "✔" in output
        assert "All good" in output

    def test_format_info_prints_dim_message(self) -> None:
        """Info message should be dimmed."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        with patch("apex_rag.cli.console", console):
            format_info("Loading…")
        output = buf.getvalue()
        assert "Loading" in output

    def test_format_warning_shows_title(self) -> None:
        """Warning should display the title."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        with patch("apex_rag.cli.console", console):
            format_warning("Disk space low", title="Storage Warning")
        output = buf.getvalue()
        assert "Disk space low" in output
        assert "Storage Warning" in output

    def test_format_table_renders_rows(self) -> None:
        """Table should render headers and rows."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        rows = [["a1", "b1"], ["a2", "b2"]]
        with patch("apex_rag.cli.console", console):
            format_table(rows, headers=["A", "B"])
        output = buf.getvalue()
        assert "a1" in output
        assert "b2" in output


# ── Progress / Spinners ───────────────────────────────────────────────────


class TestProgress:
    @pytest.mark.asyncio
    async def test_spinner_context(self) -> None:
        """Spinner context should accept updates."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        with patch("apex_rag.cli.console", console):
            with spinner_context(text="Testing…") as status:
                status.update("Still testing…")
        # Should not raise

    def test_progress_bar_context(self) -> None:
        """ProgressBar should advance correctly."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        with patch("apex_rag.cli.console", console):
            with ProgressBar(total=5, description="Test") as pb:
                pb.advance()
                pb.advance(2)
        # Should not raise

    @pytest.mark.asyncio
    async def test_display_streaming_assembles_chunks(self) -> None:
        """Streaming display should assemble chunks and return full text."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        with patch("apex_rag.cli.console", console):
            gen = _mock_stream(["Hel", "lo ", "World"])
            result = await display_streaming("Test", gen, speed=0.0)
        assert result == "Hello World"


# ── Doctor ────────────────────────────────────────────────────────────────


class TestDoctor:
    def test_doctor_check_prints_results(self) -> None:
        """Doctor should print check results without raising."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        with patch("apex_rag.cli.console", console):
            doctor_check()
        output = buf.getvalue()
        assert "ApexRAG Doctor" in output
        # Should report Python version
        assert "Python" in output

    def test_package_check_returns_bool(self) -> None:
        """_check_package should return True for installed packages."""
        assert _check_package("sys") is True
        assert _check_package("nonexistent_package_xyz") is False

    def test_print_system_info(self) -> None:
        """System info should print version and config details."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        with patch("apex_rag.cli.console", console):
            _print_system_info()
        output = buf.getvalue()
        from apex_rag import __version__
        assert __version__ in output
        assert "Version" in output


# ── REPL ──────────────────────────────────────────────────────────────────


class TestREPL:
    @pytest.mark.asyncio
    async def test_repl_with_valid_docs_starts_loop(self) -> None:
        """REPL should start and accept !list command."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1"])
        index.get_stats = AsyncMock(
            return_value={"total_nodes": 10, "leaf_count": 5, "max_depth": 3}
        )

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["!list", "!quit"]),
        ):
            await repl_loop(index)

        output = buf.getvalue()
        assert "doc1" in output
        assert "Bye" in output

    @pytest.mark.asyncio
    async def test_repl_handles_no_docs(self) -> None:
        """REPL should handle no documents gracefully."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=[])

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["", "!quit"]),
        ):
            await repl_loop(index)

        output = buf.getvalue()
        assert "No documents" in output

    @pytest.mark.asyncio
    async def test_repl_unknown_command(self) -> None:
        """REPL should handle unknown !commands."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1"])

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["!unknown", "!quit"]),
        ):
            await repl_loop(index)

        output = buf.getvalue()
        assert "Unknown command" in output

    @pytest.mark.asyncio
    async def test_repl_use_command_switches_doc(self) -> None:
        """REPL !use command should switch current document."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1", "doc2"])

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["!use doc2", "!quit"]),
        ):
            await repl_loop(index)

        output = buf.getvalue()
        assert "Switched to" in output

    @pytest.mark.asyncio
    async def test_repl_use_invalid_doc_shows_warning(self) -> None:
        """REPL !use with invalid doc should show warning."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1"])

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["!use missing_doc", "!quit"]),
        ):
            await repl_loop(index)

        output = buf.getvalue()
        assert "not found" in output

    @pytest.mark.asyncio
    async def test_repl_query_triggers_stream(self) -> None:
        """REPL query should call stream_query and display response."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1"])

        # stream_query is an async generator, so we need a real async gen as the mock
        async def stream_mock(question: str, doc_id: str, **kwargs: Any) -> AsyncGenerator[str, None]:
            _ = doc_id, kwargs
            for chunk in ["Answer ", "text"]:
                yield chunk

        index.stream_query = stream_mock

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["What is X?", "!quit"]),
        ):
            await repl_loop(index)

        output = buf.getvalue()
        assert "Answer" in output
        assert "text" in output

    @pytest.mark.asyncio
    async def test_repl_eof_exits_gracefully(self) -> None:
        """REPL should exit gracefully on EOFError."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1"])

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=EOFError),
        ):
            await repl_loop(index)

        output = buf.getvalue()
        assert "Bye" in output

    @pytest.mark.asyncio
    async def test_repl_stats_command(self) -> None:
        """REPL !stats should call get_stats."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1"])
        index.get_stats = AsyncMock(
            return_value={"total_nodes": 10, "leaf_count": 5, "max_depth": 3}
        )

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["!stats", "!quit"]),
        ):
            await repl_loop(index)

        index.get_stats.assert_awaited_once_with("doc1")

    @pytest.mark.asyncio
    async def test_repl_info_command_does_not_raise(self) -> None:
        """REPL !info should not raise any exception."""
        buf = StringIO()
        console = Console(file=buf, width=120, color_system=None)
        index = MagicMock()
        index.list_documents = AsyncMock(return_value=["doc1"])

        with (
            patch("apex_rag.cli.console", console),
            patch("apex_rag.cli.Prompt.ask", side_effect=["!info", "!quit"]),
        ):
            await repl_loop(index)

        # No exception raised


# ── __main__ Integration Tests ────────────────────────────────────────────


class TestMainEntryPoint:
    def test_version_flag(self) -> None:
        """--version should print version string."""
        buf = StringIO()
        console = Console(file=buf, width=120)
        from apex_rag import __version__

        with (
            patch("apex_rag.cli.console", console),
            patch("sys.argv", ["apex-rag", "--version"]),
        ):
            from apex_rag.__main__ import main
            main()
        output = buf.getvalue()
        assert __version__ in output

    def test_no_command_shows_help(self) -> None:
        """Running with no command should show help."""
        buf = StringIO()
        console = Console(file=buf, width=120)

        with (
            patch("apex_rag.cli.console", console),
            patch("sys.argv", ["apex-rag"]),
        ):
            from apex_rag.__main__ import main
            main()
        output = buf.getvalue()
        assert "Usage" in output
        assert "serve" in output
        assert "repl" in output
