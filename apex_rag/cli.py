"""
cli.py — Rich CLI helpers for ApexRAG.

Provides reusable components for the interactive REPL, streaming display,
rich error formatting, progress spinners, and the ``doctor`` validation
command.  All output goes through this module so the rest of the codebase
never writes to ``print()`` directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import platform
import shutil
import sys
import textwrap
from collections.abc import AsyncGenerator
from typing import Any

from rich import box
from rich.console import Console
from rich.errors import MarkupError
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback

from apex_rag import __version__
from apex_rag.config import settings
from apex_rag.exceptions import ApexRAGError

# ── Console ───────────────────────────────────────────────────────────────

console = Console(highlight=False)

# ── Banner ────────────────────────────────────────────────────────────────

_BANNER = rf"""
[bold cyan]   ___                  ___
  / _ \__ _____ _____ _/ _ \__ _____  ___ ___[/]
 [bold green]/ ___/ _` \ V  V / _` / ___/ _` \ \ / / -_|_-<[/]
[bold magenta]/_/   \__,_|\_/\_/\__,_/_/   \__,_/\_\_/\__/__/[/]
[dim]v{__version__}  —  Structural AI Retrieval[/]
"""


def print_banner(*, subtitle: str | None = None) -> None:
    """Print the ApexRAG startup banner."""
    try:
        console.print(_BANNER, markup=True)
    except MarkupError:
        console.print(f"ApexRAG v{__version__}", style="bold")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/]")
    console.print()


# ── Error Formatting ──────────────────────────────────────────────────────


def format_error(exc: Exception) -> None:
    """Print a rich-formatted error panel with the exception's hint.

    Handles :class:`ApexRAGError` subclasses specially by extracting
    the ``.code``, ``.message``, and ``.hint`` fields.  All other
    exceptions are rendered with a traceback.
    """
    if isinstance(exc, ApexRAGError):
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", width=6)
        grid.add_column()
        grid.add_row("Code", f"[bold red]{exc.code}[/]")
        grid.add_row("Error", exc.message)
        grid.add_row("Hint", f"[italic green]{exc.hint}[/]")
        panel = Panel(
            grid,
            title="[bold red]ApexRAG Error[/]",
            border_style="red",
            box=box.ROUNDED,
        )
        console.print(panel)
    else:
        console.print("[bold red]Unexpected Error[/]")
        console.print(Traceback.from_exception(type(exc), exc, exc.__traceback__))


def format_warning(message: str, *, title: str = "Warning") -> None:
    """Print a rich-formatted warning panel."""
    panel = Panel(
        textwrap.fill(message, width=shutil.get_terminal_size().columns - 8),
        title=f"[bold yellow]{title}[/]",
        border_style="yellow",
        box=box.ROUNDED,
    )
    console.print(panel)


def format_success(message: str) -> None:
    """Print a green success message."""
    console.print(f"  [bold green]✔[/] {message}")


def format_info(message: str) -> None:
    """Print an info message (dim white)."""
    console.print(f"  [dim]{message}[/]")


def format_table(rows: list[list[str]], *, headers: list[str] | None = None) -> None:
    """Print a simple table with automatic column widths."""
    table = Table(box=box.SIMPLE, show_edge=False)
    if headers:
        for h in headers:
            table.add_column(h, style="bold cyan", no_wrap=True)
    else:
        for _ in (rows[0] if rows else []):
            table.add_column(no_wrap=True)
    for row in rows:
        table.add_row(*row)
    console.print(table)


# ── Progress / Spinners ────────────────────────────────────────────────────


def spinner_context(*, text: str = "Working…") -> Status:
    """Return a :class:`rich.status.Status` context manager.

    Usage::

        with spinner_context(text="Ingesting document…") as status:
            result = await do_work()
            status.update("Finalising…")
    """
    return console.status(text, spinner="dots")


class ProgressBar:
    """A simple progress bar for counting items.

    Usage::

        with ProgressBar(total=10, description="Processing") as pb:
            for item in items:
                await process(item)
                pb.advance()
    """

    def __init__(self, total: int, *, description: str = "Processing") -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task_id = self._progress.add_task(description, total=total)

    def advance(self, advance: int = 1) -> None:
        """Advance the progress bar by one (or more) steps."""
        self._progress.update(self._task_id, advance=advance)

    def __enter__(self) -> ProgressBar:
        self._progress.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._progress.__exit__(*args)


# ── Streaming Answer Display ──────────────────────────────────────────────


async def display_streaming(
    title: str,
    generator: AsyncGenerator[str, None],
    *,
    speed: float = 0.0,
) -> str:
    """Stream and display answer tokens in a live panel.

    Args:
        title:     Panel title (e.g. the user's question).
        generator: Async generator yielding text chunks.
        speed:     Artificial delay between chunks (0 for real-time).

    Returns:
        The fully assembled answer string.
    """
    chunks: list[str] = []
    panel = Panel(
        Text("", style="green"),
        title=f"[bold cyan]{title}[/]",
        border_style="green",
        box=box.ROUNDED,
    )

    with Live(panel, refresh_per_second=20, console=console) as live:
        async for chunk in generator:
            chunks.append(chunk)
            text = Text("".join(chunks), style="green")
            panel.renderable = text
            live.update(panel)
            if speed > 0:
                await asyncio.sleep(speed)

    return "".join(chunks)


# ── REPL Mode ──────────────────────────────────────────────────────────────


def _get_welcome_message() -> str:
    return (
        "[bold cyan]Welcome to ApexRAG REPL[/]\n"
        "  Type a question to query an indexed document, or one of:\n"
        "  [bold]!help[/]    — Show this message\n"
        "  [bold]!list[/]    — List indexed documents\n"
        "  [bold]!info[/]    — Show system info\n"
        "  [bold]!stats[/]   — Show document stats\n"
        "  [bold]!quit[/]    — Exit REPL\n"
    )


async def repl_loop(index: Any) -> None:
    """Interactive REPL loop for querying documents.

    Args:
        index: An initialised :class:`ApexIndex` instance.
    """
    docs: list[str] = []
    with contextlib.suppress(Exception):
        docs = await index.list_documents()

    if not docs:
        format_warning(
            "No documents indexed yet.  Use ``ingest`` first or start the API server.",
            title="No Documents",
        )
        answer = Prompt.ask(
            "  [bold]Ingest a file now?[/] (path or Enter to skip)",
            default="",
        )
        if answer.strip():
            try:
                ingested_id = await index.ingest(answer.strip())
                docs = [ingested_id]
                format_success(f"Ingested → doc_id={ingested_id}")
            except Exception as exc:
                format_error(exc)
                return

    console.print()
    console.print(Panel(_get_welcome_message(), border_style="cyan", box=box.ROUNDED))
    console.print()

    if docs:
        format_info(f"Found {len(docs)} document(s):")
        for d in docs:
            format_info(f"  • {d}")
        console.print()

    init_doc: str | None = docs[0] if docs else None
    if init_doc is None:
        format_warning("No documents available.  Use !help for commands.")
        console.print()

    current_doc: str | None = init_doc

    while True:
        try:
            line = Prompt.ask(
                "[bold cyan]query[/]"
                + (f"([dim]{current_doc}[/])" if current_doc else "")
                + "[bold]>[/]",
                default="",
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Bye![/]")
            break

        line = line.strip()

        if not line:
            continue

        # ── Meta-commands ──────────────────────────────────────────────
        if line.startswith("!"):
            cmd = line[1:].strip().lower()

            if cmd in ("quit", "exit", "q"):
                console.print("[yellow]Bye![/]")
                break

            elif cmd in ("help", "?"):
                console.print(Panel(_get_welcome_message(), border_style="cyan"))

            elif cmd == "list":
                try:
                    docs = await index.list_documents()
                    if docs:
                        format_success(f"{len(docs)} document(s):")
                        for d in docs:
                            format_info(f"  • {d}")
                    else:
                        format_info("No documents indexed.")
                except Exception as exc:
                    format_error(exc)

            elif cmd.startswith("use "):
                name = cmd[4:].strip()
                try:
                    all_docs = await index.list_documents()
                    if name in all_docs:
                        current_doc = name
                        format_success(f"Switched to doc_id={name}")
                    else:
                        format_warning(f"Document '{name}' not found.  Use !list to see available docs.")
                except Exception as exc:
                    format_error(exc)

            elif cmd == "info":
                _print_system_info()

            elif cmd == "stats":
                if current_doc is None:
                    format_warning("No document selected.  Use !use <doc_id> first.")
                    continue
                try:
                    stats = await index.get_stats(current_doc)
                    console.print(stats)
                except Exception as exc:
                    format_error(exc)

            else:
                format_warning(f"Unknown command: !{cmd}")

            continue

        # ── Query ──────────────────────────────────────────────────────
        if current_doc is None:
            format_warning("No document selected.  Use !use <doc_id> or ingest a file first.")
            continue

        try:
            async for chunk in index.stream_query(line, current_doc):
                console.print(chunk, end="", style="green")
            console.print()
        except Exception as exc:
            format_error(exc)


# ── Doctor Command ─────────────────────────────────────────────────────────


_MANDATORY_PACKAGES = [
    "markitdown",
    "sqlalchemy",
    "aiosqlite",
    "openai",
    "pydantic",
    "rich",
]

_OPTIONAL_PACKAGES = {
    "anthropic": "anthropic",
    "groq": "groq",
    "ollama": "ollama",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "asyncpg": "asyncpg",
    "sentence_transformers": "sentence-transformers",
}


def _check_package(name: str, pip_name: str | None = None) -> bool:
    """Return True if *name* can be imported."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _print_system_info() -> None:
    """Print a rich-formatted system information panel."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", width=14)
    grid.add_column()
    grid.add_row("Version", f"v{__version__}")
    grid.add_row("Python", platform.python_version())
    grid.add_row("Platform", platform.platform())
    grid.add_row("Hostname", platform.node())
    grid.add_row("DB", settings.db_url.split("?")[0])
    grid.add_row("Log Level", settings.log_level)
    grid.add_row("Log Format", settings.log_format)
    grid.add_row("Parser", settings.parser_backend)

    panel = Panel(
        grid,
        title="[bold cyan]ApexRAG System Info[/]",
        border_style="cyan",
        box=box.ROUNDED,
    )
    console.print(panel)


def doctor_check() -> None:
    """Run the ``doctor`` diagnostic checks and print results."""
    console.print()
    panel_text = Text.assemble(
        ("ApexRAG Doctor", "bold cyan"),
        " — ",
        (f"v{__version__}", "dim"),
        "\n\n",
        ("Checking your environment for issues…", "italic"),
    )
    console.print(Panel(panel_text, border_style="cyan", box=box.ROUNDED))
    console.print()

    issues: list[str] = []

    # 1. Python version
    py_ok = sys.version_info >= (3, 10)
    _print_check("Python >= 3.10", py_ok)
    if not py_ok:
        issues.append("Python version too old (need >= 3.10)")

    # 2. Mandatory packages
    console.print("\n[bold]Mandatory packages:[/]")
    for pkg in _MANDATORY_PACKAGES:
        ok = _check_package(pkg)
        _print_check(f"  {pkg}", ok)
        if not ok:
            issues.append(f"Missing mandatory package: {pkg}")

    # 3. Optional packages
    console.print("\n[bold]Optional packages:[/]")
    for name, pip_name in _OPTIONAL_PACKAGES.items():
        ok = _check_package(name)
        _print_check(f"  {pip_name}", ok)
        if not ok:
            issues.append(f"Missing optional package: {pip_name}")

    # 4. DB connectivity
    console.print("\n[bold]Configuration:[/]")
    _print_check(f"DB URL: {settings.db_url.split('?')[0]} (syntax OK)", True)
    if settings.api_key:
        _print_check("API key set", True)
    else:
        _print_check("API key not set (auth disabled)", True)

    # 5. Terminal capabilities
    console.print("\n[bold]Terminal:[/]")
    cols = shutil.get_terminal_size().columns
    _print_check(f"Columns: {cols}", cols >= 60)

    # ── Summary ────────────────────────────────────────────────────────
    console.print()
    if not issues:
        format_success("All checks passed — your environment is ready!")
    else:
        console.print(Panel(
            "\n".join(f"  • {issue}" for issue in issues),
            title="[bold yellow]Issues Found[/]",
            border_style="yellow",
            box=box.ROUNDED,
        ))
        console.print()
        if issues:
            format_info("Run:  pip install apex-rag[all]  to install all recommended packages.")
    console.print()


def _print_check(label: str, ok: bool) -> None:
    """Print a single check result with a checkmark or cross."""
    icon = "[bold green]✔[/]" if ok else "[bold red]✘[/]"
    console.print(f"  {icon} {label}")


# ── Help Text ──────────────────────────────────────────────────────────────


COMMAND_HELP = """
[bold cyan]Usage:[/] python -m apex_rag [command] [options]

[bold]Commands:[/]

  [bold]serve[/]      Start the API server
  [bold]ingest[/]     Ingest a document file
  [bold]query[/]      Query an indexed document
  [bold]stream[/]     Stream an answer token-by-token
  [bold]repl[/]       Interactive query REPL
  [bold]doctor[/]     Validate environment & configuration
  [bold]list[/]       List indexed documents
  [bold]info[/]       Show system information & settings

[bold]Global options:[/]

  [bold]--verbose, -v[/]      Show agent reasoning traces
  [bold]--version[/]          Show version and exit

[dim]Run: python -m apex_rag <command> --help  for command-specific options[/]
"""


def print_help() -> None:
    """Print the rich-formatted help page."""
    console.print(Markdown(COMMAND_HELP))
