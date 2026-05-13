"""
utils.py — Async helpers, logging infrastructure, and Reasoning Trace for ApexRAG.

The ReasoningTrace is the observability backbone: every agent decision is
color-coded and printed to stdout in real-time, giving users a transparent
window into the navigation process.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Theme & Console
# ---------------------------------------------------------------------------

_APEX_THEME = Theme(
    {
        "apex.enter": "bold cyan",
        "apex.explore": "bold yellow",
        "apex.found": "bold green",
        "apex.backtrack": "bold magenta",
        "apex.leaf": "bold bright_green",
        "apex.none": "dim red",
        "apex.error": "bold red",
        "apex.info": "dim white",
        "apex.timing": "italic dim cyan",
    }
)

console = Console(theme=_APEX_THEME, highlight=False)

# ---------------------------------------------------------------------------
# Standard Logger (Rich-enhanced)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)

logger = logging.getLogger("apex_rag")


def set_log_level(level: str) -> None:
    """Adjust ApexRAG log verbosity at runtime."""
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# Reasoning Trace
# ---------------------------------------------------------------------------


class ReasoningTrace:
    """
    Structured, color-coded console output for the Navigation Agent.

    Each method corresponds to a distinct agent action, making the navigation
    path immediately readable during development and production debugging.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._depth: int = 0
        self._start_time: float = time.monotonic()

    # -- Internal helpers ---------------------------------------------------

    def _indent(self) -> str:
        return "  " * self._depth

    def _elapsed(self) -> str:
        return f"{time.monotonic() - self._start_time:.2f}s"

    def _print(self, markup: str) -> None:
        if self.enabled:
            console.print(markup)

    # -- Public trace methods -----------------------------------------------

    def start(self, query: str, root_id: int) -> None:
        self._depth = 0
        self._start_time = time.monotonic()
        self._print(
            f"\n[apex.info]━━━ ApexRAG Navigation Start ━━━[/]\n"
            f"[apex.info]Query :[/] [bold white]{query}[/]\n"
            f"[apex.info]Root  :[/] node_id={root_id}\n"
        )

    def enter_node(self, node_id: int, summary: str, path: str) -> None:
        self._depth += 1
        prefix = self._indent()
        self._print(
            f"{prefix}[apex.enter]↳ ENTER[/] "
            f"node={node_id} path=[italic]{path}[/italic]\n"
            f"{prefix}  [apex.info]{summary[:120]}{'…' if len(summary) > 120 else ''}[/]"
        )

    def exploring_children(self, node_id: int, child_count: int) -> None:
        prefix = self._indent()
        self._print(
            f"{prefix}[apex.explore]⟳ EXPLORE[/] "
            f"node={node_id} → evaluating {child_count} child summaries"
        )

    def agent_choice(self, chosen_id: int | None, reason: str) -> None:
        prefix = self._indent()
        if chosen_id is None:
            self._print(
                f"{prefix}[apex.none]✗ AGENT → NONE[/]  reason: {reason[:80]}"
            )
        else:
            self._print(
                f"{prefix}[apex.found]✔ AGENT → node={chosen_id}[/]  reason: {reason[:80]}"
            )

    def leaf_reached(self, node_id: int, content_preview: str) -> None:
        prefix = self._indent()
        self._print(
            f"{prefix}[apex.leaf]★ LEAF REACHED[/] node={node_id}\n"
            f"{prefix}  preview: {content_preview[:160]}{'…' if len(content_preview) > 160 else ''}"
        )

    def backtrack(self, from_id: int, to_id: int | None, *, reason: str = "") -> None:
        self._depth = max(0, self._depth - 1)
        prefix = self._indent()
        target = f"node={to_id}" if to_id is not None else "root (exhausted)"
        reason_str = f"  [{reason}]" if reason else ""
        self._print(
            f"{prefix}[apex.backtrack]↑ BACKTRACK[/] from={from_id} → {target}{reason_str}"
        )

    def finish(self, found: bool) -> None:
        elapsed = self._elapsed()
        status = "[apex.found]SUCCESS[/]" if found else "[apex.error]NOT FOUND[/]"
        self._print(
            f"\n[apex.info]━━━ Navigation Complete ━━━[/]  "
            f"result={status}  [apex.timing]elapsed={elapsed}[/]\n"
        )

    def error(self, message: str) -> None:
        self._print(f"[apex.error]✖ ERROR[/] {message}")


# ---------------------------------------------------------------------------
# Async Retry Decorator (for Ollama calls)
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


def async_retry(
    *,
    max_attempts: int = 3,
    backoff_base: float = 1.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[_F], _F]:
    """
    Decorator that retries an async function with exponential back-off.

    Args:
        max_attempts: Maximum number of total attempts.
        backoff_base: Multiplier for sleep between attempts (seconds).
        exceptions: Exception types that trigger a retry.
    """

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    wait = backoff_base ** (attempt - 1)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs…",
                        attempt,
                        max_attempts,
                        fn.__name__,
                        exc,
                        wait,
                    )
                    if attempt < max_attempts:
                        await asyncio.sleep(wait)
            raise RuntimeError(
                f"{fn.__name__} failed after {max_attempts} attempts"
            ) from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Miscellaneous Helpers
# ---------------------------------------------------------------------------


def truncate(text: str, max_len: int = 200) -> str:
    """Return a safely truncated string for log display."""
    return text[:max_len] + "…" if len(text) > max_len else text


def build_ltree_path(parent_path: str | None, position: int) -> str:
    """
    Construct an LTree-style path string.

    Examples:
        build_ltree_path(None, 1)      → "1"
        build_ltree_path("1", 2)       → "1.2"
        build_ltree_path("1.2", 3)     → "1.2.3"
    """
    if parent_path is None:
        return str(position)
    return f"{parent_path}.{position}"


def path_depth(path: str) -> int:
    """Return the nesting depth of an LTree path (0 = root)."""
    return len(path.split(".")) - 1
