"""
__main__.py — CLI entry point for ApexRAG.

Usage:

    python -m apex_rag serve              # Start API server
    python -m apex_rag ingest <file>       # Ingest a document
    python -m apex_rag query <id> <q>      # Query a document
    python -m apex_rag stream <id> <q>     # Stream an answer
    python -m apex_rag repl                # Interactive REPL
    python -m apex_rag doctor              # Validate environment
    python -m apex_rag list                # List documents
    python -m apex_rag info                # Show system info
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from apex_rag import __version__
from apex_rag.cli import (
    console,
    display_streaming,
    doctor_check,
    format_error,
    format_info,
    format_success,
    format_table,
    print_banner,
    print_help,
    repl_loop,
    spinner_context,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m apex_rag",
        description="ApexRAG — Structural AI Retrieval Infrastructure",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show agent reasoning traces")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command")

    # ── serve ────────────────────────────────────────────────────────
    serve = sub.add_parser("serve", help="Start the ApexRAG API server")
    serve.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")

    # ── ingest ────────────────────────────────────────────────────────
    ingest = sub.add_parser("ingest", help="Ingest a document file")
    ingest.add_argument("file", type=str, help="Path to the document file")
    ingest.add_argument("--doc-id", type=str, default=None, help="Override doc ID")
    ingest.add_argument("--no-summaries", action="store_true", help="Skip LLM summaries")

    # ── query ─────────────────────────────────────────────────────────
    query = sub.add_parser("query", help="Query an indexed document")
    query.add_argument("doc_id", type=str, help="Document ID")
    query.add_argument("question", type=str, help="Natural-language question")

    # ── stream ───────────────────────────────────────────────────────
    stream_p = sub.add_parser("stream", help="Stream answer tokens as they arrive")
    stream_p.add_argument("doc_id", type=str, help="Document ID")
    stream_p.add_argument("question", type=str, help="Natural-language question")

    # ── repl ─────────────────────────────────────────────────────────
    sub.add_parser("repl", help="Interactive query REPL")

    # ── doctor ───────────────────────────────────────────────────────
    sub.add_parser("doctor", help="Validate environment & configuration")

    # ── global-query ────────────────────────────────────────────────
    gquery = sub.add_parser("global-query", help="Query across all documents")
    gquery.add_argument("question", type=str, help="Natural-language question")

    # ── list ─────────────────────────────────────────────────────────
    sub.add_parser("list", help="List all indexed documents")

    # ── info ─────────────────────────────────────────────────────────
    sub.add_parser("info", help="Show system information & settings")

    return parser


# ── Command Handlers ─────────────────────────────────────────────────────


async def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        format_error(
            ImportError("'serve' requires extra dependencies.\nInstall:  pip install apex-rag[web]")
        )
        sys.exit(1)

    from apex_rag.api import app
    from apex_rag.config import settings

    print_banner(subtitle=f"API Server — http://{args.host}:{args.port}")
    format_info(f"DB:        {settings.db_url.split('?')[0]}")
    format_info(f"Model:     {settings.model}")
    format_info(f"Log level: {settings.log_level}")
    format_info(f"Docs:      http://{args.host}:{args.port}/docs")
    console.print()

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a document file with rich progress output."""
    from apex_rag import ApexIndex

    file_path = Path(args.file)
    if not file_path.exists():
        from apex_rag.exceptions import FileValidationError

        format_error(
            FileValidationError(
                message=f"File not found: {file_path}",
                hint="Check the file path and make sure the file exists.",
            )
        )
        sys.exit(1)

    print_banner(subtitle=f"Ingesting — {file_path.name}")

    with spinner_context(text=f"Parsing and indexing [bold]{file_path.name}[/]…") as status:
        async with await ApexIndex.create(
            trace_enabled=args.verbose,
        ) as index:
            status.update("Generating summaries…")
            doc_id = await index.ingest(
                file_path,
                doc_id=args.doc_id,
                synthesize_summaries=not args.no_summaries,
            )
            status.update("Computing stats…")
            stats = await index.get_stats(doc_id)

    format_success(f"Ingested [bold]{file_path.name}[/]")
    format_info(f"  doc_id:   {doc_id}")
    format_info(f"  Nodes:    {stats['total_nodes']}")
    format_info(f"  Leaves:   {stats['leaf_count']}")
    format_info(f"  Max depth: {stats['max_depth']}")
    console.print()


async def _cmd_query(args: argparse.Namespace) -> None:
    """Query an indexed document."""
    from apex_rag import ApexIndex

    print_banner(subtitle="Query")

    async with await ApexIndex.create(trace_enabled=args.verbose) as index:
        with spinner_context(text="Searching document tree…"):
            result = await index.query(args.question, args.doc_id)

        if result and result.answer_text:
            confidence = result.coverage_guarantee or 1.0
            evidence_count = len(result.evidence_packets)
            format_success("Answer generated")
            format_info(
                f"  evidence={evidence_count}, confidence={confidence:.2f}, "
                f"latency_ms={result.latency_ms:.1f}"
            )
            console.print()
            console.print(result.answer_text)
        else:
            console.print("[bold yellow]No answer found.[/]")
    console.print()


async def _cmd_stream(args: argparse.Namespace) -> None:
    """Stream query tokens as they arrive."""
    from apex_rag import ApexIndex

    print_banner(subtitle="Streaming Query")

    async with await ApexIndex.create() as index:
        generator = index.stream_query(args.question, args.doc_id)
        result = await display_streaming(
            title=f"Q: {args.question}",
            generator=generator,
        )

        if not result or result.startswith("I could not find"):
            console.print(f"\n[bold yellow]{result}[/]")
    console.print()


async def _cmd_repl(args: argparse.Namespace | None = None) -> None:
    """Interactive REPL mode."""
    from apex_rag import ApexIndex

    print_banner(subtitle="Interactive REPL")

    async with await ApexIndex.create(trace_enabled=args.verbose if args else False) as index:
        await repl_loop(index)


async def _cmd_doctor(_args: argparse.Namespace | None = None) -> None:
    """Validate environment and configuration."""
    doctor_check()


async def _cmd_global_query(args: argparse.Namespace) -> None:
    """Query across all documents."""
    from apex_rag import ApexIndex

    print_banner(subtitle="Global Query")

    async with await ApexIndex.create(trace_enabled=args.verbose) as index:
        with spinner_context(text="Searching all documents…"):
            result = await index.query_global(args.question)
        if result:
            content = result.answer_text if hasattr(result, "answer_text") else str(result)
            format_success("Answer found:")
            console.print()
            console.print(content)
        else:
            console.print("[bold yellow]No answer found across documents.[/]")
    console.print()


async def _cmd_list(_args: argparse.Namespace | None = None) -> None:
    """List all documents."""
    from apex_rag import ApexIndex

    async with await ApexIndex.create() as index:
        docs = await index.list_documents()
        if not docs:
            format_info("No documents indexed yet.  Use [bold]ingest[/] to add one.")
            return

        format_success(f"{len(docs)} document(s):")
        console.print()

        rows: list[list[str]] = []
        for doc_id in docs:
            try:
                stats = await index.get_stats(doc_id)
                rows.append(
                    [
                        doc_id,
                        str(stats["total_nodes"]),
                        str(stats["leaf_count"]),
                        str(stats["max_depth"]),
                    ]
                )
            except Exception:
                rows.append([doc_id, "—", "—", "—"])

        format_table(rows, headers=["Document ID", "Nodes", "Leaves", "Depth"])


async def _cmd_info(_args: argparse.Namespace | None = None) -> None:
    """Show system information."""
    from apex_rag.cli import _print_system_info

    _print_system_info()


# ── Main Dispatch ─────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Version or help ──────────────────────────────────────────────
    if args.version:
        console.print(f"ApexRAG v{__version__}")
        return

    if args.command is None:
        print_banner()
        print_help()
        return

    # ── Dispatch ─────────────────────────────────────────────────────
    try:
        if args.command == "serve":
            asyncio.run(_cmd_serve(args))
        elif args.command == "ingest":
            asyncio.run(_cmd_ingest(args))
        elif args.command == "query":
            asyncio.run(_cmd_query(args))
        elif args.command == "stream":
            asyncio.run(_cmd_stream(args))
        elif args.command == "repl":
            asyncio.run(_cmd_repl(args))
        elif args.command == "doctor":
            asyncio.run(_cmd_doctor(args))
        elif args.command == "global-query":
            asyncio.run(_cmd_global_query(args))
        elif args.command == "list":
            asyncio.run(_cmd_list(args))
        elif args.command == "info":
            asyncio.run(_cmd_info(args))
        else:
            console.print(f"[bold red]Unknown command:[/] {args.command}")
            console.print()
            print_help()
    except Exception as exc:
        format_error(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
