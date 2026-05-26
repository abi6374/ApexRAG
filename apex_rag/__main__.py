"""
__main__.py — CLI entry point for ApexRAG.

Usage:
    python -m apex_rag serve          # Start the API server
    python -m apex_rag ingest <file>   # Ingest a document
    python -m apex_rag query <doc_id> <question>  # Query a document
    python -m apex_rag list            # List documents
    python -m apex_rag info            # Show system info
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m apex_rag",
        description="ApexRAG — Local-first Agentic RAG Library",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve = sub.add_parser("serve", help="Start the ApexRAG API server")
    serve.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")

    # ingest
    ingest = sub.add_parser("ingest", help="Ingest a document file")
    ingest.add_argument("file", type=str, help="Path to the document file")
    ingest.add_argument("--doc-id", type=str, default=None, help="Override doc ID")
    ingest.add_argument("--no-summaries", action="store_true", help="Skip LLM summaries")

    # query
    query = sub.add_parser("query", help="Query an indexed document")
    query.add_argument("doc_id", type=str, help="Document ID")
    query.add_argument("question", type=str, help="Natural-language question")

    # global-query
    gquery = sub.add_parser("global-query", help="Query across all documents")
    gquery.add_argument("question", type=str, help="Natural-language question")

    # list
    sub.add_parser("list", help="List all indexed documents")

    # info
    sub.add_parser("info", help="Show system information")

    return parser


async def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: 'serve' requires extra dependencies.\nInstall:  pip install apex-rag[web]")
        sys.exit(1)

    from apex_rag.api import app
    from apex_rag.config import settings

    print(f"⚡ ApexRAG API server starting on http://{args.host}:{args.port}")
    print(f"   DB: {settings.db_url.split('?')[0]}")
    print(f"   Ollama: {settings.ollama_host}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print()

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
    """Ingest a document file."""
    from apex_rag import ApexIndex

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"📄 Ingesting: {file_path.name}...")
    async with await ApexIndex.create() as index:
        doc_id = await index.ingest(
            file_path,
            doc_id=args.doc_id,
            synthesize_summaries=not args.no_summaries,
        )
        stats = await index.get_stats(doc_id)
        print(f"✅ Done! doc_id={doc_id}")
        print(
            f"   Nodes: {stats['total_nodes']}, Leaves: {stats['leaf_count']}, Depth: {stats['max_depth']}"
        )


async def _cmd_query(args: argparse.Namespace) -> None:
    """Query an indexed document."""
    from apex_rag import ApexIndex

    async with await ApexIndex.create() as index:
        result = await index.query(args.question, args.doc_id)
        if result:
            print(f"✅ Found in section: {result.title} (path={result.path})")
            print(f"   Verified: {result.verified}, Confidence: {result.confidence:.2f}")
            print(f"\n{'─' * 60}")
            print(result.content)
        else:
            print("❌ No answer found.")


async def _cmd_global_query(args: argparse.Namespace) -> None:
    """Query across all documents."""
    from apex_rag import ApexIndex

    async with await ApexIndex.create() as index:
        result = await index.query_global(args.question, synthesize=True)
        if result:
            content = result.content if hasattr(result, "content") else str(result)
            print(f"✅ Answer:\n{'─' * 60}")
            print(content)
        else:
            print("❌ No answer found across documents.")


async def _cmd_list() -> None:
    """List all documents."""
    from apex_rag import ApexIndex

    async with await ApexIndex.create() as index:
        docs = await index.list_documents()
        if not docs:
            print("No documents indexed yet.")
            return
        print(f"📚 {len(docs)} document(s):")
        print()
        for i, doc_id in enumerate(docs, 1):
            try:
                stats = await index.get_stats(doc_id)
                print(f"  {i}. {doc_id}")
                print(
                    f"     Nodes: {stats['total_nodes']}, Leaves: {stats['leaf_count']}, Depth: {stats['max_depth']}"
                )
            except Exception:
                print(f"  {i}. {doc_id}")


async def _cmd_info() -> None:
    """Show system information."""
    from apex_rag import __version__
    from apex_rag.config import settings

    print(f"ApexRAG v{__version__}")
    print(f"Python: {sys.version}")
    print()
    print("Settings:")
    print(f"  DB:          {settings.db_url.split('?')[0]}")
    print(f"  Ollama:      {settings.ollama_host}")
    print(f"  Model:       {settings.model}")
    print(f"  Verify:      {settings.verify_leaves}")
    print(f"  Parser:      {settings.parser_backend}")
    print(f"  Log level:   {settings.log_level}")
    print(f"  Log format:  {settings.log_format}")
    print(f"  Max upload:  {settings.max_upload_size_mb} MB")
    print(f"  CORS:        {settings.cors_origins}")
    print(f"  Auth:        {'enabled' if settings.api_key else 'disabled'}")
    print()
    print("Optional Extras:")
    for extra in ["web", "postgres", "docling", "migrations"]:
        try:
            __import__(extra)
            print(f"  ✅ {extra}: installed")
        except ImportError:
            print(f"  ❌ {extra}: not installed")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        asyncio.run(_cmd_serve(args))
    elif args.command == "ingest":
        asyncio.run(_cmd_ingest(args))
    elif args.command == "query":
        asyncio.run(_cmd_query(args))
    elif args.command == "global-query":
        asyncio.run(_cmd_global_query(args))
    elif args.command == "list":
        asyncio.run(_cmd_list())
    elif args.command == "info":
        asyncio.run(_cmd_info())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
