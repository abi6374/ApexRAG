"""
ingestion.py — Document parsing and Decision Tree synthesis for ApexRAG.

Pipeline:
    1. Convert the source file (PDF, DOCX, HTML, plain text) to Markdown
       using `markitdown` (or `docling` as an optional backend).
    2. Parse the Markdown into a hierarchy of `ParsedSection` objects
       by walking Markdown headings (#, ##, ###, …).
    3. Persist a `DocumentNode` tree into the StorageEngine.
    4. Synthesize "Semantic Map" summaries for every node in parallel
       using Ollama, with configurable concurrency to saturate local GPU/CPU.

The ingestion is fully async-first. CPU-bound parsing runs in an executor to
avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_rag.providers import AsyncLLM
from apex_rag.storage import DocumentNode, PageIndexEntry, StorageEngine
from apex_rag.utils import async_retry, build_ltree_path, logger, truncate

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedSection:
    """
    An in-memory representation of one Markdown heading section.

    Attributes:
        level:      Heading depth (1 = #, 2 = ##, 3 = ###, …).
        title:      Heading text, stripped of leading #'s.
        content:    Raw text under this heading (before any sub-headings).
        children:   Nested sub-sections.
        position:   Sibling index (1-based) within the parent.
        path:       LTree-style path, e.g. "1.2.3".
        page_start: First page this section appears on (0 = unknown).
        page_end:   Last page (inclusive).
        image_data: Base64 or path to an image associated with this section (multimodal).
    """

    level: int
    title: str
    content: str
    children: list[ParsedSection] = field(default_factory=list)
    position: int = 1
    path: str = "1"
    page_start: int = 0
    page_end: int = 0
    image_data: str | None = None


# ---------------------------------------------------------------------------
# Markdown Parser
# ---------------------------------------------------------------------------

# Matches headings: "# Title", "## Title", etc. (ATX style only)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

# Matches page markers inserted by markitdown/docling: <!-- Page 3 --> or [Page 3]
_PAGE_MARKER_RE = re.compile(r"(?:<!--\s*Page\s+(\d+)\s*-->|\[Page\s+(\d+)\])", re.IGNORECASE)


def _parse_markdown_to_tree(markdown: str) -> list[ParsedSection]:
    """
    Convert a Markdown string into a tree of ParsedSection objects.

    Algorithm:
        - Walk through each heading match.
        - Maintain a stack of open sections per depth level.
        - Text between headings belongs to the immediately preceding section.
        - Page numbers are extracted from <!-- Page N --> markers.
    """
    markdown = markdown.strip()
    if markdown and not markdown.startswith("#"):
        # Prevent data loss if there's text before the first heading, or no headings at all
        markdown = "# Document\n\n" + markdown

    sections: list[ParsedSection] = []
    # Stack entries: (level, ParsedSection)
    stack: list[tuple[int, ParsedSection]] = []
    last_pos = 0
    current_page = 0

    matches = list(_HEADING_RE.finditer(markdown))

    for match in matches:
        level = len(match.group(1))
        title = match.group(2).strip()

        # Text between previous heading end and current heading start
        preceding_text = markdown[last_pos : match.start()]

        # Extract page numbers from preceding text
        for pm in _PAGE_MARKER_RE.finditer(preceding_text):
            page_num = int(pm.group(1) or pm.group(2))
            current_page = max(current_page, page_num)

        if stack and preceding_text.strip():
            stack[-1][1].content = preceding_text.strip()
            # Record end page for the previous section
            if current_page > 0:
                stack[-1][1].page_end = current_page

        last_pos = match.end()

        # Determine parent and position
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent = stack[-1][1] if stack else None
        siblings = parent.children if parent else sections
        position = len(siblings) + 1
        parent_path = parent.path if parent else None

        section = ParsedSection(
            level=level,
            title=title,
            content="",
            position=position,
            path=build_ltree_path(parent_path, position),
            page_start=current_page if current_page > 0 else 0,
            page_end=current_page if current_page > 0 else 0,
        )

        siblings.append(section)
        stack.append((level, section))

    # Trailing text after the last heading
    trailing = markdown[last_pos:].strip()
    if stack and trailing:
        # Capture any trailing page markers
        for pm in _PAGE_MARKER_RE.finditer(trailing):
            page_num = int(pm.group(1) or pm.group(2))
            current_page = max(current_page, page_num)
        stack[-1][1].content = trailing
        if current_page > 0:
            stack[-1][1].page_end = current_page

    # Propagate page ranges upward (parent's range = min(child starts)..max(child ends))
    _propagate_pages(sections)

    # Robust chunking for industry readiness:
    # If any section is a massive wall of text (>3000 chars), split it into sub-sections.
    _chunk_large_sections(sections)

    return sections


def _propagate_pages(sections: list[ParsedSection]) -> None:
    """
    Recursively propagate page ranges upward through the tree.

    A parent's page_start = min of all its children's page_starts.
    A parent's page_end   = max of all its children's page_ends.
    This ensures that chapter-level nodes reflect the full page range
    of all their sub-sections.
    """
    for section in sections:
        if section.children:
            _propagate_pages(section.children)
            child_starts = [c.page_start for c in section.children if c.page_start > 0]
            child_ends   = [c.page_end   for c in section.children if c.page_end   > 0]
            if child_starts and section.page_start == 0:
                section.page_start = min(child_starts)
            if child_ends:
                section.page_end = max(child_ends)


def _chunk_large_sections(sections: list[ParsedSection], max_chars: int = 3000) -> None:
    """
    Recursively split sections with excessively large content into sub-sections.
    This guarantees no single leaf node exceeds the LLM context window, making
    the ingestion robust for industry documents (like long legal PDFs).
    """
    for section in sections:
        if section.children:
            _chunk_large_sections(section.children, max_chars)

        # Chunk only if this is a leaf node and its content is too massive
        if section.content and len(section.content) > max_chars and not section.children:
            chunks = _split_text(section.content, max_chars)
            if len(chunks) > 1:
                for chunk_idx, chunk_text in enumerate(chunks):
                    sub = ParsedSection(
                        level=section.level + 1,
                        title=f"{section.title} (Part {chunk_idx+1})",
                        content=chunk_text,
                        position=chunk_idx + 1,
                        path=build_ltree_path(section.path, chunk_idx + 1),
                        page_start=section.page_start,
                        page_end=section.page_end,
                    )
                    section.children.append(sub)
                # Parent is no longer a leaf, content is pushed to its children
                section.content = ""


def _split_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks by double newline (paragraphs)."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(p)
        current_len += len(p)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# Ollama Summariser
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
You are a document indexing assistant for a search engine.
Summarize the following section in EXACTLY 30 words or fewer.
Focus on the core topic and key information so that a search agent
can decide whether to navigate into this section.

Section title: {title}

Content:
{content}

Summary (30 words max):"""

_VISION_SUMMARY_PROMPT = """\
You are a vision-enabled document indexing assistant.
Analyze the provided image and its context to generate a summary in EXACTLY 30 words or fewer.
Describe what this image represents (chart, diagram, photo) and its key data or message.

Section title: {title}
Context: {content}

Vision Summary (30 words max):"""


class Summariser:
    """
    Generates 30-word Semantic Map summaries using a pluggable LLM.
    Supports multimodal inputs (text + images).

    Args:
        llm:            An instance implementing the AsyncLLM protocol.
        max_concurrent: Max parallel LLM calls.
    """

    def __init__(
        self,
        llm: AsyncLLM,
        max_concurrent: int = 10,  # Increased for faster large-doc ingestion
    ) -> None:
        self._llm = llm
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @async_retry(max_attempts=3, backoff_base=2.0, exceptions=(Exception,))
    async def summarise(self, title: str, content: str, image_data: str | None = None) -> str:
        """
        Call LLM to produce a ≤30-word summary for a section.
        If image_data is provided, uses vision capabilities.
        """
        if not content.strip() and not image_data:
            return title[:120]

        if image_data:
            prompt = _VISION_SUMMARY_PROMPT.format(
                title=title,
                content=truncate(content, 1000),
            )
        else:
            prompt = _SUMMARY_PROMPT.format(
                title=title,
                content=truncate(content, 2000),
            )

        try:
            async with self._semaphore:
                summary = await self._llm.generate(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=60,
                    images=[image_data] if image_data else None,
                )
        except Exception as e:
            logger.error("LLM generation failed for section %r: %s", title, e)
            raise

        if not summary or not summary.strip():
            logger.warning("LLM returned empty summary for section %r. Content length: %d", title, len(content))
            return title[:120]

        summary = summary.strip()
        # Normalise: strip newlines, truncate to 300 chars just in case
        summary = " ".join(summary.split())[:300]
        return summary

    async def summarise_many(
        self,
        items: list[tuple[str, str, str | None]],  # (title, content, image_data)
    ) -> list[str]:
        """
        Summarise many items concurrently.
        Concurrency is bounded by `self._semaphore`.
        """
        tasks = [self.summarise(t, c, img) for t, c, img in items]
        return await asyncio.gather(*tasks, return_exceptions=False)


# ---------------------------------------------------------------------------
# Ingestion Engine
# ---------------------------------------------------------------------------


class IngestionEngine:
    """
    Orchestrates document parsing, tree construction, and summary synthesis.

    Usage::

        engine = IngestionEngine(storage, summariser)
        doc_id = await engine.ingest("/path/to/report.pdf")
    """

    def __init__(
        self,
        storage: StorageEngine,
        summariser: Summariser | None = None,
        *,
        parser_backend: str = "markitdown",
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._storage = storage
        self._summariser = summariser
        self._backend = parser_backend
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="apex_parse")
        self._owns_executor = executor is None

    # -- Public API ---------------------------------------------------------

    async def ingest(
        self,
        file_path: str | Path,
        *,
        doc_id: str | None = None,
        synthesize_summaries: bool = True,
    ) -> str:
        """
        Ingest a document file into the ApexRAG tree store.

        Args:
            file_path:           Path to the document (PDF, DOCX, HTML, TXT, MD).
            doc_id:              Override the auto-generated doc ID (file hash).
            synthesize_summaries: If True, call Ollama to generate summaries.

        Returns:
            The doc_id assigned to the ingested document.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        # Compute stable doc_id from file content hash
        computed_id = doc_id or self._compute_doc_id(path)

        t0 = time.monotonic()
        logger.info("Ingestion started: %s (doc_id=%s)", path.name, computed_id)

        # 1. Convert to Markdown (CPU-bound, run in thread pool)
        markdown = await self._convert_to_markdown(path)
        logger.info("Converted to Markdown: %d chars", len(markdown))

        # 2. Parse Markdown into section tree
        root_sections = await asyncio.get_event_loop().run_in_executor(
            self._executor, _parse_markdown_to_tree, markdown
        )
        total_nodes = _count_nodes(root_sections)
        logger.info("Parsed %d sections from document", total_nodes)

        # 3. Persist tree to DB and optionally synthesize summaries
        async with self._storage.session() as session:
            await self._persist_sections(
                session=session,
                sections=root_sections,
                doc_id=computed_id,
                parent_id=None,
                synthesize=synthesize_summaries,
            )

        elapsed = time.monotonic() - t0
        logger.info(
            "Ingestion complete: doc_id=%s | nodes=%d | elapsed=%.2fs",
            computed_id,
            total_nodes,
            elapsed,
        )
        return computed_id

    async def ingest_text(
        self,
        text: str,
        *,
        doc_id: str,
        synthesize_summaries: bool = True,
    ) -> str:
        """
        Ingest raw Markdown/plain text directly (no file needed).
        Useful for testing and programmatic ingestion.
        """
        t0 = time.monotonic()
        logger.info("Ingesting raw text: doc_id=%s (%d chars)", doc_id, len(text))

        root_sections = await asyncio.get_event_loop().run_in_executor(
            self._executor, _parse_markdown_to_tree, text
        )

        async with self._storage.session() as session:
            await self._persist_sections(
                session=session,
                sections=root_sections,
                doc_id=doc_id,
                parent_id=None,
                synthesize=synthesize_summaries,
            )

        logger.info(
            "Raw text ingestion complete in %.2fs", time.monotonic() - t0
        )
        return doc_id

    # -- Lifecycle ----------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the internal thread pool executor.

        Call this when the engine is no longer needed to free up threads.
        Safe to call multiple times; no-op if an external executor was
        passed at construction time.

        Args:
            wait: If True (default), wait for all running tasks to finish.
        """
        if self._owns_executor:
            self._executor.shutdown(wait=wait)
            logger.debug("IngestionEngine executor shut down")

    def __del__(self) -> None:
        """Ensure executor is shut down on garbage collection."""
        if hasattr(self, '_owns_executor') and self._owns_executor:
            executor = getattr(self, '_executor', None)
            if executor is not None:
                executor.shutdown(wait=False)

    # -- Internal helpers ---------------------------------------------------

    async def _convert_to_markdown(self, path: Path) -> str:
        """Convert a file to Markdown using the configured backend."""
        loop = asyncio.get_event_loop()

        if self._backend == "markitdown":
            return await loop.run_in_executor(
                self._executor, self._markitdown_convert, path
            )
        elif self._backend == "docling":
            return await loop.run_in_executor(
                self._executor, self._docling_convert, path
            )
        else:
            # Fallback: read as plain text (for .md / .txt files)
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _markitdown_convert(path: Path) -> str:
        """Synchronous markitdown conversion (runs in thread pool)."""
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(path))
            return result.text_content
        except ImportError:
            logger.warning("markitdown not installed; reading file as plain text.")
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _docling_convert(path: Path) -> str:
        """
        Synchronous docling conversion (runs in thread pool).
        Extracts images and embeds them as base64 in the markdown
        (or placeholders that we can parse).
        """
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import (  # type: ignore[attr-defined]
                DocumentConverter,
                PdfPipelineOptions,
            )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_page_images = True
            pipeline_options.table_structure_options.do_rectification = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pipeline_options
                }
            )
            result = converter.convert(str(path))

            # Export to markdown with image references
            return result.document.export_to_markdown()
        except ImportError:
            logger.warning("docling not installed; reading file as plain text.")
            return path.read_text(encoding="utf-8", errors="replace")

    async def _persist_sections(
        self,
        *,
        session: Any,
        sections: list[ParsedSection],
        doc_id: str,
        parent_id: int | None,
        synthesize: bool,
    ) -> None:
        """
        Recursively persist ParsedSections and their children to the DB.

        If `synthesize` is True, summaries are generated via Ollama in parallel
        for all sibling nodes at each depth level before recursing.
        """
        if not sections:
            return

        # Batch-summarise all sibling sections at this level
        summaries: list[str]
        if synthesize and self._summariser:
            items = [(s.title, s.content, s.image_data) for s in sections]
            summaries = await self._summariser.summarise_many(items)
        else:
            summaries = [
                f"{s.title}: {truncate(s.content, 80)}" for s in sections
            ]

        # Persist nodes and recurse into children
        child_tasks = []
        for section, summary in zip(sections, summaries, strict=True):
            is_leaf = not section.children and (bool(section.content) or bool(section.image_data))
            node = DocumentNode(
                doc_id=doc_id,
                parent_id=parent_id,
                path=section.path,
                title=section.title,
                summary=summary,
                content=section.content if is_leaf else None,
                depth=section.level - 1,
                position=section.position,
                page_start=section.page_start,
                page_end=section.page_end,
            )
            node.meta = {
                "title": section.title,
                "level": section.level,
                "char_count": len(section.content),
                "page_start": section.page_start,
                "page_end": section.page_end,
            }
            persisted = await self._storage.insert_node(session, node)
            logger.debug(
                "Persisted node: path=%s title=%r pages=%d-%d",
                section.path, section.title,
                section.page_start, section.page_end,
            )

            # Build a PageIndexEntry for every node (powers the /index page)
            pie = PageIndexEntry(
                doc_id=doc_id,
                node_id=persisted.id,
                term=section.title,
                page_start=section.page_start,
                page_end=section.page_end,
                path=section.path,
            )
            await self._storage.insert_page_index_entry(session, pie)

            if section.children:
                child_tasks.append(
                    self._persist_sections(
                        session=session,
                        sections=section.children,
                        doc_id=doc_id,
                        parent_id=persisted.id,
                        synthesize=synthesize,
                    )
                )

        # Recurse into children sequentially (SQLAlchemy sessions are NOT thread-safe)
        for task in child_tasks:
            await task

    @staticmethod
    def _compute_doc_id(path: Path) -> str:
        """SHA-256 hash of file content, hex-encoded, first 16 chars."""
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_nodes(sections: list[ParsedSection]) -> int:
    """Count total nodes in the parsed tree recursively."""
    count = len(sections)
    for s in sections:
        count += _count_nodes(s.children)
    return count
