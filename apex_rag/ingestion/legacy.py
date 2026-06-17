"""
ingestion.py — Document parsing and Decision Tree synthesis for ApexRAG.

Pipeline:
    1. Convert the source file (PDF, DOCX, HTML, plain text) to Markdown
       using the ApexParser (which uses `markitdown` / `docling` internally).
    2. Parse the Markdown into a flat list of :class:`ASTNode` objects,
       then convert to a hierarchy of :class:`ParsedSection` objects.
    3. Image files (PNG, JPG, WebP, etc.) are parsed via :class:`ImageParser`
       and converted to :class:`ParsedSection` with ``image_data``.
    4. Persist a `DocumentNode` tree into the StorageEngine.
    5. Synthesize "Semantic Map" summaries for every node in parallel
       using the configured LLM, with configurable concurrency.

The ingestion is fully async-first. The ApexParser handles CPU-bound parsing.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_rag.ingestion.apex_parser import ApexParser
from apex_rag.models.unified_models import ASTNode, NodeType
from apex_rag.providers import AsyncLLM
from apex_rag.retrieval.vision.parser import SUPPORTED_EXTENSIONS as _IMAGE_EXTENSIONS
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
# ASTNode → ParsedSection Converter
# ---------------------------------------------------------------------------


def _collect_descendant_text(node_id: str, node_map: dict[str, ASTNode]) -> str:
    """
    Recursively collect text content from a node and all its descendants.

    This handles the case where ``ApexParser``'s internal chunking has
    split a leaf node's content into child chunk nodes.  By recursing
    into all descendants we recover the full content, regardless of how
    deeply it was chunked.
    """
    node = node_map.get(node_id)
    if not node:
        return ""
    parts: list[str] = []
    if node.content:
        parts.append(node.content)
    for child_id in node.children:
        child_text = _collect_descendant_text(child_id, node_map)
        if child_text:
            parts.append(child_text)
    return "\n\n".join(parts)


def _ast_nodes_to_parsed_sections(nodes: list[ASTNode]) -> list[ParsedSection]:
    """
    Convert a flat list of :class:`ASTNode` objects into a hierarchical
    list of :class:`ParsedSection` objects.

    Only ``HEADING``-type ASTNodes become ``ParsedSection`` objects.
    Non-heading children (PARAGRAPH, TABLE, CODE) are collected into
    the parent section's ``content`` field via recursive descent so
    that content chunked by ``ApexParser`` is still fully recoverable.
    """
    node_map = {n.node_id: n for n in nodes}

    def _build_section(
        node: ASTNode,
        position: int,
        parent_path: str | None,
    ) -> ParsedSection:
        """Recursively build a ParsedSection tree from an ASTNode heading."""
        # Recursively collect ALL descendant text (handles ApexParser chunking)
        content_parts: list[str] = []
        for child_id in node.children:
            child = node_map.get(child_id)
            if child and child.node_type != NodeType.HEADING:
                child_text = _collect_descendant_text(child_id, node_map)
                if child_text:
                    content_parts.append(child_text)
        content = "\n\n".join(content_parts).strip()

        path = build_ltree_path(parent_path, position)

        # Map ApexParser's implicit root title back to the legacy "Document" title
        title = node.content
        if title == "(implicit root)":
            title = "Document"

        section = ParsedSection(
            level=node.depth + 1,
            title=title,
            content=content,
            path=path,
            position=position,
            page_start=node.page_number or 0,
            page_end=node.page_number or 0,
        )

        # Build child sections from sub-headings
        child_pos = 0
        for child_id in node.children:
            child = node_map.get(child_id)
            if child and child.node_type == NodeType.HEADING:
                child_pos += 1
                child_section = _build_section(child, child_pos, path)
                section.children.append(child_section)

        return section

    # Find root-level headings (parent_id is None)
    root_headings = [n for n in nodes if n.parent_id is None and n.node_type == NodeType.HEADING]

    # If no headings exist, create a single section from all content
    if not root_headings:
        all_content = "\n\n".join(n.content for n in nodes if n.content).strip()
        if all_content:
            return [
                ParsedSection(
                    level=1,
                    title="Document",
                    content=all_content,
                    path="1",
                    position=1,
                )
            ]
        return []

    result: list[ParsedSection] = []
    for i, root in enumerate(root_headings, 1):
        section = _build_section(root, i, None)
        result.append(section)

    return result


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
            logger.warning(
                "LLM returned empty summary for section %r. Content length: %d", title, len(content)
            )
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

    Args:
        storage:                  StorageEngine instance.
        summariser:               Optional Summariser for generating LLM summaries.
        parser_backend:           Parser backend (``markitdown``, ``docling``, or ``plaintext``).
        parse_images_with_vision: If True, use LLM vision to generate summaries for image
                                  nodes via :class:`VisionAdapter`.  Requires ``summariser``
                                  to be set and the underlying LLM to support ``images``.
    """

    def __init__(
        self,
        storage: StorageEngine,
        summariser: Summariser | None = None,
        *,
        parser_backend: str = "markitdown",
        parse_images_with_vision: bool = False,
        apex_parser: ApexParser | None = None,
    ) -> None:
        self._storage = storage
        self._summariser = summariser
        self._backend = parser_backend
        self._parse_images_with_vision = parse_images_with_vision
        self._apex_parser = apex_parser or ApexParser(default_doc_id="legacy")

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
            file_path:           Path to the document (PDF, DOCX, HTML, TXT, MD, or
                                 supported image formats: PNG, JPG, WebP, BMP, etc.).
            doc_id:              Override the auto-generated doc ID (file hash).
            synthesize_summaries: If True, call the LLM to generate summaries.
                                  For image files, the ``parse_images_with_vision``
                                  constructor flag controls whether vision-capable
                                  LLM summaries are generated.

        Returns:
            The doc_id assigned to the ingested document.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        # Compute stable doc_id from file content hash
        computed_id = doc_id or self._compute_doc_id(path)

        # Short-circuit for image files
        ext = path.suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            return await self._ingest_image(
                path,
                doc_id=computed_id,
                synthesize_summaries=synthesize_summaries,
            )

        t0 = time.monotonic()
        logger.info("Ingestion started: %s (doc_id=%s)", path.name, computed_id)

        # 1. Parse file via ApexParser (handles markitdown/docling/text conversion)
        ast_nodes = await self._apex_parser.parse_file(path, doc_id=computed_id)
        logger.info("Parsed file into %d AST nodes", len(ast_nodes))

        # 2. Convert AST nodes into ParsedSection hierarchy for persistence
        root_sections = _ast_nodes_to_parsed_sections(ast_nodes)
        total_nodes = _count_nodes(root_sections)
        logger.info("Converted to %d ParsedSections", total_nodes)

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

        # 1. Parse via ApexParser
        ast_nodes = self._apex_parser.parse_markdown(text, doc_id=doc_id)

        # 2. Convert to ParsedSections for persistence
        root_sections = _ast_nodes_to_parsed_sections(ast_nodes)

        async with self._storage.session() as session:
            await self._persist_sections(
                session=session,
                sections=root_sections,
                doc_id=doc_id,
                parent_id=None,
                synthesize=synthesize_summaries,
            )

        logger.info("Raw text ingestion complete in %.2fs", time.monotonic() - t0)
        return doc_id

    # -- Lifecycle ----------------------------------------------------------

    # -- Image ingestion ----------------------------------------------------

    async def _ingest_image(
        self,
        path: Path,
        *,
        doc_id: str,
        synthesize_summaries: bool,
    ) -> str:
        """Ingest a single image file into the tree store.

        Uses :class:`ImageParser` to parse the image, then optionally
        generates a vision-powered summary via :class:`Summariser` if
        ``parse_images_with_vision`` is enabled.

        Args:
            path:                  Path to the image file.
            doc_id:                Document ID (computed from file hash).
            synthesize_summaries:  Whether to generate summaries.

        Returns:
            The ``doc_id`` assigned to the ingested image.
        """
        t0 = time.monotonic()
        logger.info(
            "Image ingestion started: %s (doc_id=%s, vision=%s)",
            path.name,
            doc_id,
            self._parse_images_with_vision,
        )

        # Parse the image via ImageParser (produces a single ASTNode)
        from apex_rag.retrieval.vision.parser import ImageParser

        parser = ImageParser(default_doc_id=doc_id)
        nodes = await parser.parse_file(path, doc_id=doc_id)

        # Convert ASTNode to a ParsedSection for the persistence layer
        ast_node = nodes[0]
        section = ParsedSection(
            level=1,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            content=ast_node.content or "",
            path="1",
            position=1,
            image_data=ast_node.image_data,
        )

        # If vision summaries are requested and a summariser is available,
        # generate a description using the LLM with image input
        use_vision = synthesize_summaries and self._parse_images_with_vision

        if use_vision and self._summariser:
            # The Summariser.summarise() method already handles image_data
            # by switching to _VISION_SUMMARY_PROMPT and passing images=.
            summary = await self._summariser.summarise(
                title=section.title,
                content=section.content,
                image_data=section.image_data,
            )
        elif synthesize_summaries and self._summariser:
            # Standard (non-vision) summary from extracted OCR text
            summary = await self._summariser.summarise(
                title=section.title,
                content=section.content or "(image file)",
                image_data=None,
            )
        else:
            summary = f"{section.title}: {path.name}"

        # Persist the single image node
        async with self._storage.session() as session:
            node = DocumentNode(
                doc_id=doc_id,
                parent_id=None,
                path=section.path,
                title=section.title,
                summary=summary,
                content=section.content if section.content else None,
                depth=0,
                position=1,
                page_start=0,
                page_end=0,
                image_data=section.image_data,
            )
            node.meta = {
                "title": section.title,
                "level": 1,
                "char_count": len(section.content),
                "type": "image",
                "filename": path.name,
                "vision_summary": self._parse_images_with_vision,
            }
            persisted = await self._storage.insert_node(session, node)

            # Build a PageIndexEntry for the image node
            pie = PageIndexEntry(
                doc_id=doc_id,
                node_id=persisted.id,
                term=section.title,
                page_start=0,
                page_end=0,
                path=section.path,
            )
            await self._storage.insert_page_index_entry(session, pie)

        elapsed = time.monotonic() - t0
        logger.info(
            "Image ingestion complete: doc_id=%s | vision=%s | elapsed=%.2fs",
            doc_id,
            use_vision,
            elapsed,
        )
        return doc_id

    # -- Internal helpers ---------------------------------------------------

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
            summaries = [f"{s.title}: {truncate(s.content, 80)}" for s in sections]

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
                section.path,
                section.title,
                section.page_start,
                section.page_end,
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
