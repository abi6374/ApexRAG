import re
import typing
import uuid

from apex_rag.core.ast.models import ASTNode, ASTNodeMetadata
from apex_rag.core.protocols.interfaces import DocumentParser


class MarkdownASTParser(DocumentParser):
    """
    Parses Markdown text into the Universal Document AST.
    """

    def __init__(self) -> None:
        # Matches headings: # Heading 1, ## Heading 2
        self.heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

    async def parse(self, file_path: str, **kwargs: typing.Any) -> ASTNode:
        """
        Since we might want to pass raw text in testing, kwargs can accept 'raw_text'.
        Otherwise, reads from file_path.
        """
        raw_text = kwargs.get("raw_text")
        if not raw_text:
            with open(file_path, encoding="utf-8") as f:
                raw_text = f.read()

        return self._parse_text(raw_text, source=file_path)

    def _parse_text(self, text: str, source: str | None = None) -> ASTNode:
        root = ASTNode(
            id=str(uuid.uuid4()),
            node_type="Document",
            content="",
            metadata=ASTNodeMetadata(source_file=source),
        )

        lines = text.split("\n")

        # Stack to keep track of the current hierarchy (level, node)
        # Root is treated as level 0
        stack: list[tuple[int, ASTNode]] = [(0, root)]

        current_paragraph_lines: list[str] = []

        def flush_paragraph() -> None:
            if current_paragraph_lines:
                para_text = "\n".join(current_paragraph_lines).strip()
                if para_text:
                    para_node = ASTNode(
                        id=str(uuid.uuid4()),
                        node_type="Paragraph",
                        content=para_text,
                        metadata=ASTNodeMetadata(source_file=source),
                    )
                    # Attach to the deepest section in the stack
                    stack[-1][1].add_child(para_node)
                current_paragraph_lines.clear()

        for line in lines:
            heading_match = self.heading_pattern.match(line)
            if heading_match:
                flush_paragraph()

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                section_node = ASTNode(
                    id=str(uuid.uuid4()),
                    node_type="Section",
                    content=title,
                    metadata=ASTNodeMetadata(source_file=source),
                )

                # Pop from stack until we find a parent with a lower level
                while len(stack) > 1 and stack[-1][0] >= level:
                    stack.pop()

                stack[-1][1].add_child(section_node)
                stack.append((level, section_node))
            else:
                if line.strip() == "":
                    flush_paragraph()
                else:
                    current_paragraph_lines.append(line)

        flush_paragraph()

        return root
