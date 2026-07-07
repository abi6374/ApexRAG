import ast
import typing
import uuid

from apex_rag.core.ast.models import ASTNode, ASTNodeMetadata
from apex_rag.core.protocols.interfaces import DocumentParser


class PythonCodeParser(DocumentParser):
    """
    Parses Python source code into the Universal Document AST.
    """

    async def parse(self, file_path: str, **kwargs: typing.Any) -> ASTNode:
        raw_text = kwargs.get("raw_text")
        if not raw_text:
            with open(file_path, encoding="utf-8") as f:
                raw_text = f.read()

        return self._parse_code(raw_text, source=file_path)

    def _parse_code(self, code: str, source: str | None = None) -> ASTNode:
        root = ASTNode(
            id=str(uuid.uuid4()),
            node_type="Module",
            content=source or "anonymous_module",
            metadata=ASTNodeMetadata(source_file=source),
        )

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback for broken code
            root.content = "Failed to parse Python code"
            return root

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_node = ASTNode(
                    id=str(uuid.uuid4()),
                    node_type="FunctionDef",
                    content=node.name,
                    metadata=ASTNodeMetadata(source_file=source, custom={"lineno": node.lineno}),
                )

                # Extract docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    doc_node = ASTNode(
                        id=str(uuid.uuid4()),
                        node_type="DocString",
                        content=docstring,
                        parent_id=func_node.id,
                    )
                    func_node.children.append(doc_node)

                root.add_child(func_node)

            elif isinstance(node, ast.ClassDef):
                class_node = ASTNode(
                    id=str(uuid.uuid4()),
                    node_type="ClassDef",
                    content=node.name,
                    metadata=ASTNodeMetadata(source_file=source, custom={"lineno": node.lineno}),
                )
                root.add_child(class_node)

        return root

    def extract_edges(self, _root: ASTNode) -> list:
        """
        Placeholder for AST-based edge extraction.

        Intended to walk the parsed AST and discover function call edges
        (CALLS, IMPORTS, DEPENDS_ON). Currently returns an empty list;
        implement with a full AST walker when call-graph analysis is needed.
        """
        return []
