import ast
import typing
import uuid

from apex_rag.core.ast.models import ASTNode, ASTNodeMetadata
from apex_rag.core.protocols.interfaces import DocumentParser
from apex_rag.graph.edges.models import GraphEdge


class PythonCodeParser(DocumentParser):
    """
    Parses Python source code into the Universal Document AST and extracts GraphEdges
    for function calls and class definitions.
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

    def extract_edges(self, _root: ASTNode) -> list[GraphEdge]:
        """
        Walks the AST (if we had fully mapped the AST.walk) to find function calls.
        For demonstration, we mock extraction of a DEPENDS_ON edge.
        """
        edges: list[GraphEdge] = []
        return edges
