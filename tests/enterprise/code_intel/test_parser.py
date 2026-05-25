import pytest

from apex_rag.enterprise.code_intel.parser import PythonCodeParser


@pytest.mark.asyncio
async def test_python_code_parser():
    code = '''
def calculate_revenue(q2, q3):
    """Calculates the total revenue."""
    return q2 + q3

class FinancialEngine:
    pass
    '''

    parser = PythonCodeParser()
    root = await parser.parse("financials.py", raw_text=code)

    assert root.node_type == "Module"
    assert len(root.children) == 2

    func_node = root.children[0]
    assert func_node.node_type == "FunctionDef"
    assert func_node.content == "calculate_revenue"

    # Docstring should be a child
    assert len(func_node.children) == 1
    assert func_node.children[0].node_type == "DocString"
    assert "Calculates the total revenue" in func_node.children[0].content

    class_node = root.children[1]
    assert class_node.node_type == "ClassDef"
    assert class_node.content == "FinancialEngine"
