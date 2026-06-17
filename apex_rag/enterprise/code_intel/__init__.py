"""
apex_rag.enterprise.code_intel — Code intelligence for structural retrieval.

Provides parsers that convert source code into the Universal Document AST,
enabling structural reasoning over codebases alongside document corpora.
"""

from apex_rag.enterprise.code_intel.parser import PythonCodeParser

__all__ = [
    "PythonCodeParser",
]
