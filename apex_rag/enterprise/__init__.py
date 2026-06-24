"""
apex_rag.enterprise — Enterprise ecosystem components.

Sub-packages:

    - **auth**       — Multi-tenant RBAC models and FastAPI middleware.
    - **code_intel** — Code intelligence (Python AST parser).
    - **distributed** — Distributed ingestion queue protocol and implementations.
"""

from typing import Any

from apex_rag.enterprise.auth.access_control import AccessControlAgent, Roles
from apex_rag.enterprise.auth.models import APIKey, TenantContext
from apex_rag.enterprise.code_intel.parser import PythonCodeParser


def __getattr__(name: str) -> Any:
    if name == "get_tenant_context":
        try:
            from apex_rag.enterprise.auth.middleware import get_tenant_context

            return get_tenant_context
        except ImportError as e:
            raise ImportError(
                "fastapi is required to use get_tenant_context. Install it with: pip install 'apex-rag[web]'"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TenantContext",
    "APIKey",
    "get_tenant_context",
    "Roles",
    "AccessControlAgent",
    "PythonCodeParser",
]
