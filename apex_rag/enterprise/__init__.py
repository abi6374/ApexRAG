"""
apex_rag.enterprise — Enterprise ecosystem components.

Sub-packages:

    - **auth**       — Multi-tenant RBAC models and FastAPI middleware.
    - **code_intel** — Code intelligence (Python AST parser).
    - **distributed** — Distributed ingestion queue protocol and implementations.
"""

from apex_rag.enterprise.auth.models import APIKey, TenantContext
from apex_rag.enterprise.auth.middleware import get_tenant_context
from apex_rag.enterprise.auth.access_control import Roles, AccessControlAgent
from apex_rag.enterprise.code_intel.parser import PythonCodeParser

__all__ = [
    "TenantContext",
    "APIKey",
    "get_tenant_context",
    "Roles",
    "AccessControlAgent",
    "PythonCodeParser",
]
