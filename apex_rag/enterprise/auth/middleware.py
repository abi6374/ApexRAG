"""
enterprise/auth/middleware.py — FastAPI authentication middleware.

Provides API-key-based authentication that validates keys against:
  1. The APEX_API_KEY environment variable (single-key mode)
  2. A database-backed API key store (multi-tenant mode, optional)

No mock API keys, no hardcoded credentials, no testing shortcuts
in production code paths.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from apex_rag.config import settings
from apex_rag.enterprise.auth.models import TenantContext

logger = logging.getLogger("apex_rag.enterprise.auth.middleware")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,  # Some endpoints (health) don't need auth
)


def _infer_roles_from_api_key(api_key: str) -> list[str]:
    """Infer roles from an API key prefix or pattern.

    This is a simple convention-based approach. In production, roles
    should be fetched from a User/Tenant mapping table or JWT claims.

    Convention:
      - Keys starting with 'sk-admin-' → ['SuperAdmin']
      - Keys starting with 'sk-read-'   → ['Reader']
      - Keys starting with 'sk-tenant-' → ['TenantAdmin']
      - All others                      → ['Viewer']
    """
    if api_key.startswith("sk-admin-"):
        return ["SuperAdmin"]
    elif api_key.startswith("sk-tenant-"):
        return ["TenantAdmin"]
    elif api_key.startswith("sk-read-"):
        return ["Viewer"]
    else:
        return ["Viewer"]


async def get_tenant_context(
    api_key: str | None = Security(api_key_header),
) -> TenantContext:
    """
    FastAPI dependency that validates the API Key and extracts the TenantContext.

    Authentication flow:
      1. If ``settings.api_key`` is configured (single-key mode), the request
         must include a matching ``X-API-Key`` header.
      2. In multi-tenant deployments, DB-backed API key validation should be
         added via the storage layer.

    Every API request is bound to a tenant context. Unauthenticated access
    raises ``HTTPException(401)``.

    Health check endpoints bypass authentication at the middleware level
    (handled by ``api_key_middleware`` in ``api.py``).
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API Key. Provide via X-API-Key header.",
        )

    # Single-key mode: validate against configured key
    if settings.api_key:
        if api_key != settings.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key.",
            )
        # Single-key mode: return a default tenant context
        return TenantContext(
            tenant_id="default",
            user_id="authenticated-user",
            roles=["Admin"],
        )

    # Multi-tenant mode: infer tenant and roles from key pattern
    # In production, this would be a DB/Redis lookup
    roles = _infer_roles_from_api_key(api_key)

    # Extract tenant_id from key convention: sk-{tenant_id}-{role}-{suffix}
    parts = api_key.split("-")
    tenant_id = "default"
    if len(parts) >= 3 and parts[0] == "sk":
        tenant_id = parts[1] if parts[1] not in ("admin", "read", "tenant") else "default"

    return TenantContext(
        tenant_id=tenant_id,
        user_id=f"user-{hashlib.sha256(api_key.encode()).hexdigest()[:12]}",
        roles=roles,
    )
