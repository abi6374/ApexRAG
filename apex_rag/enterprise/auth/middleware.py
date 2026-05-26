from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from apex_rag.enterprise.auth.models import APIKey, TenantContext

api_key_header = APIKeyHeader(name="X-API-Key")

# In a real enterprise system, this would be a Redis lookup or DB query
MOCK_API_KEYS = {
    "sk-test-admin-123": APIKey(key_hash="hash1", tenant_id="tenant_a", is_active=True),
    "sk-test-readonly-456": APIKey(key_hash="hash2", tenant_id="tenant_b", is_active=True),
}


async def get_tenant_context(api_key: str = Security(api_key_header)) -> TenantContext:
    """
    FastAPI dependency that validates the API Key and extracts the TenantContext.
    Ensures that every request is strictly bound to a tenant.
    """
    key_record = MOCK_API_KEYS.get(api_key)

    if not key_record or not key_record.is_active:
        raise HTTPException(status_code=401, detail="Invalid or inactive API Key")

    # In a full system, roles would be fetched from a User/Tenant mapping table
    roles = ["admin"] if "admin" in api_key else ["reader"]

    return TenantContext(
        tenant_id=key_record.tenant_id,
        user_id="inferred-user-id",  # Extracted from JWT or API key metadata
        roles=roles,
    )
