from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    """
    Enterprise Multi-Tenant Context.
    Passed through the entire application stack to guarantee data isolation.
    """

    tenant_id: str = Field(..., description="Unique identifier for the workspace or tenant")
    user_id: str = Field(..., description="ID of the user making the request")
    roles: list[str] = Field(
        default_factory=list, description="RBAC roles (e.g. 'admin', 'reader')"
    )


class APIKey(BaseModel):
    """
    Representation of an authenticated API Key.
    """

    key_hash: str
    tenant_id: str
    is_active: bool = True
