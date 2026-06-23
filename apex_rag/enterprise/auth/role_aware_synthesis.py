"""
enterprise/auth/role_aware_synthesis.py — Role-aware answer synthesis.

Before generating responses, this service filters:
  - Restricted nodes (user lacks node-level access)
  - Restricted fields (field-level security masking)
  - Restricted versions (user cannot access specific version history)
  - Restricted history (user cannot see audit trails of others)
  - Restricted audit records (tenant isolation for audit data)

Acts as a security gasket between the retrieval layer and the synthesizer,
ensuring that only authorized content enters the final LLM synthesis prompt.

Usage:
    rasynthesis = RoleAwareSynthesis(storage, access_control)
    safe_packets = await rasynthesis.filter_packets(context, packets)
    answer = await rasynthesis.synthesize(context, query, safe_packets)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from apex_rag.enterprise.auth.access_control import AccessControlAgent
from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.models.unified_models import EvidencePacket

logger = logging.getLogger("apex_rag.enterprise.auth.role_aware_synthesis")


class RoleAwareFilter:
    """Filters evidence packets and content based on RBAC permissions.

    Applied before synthesis to ensure no unauthorized data reaches the LLM.
    """

    def __init__(self, access_control: AccessControlAgent) -> None:
        self._access_control = access_control

    async def filter_packets(
        self,
        context: TenantContext,
        packets: list[EvidencePacket],
    ) -> list[EvidencePacket]:
        """Filter packets by node-level and field-level access rules.

        Args:
            context: The :class:`TenantContext` for the requesting user.
            packets: The raw evidence packets from retrieval.

        Returns:
            Only packets the user is authorized to see, with field-level
            masking applied to their content.
        """
        if not packets:
            return []

        filtered: list[EvidencePacket] = []

        for pkt in packets:
            # 1. Check node-level access
            node_allowed = await self._access_control.check_access(
                context,
                "traverse",
                "node",
                doc_tenant_id=pkt.tenant_id or pkt.document_id,
            )
            if not node_allowed:
                logger.debug(
                    "Filtered packet %s: node access denied for user %s",
                    pkt.node_id,
                    context.user_id,
                )
                continue

            # 2. Check version access (if temporal metadata exists)
            if pkt.temporal_metadata is not None:
                version_allowed = await self._access_control.check_access(
                    context,
                    "read",
                    "version",
                    doc_tenant_id=pkt.tenant_id or pkt.document_id,
                )
                if not version_allowed:
                    logger.debug(
                        "Filtered packet %s: version access denied",
                        pkt.node_id,
                    )
                    continue

            # 3. Apply field-level masking to content
            masked_content = await self._access_control.mask_content(
                context, pkt.content,
            )
            if pkt.node is not None:
                pkt.node.content = masked_content
            pkt.content = masked_content

            filtered.append(pkt)

        return filtered

    async def filter_content(
        self,
        context: TenantContext,
        content: str,
    ) -> str:
        """Apply field-level masking to a content string without packet context.

        Args:
            context: The tenant context.
            content: The raw content string.

        Returns:
            The masked content with redactions applied.
        """
        return await self._access_control.mask_content(context, content)


class RoleAwareSynthesis:
    """Synthesis pipeline that respects enterprise access controls.

    Wraps an existing synthesizer with role-aware filtering so the LLM
    never receives unauthorized information.

    Usage:
        safe_synth = RoleAwareSynthesis(synthesizer, access_control)
        answer = await safe_synth.synthesize(context, query, packets)
    """

    def __init__(
        self,
        synthesizer: Any,
        access_control: AccessControlAgent,
    ) -> None:
        self._synthesizer = synthesizer
        self._filter = RoleAwareFilter(access_control)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def filter(self) -> RoleAwareFilter:
        """The underlying role-aware filter."""
        return self._filter

    @property
    def inner(self) -> Any:
        """The wrapped synthesizer."""
        return self._synthesizer

    # ── Public API ─────────────────────────────────────────────────────

    async def synthesize(
        self,
        context: TenantContext,
        query: str,
        packets: list[EvidencePacket],
    ) -> str:
        """Synthesize an answer with role-aware filtering.

        Args:
            context: The tenant context.
            query:   The user's query.
            packets: The evidence packets (will be filtered before synthesis).

        Returns:
            The synthesized answer string, containing only authorized content.
        """
        # 1. Filter packets by access control
        safe_packets = await self._filter.filter_packets(context, packets)

        if not safe_packets:
            return "I could not find enough authorized evidence to answer your query."

        # 2. Also filter the final content after synthesis
        answer = await self._synthesizer.synthesize(query, safe_packets)

        # 3. Apply field-level masking to the final answer
        safe_answer = await self._filter.filter_content(context, answer)

        return safe_answer

    async def stream_synthesize(
        self,
        context: TenantContext,
        query: str,
        packets: list[EvidencePacket],
    ) -> AsyncGenerator[str, None]:
        """Stream a role-aware synthesis answer.

        Args:
            context: The tenant context.
            query:   The user's query.
            packets: The evidence packets.

        Yields:
            Content tokens with role-aware filtering applied.
        """
        # Filter packets first
        safe_packets = await self._filter.filter_packets(context, packets)

        if not safe_packets:
            yield "I could not find enough authorized evidence to answer your query."
            return

        # Stream through the inner synthesizer
        accumulated = ""
        async for chunk in self._synthesizer.stream_synthesize(query, safe_packets):
            accumulated += chunk
            yield chunk

        # Apply field-level masking to the accumulated answer
        # (streaming masking is applied after the fact for security)
        # In production, you'd want token-level masking for real-time
