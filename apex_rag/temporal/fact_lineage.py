"""
temporal/fact_lineage.py — Fact Lineage Engine and Lineage Validator.

Provides:
  - FactLineageEngine:   Trace fact origins, descendants, history, supersession chains.
  - LineageValidator:    Enforce DAG acyclicity at write time (Principles 3, 11).

PRINCIPLE 3 — DAG Lineage.
  Fact lineage is a directed acyclic graph (DAG).  Cycles are never allowed.

PRINCIPLE 11 — Enforce Acyclicity At Write Time.
  Cycle detection occurs during write operations, never during reads.
  Invalid writes are rejected immediately.

Usage:
    lineage = FactLineageEngine(storage)
    origin = await lineage.find_origin(fact_id)
    descendants = await lineage.find_descendants(fact_id)
    chain = await lineage.find_superseded_chain(fact_id)

    validator = LineageValidator(storage)
    await validator.validate_edge(parent_id, child_id)  # raises ValueError if cycle
"""

from __future__ import annotations

import logging
from collections import deque

from sqlalchemy import select

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.fact_store import FactRow, FactStore, TemporalFact

logger = logging.getLogger("apex_rag.temporal.fact_lineage")


# ═══════════════════════════════════════════════════════════════
# LineageValidator — Write-Time Cycle Detection
# ═══════════════════════════════════════════════════════════════


class LineageValidator:
    """Validates fact lineage edges for DAG acyclicity at write time.

    PRINCIPLE 3 — DAG Lineage.
    PRINCIPLE 11 — Enforce Acyclicity At Write Time.

    All cycle detection occurs during write operations.
    Rejected edges raise ValueError immediately.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    async def validate_edge(
        self,
        parent_fact_id: str,
        child_fact_id: str,
        max_depth: int = 50,
    ) -> None:
        """Validate that adding a parent → child edge does not create a cycle.

        Uses BFS from ``child_fact_id`` following parent links to see if
        ``parent_fact_id`` can be reached.  If yes, the edge would create
        a cycle and is rejected.

        Args:
            parent_fact_id: The origin fact in the proposed edge.
            child_fact_id:  The destination fact in the proposed edge.
            max_depth:      Maximum BFS depth guard.

        Raises:
            ValueError: If the edge would create a cycle.
        """
        if await self._would_create_cycle(parent_fact_id, child_fact_id, max_depth):
            raise ValueError(
                f"Cannot create lineage edge: {parent_fact_id} → {child_fact_id} "
                f"would create a cycle.  Cycles are rejected at write time "
                f"(Principle 11)."
            )

    async def detect_cycle(
        self,
        start_fact_id: str,
        max_depth: int = 50,
    ) -> bool:
        """Detect if a fact participates in a cycle.

        Follows parent_fact_id links from ``start_fact_id`` to see if
        any descendant can reach back to ``start_fact_id``.

        Args:
            start_fact_id: The fact to check.
            max_depth:     Maximum BFS depth.

        Returns:
            True if a cycle is detected.
        """
        return await self._follow_backward(start_fact_id, start_fact_id, max_depth)

    async def assert_acyclic(
        self,
        fact_id: str,
        max_depth: int = 50,
    ) -> None:
        """Assert that a fact's lineage is acyclic.

        Raises:
            ValueError: If a cycle is detected.
        """
        if await self.detect_cycle(fact_id, max_depth):
            raise ValueError(
                f"Fact {fact_id} participates in a cycle.  Lineage must be a DAG (Principle 3)."
            )

    async def _would_create_cycle(
        self,
        source_id: str,
        target_id: str,
        max_depth: int,
    ) -> bool:
        """Check if proposed edge source_id → target_id would create a cycle.

        Checks THREE directions:
          1. Backward from target_id following parent links — if we reach
             source_id, target_id is already an ancestor of source_id.
          2. Backward from source_id following parent links — if we reach
             target_id, then making source_id target_id's parent would create:
             target_id → source_id → ... → target_id (cycle).
          3. Forward from source_id following child links — if we reach
             target_id, source_id is already a descendant of target_id.

        Any path means the edge would create a cycle.
        """
        async with self._storage.session() as session:
            # Direction 1: Backward from target_id following parent links
            visited: set[str] = {target_id}
            bfs_queue: deque[str] = deque([target_id])
            depth = 0

            while bfs_queue and depth < max_depth:
                current_id = bfs_queue.popleft()
                if current_id == source_id:
                    return True

                stmt = select(FactRow).where(FactRow.fact_id == current_id)
                result = await session.execute(stmt)
                row = result.scalars().first()
                if row and row.parent_fact_id and row.parent_fact_id not in visited:
                    visited.add(row.parent_fact_id)
                    bfs_queue.append(row.parent_fact_id)

                depth += 1

            # Direction 2: Backward from source_id (proposed parent) following
            # parent links — if we reach target_id, then after making source_id
            # the parent of target_id, there would be a path:
            # target_id → source_id → ... → target_id (cycle).
            visited2: set[str] = {source_id}
            bfs_back: deque[str] = deque([source_id])
            depth = 0

            while bfs_back and depth < max_depth:
                current_id = bfs_back.popleft()
                if current_id == target_id:
                    return True

                stmt = select(FactRow).where(FactRow.fact_id == current_id)
                result = await session.execute(stmt)
                row = result.scalars().first()
                if row and row.parent_fact_id and row.parent_fact_id not in visited2:
                    visited2.add(row.parent_fact_id)
                    bfs_back.append(row.parent_fact_id)

                depth += 1

            # Direction 3: Forward from source_id following child links
            visited3: set[str] = {source_id}
            bfs_fwd: deque[str] = deque([source_id])
            depth = 0

            while bfs_fwd and depth < max_depth:
                current_id = bfs_fwd.popleft()

                stmt = select(FactRow).where(FactRow.parent_fact_id == current_id)
                result = await session.execute(stmt)
                for row in result.scalars().all():
                    if row.fact_id == target_id:
                        return True
                    if row.fact_id not in visited3:
                        visited3.add(row.fact_id)
                        bfs_fwd.append(row.fact_id)

                depth += 1

            return False

    async def _follow_backward(
        self,
        current_id: str,
        target_id: str,
        max_depth: int,
    ) -> bool:
        """BFS backward from current_id following parent links to see if we reach target_id."""
        async with self._storage.session() as session:
            visited: set[str] = set()
            bfs_queue: deque[str] = deque([current_id])
            depth = 0

            while bfs_queue and depth < max_depth:
                node_id = bfs_queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)

                stmt = select(FactRow).where(FactRow.parent_fact_id == node_id)
                result = await session.execute(stmt)
                for row in result.scalars().all():
                    if row.fact_id == target_id:
                        return True
                    if row.fact_id not in visited:
                        bfs_queue.append(row.fact_id)

                depth += 1

            return False


# ═══════════════════════════════════════════════════════════════
# FactLineageEngine — Lineage Chain Navigation
# ═══════════════════════════════════════════════════════════════


class FactLineageEngine:
    """Navigates fact lineage chains through the DAG.

    Supports:
      - find_origin():          Trace to the oldest ancestor fact.
      - find_descendants():     Find all facts derived from a given fact.
      - find_fact_history():    Get the full version history of a fact.
      - find_superseded_chain(): Get the supersession chain.
      - find_related_facts():   Find facts related by document or subject.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage
        self._fact_store = FactStore(storage)

    async def find_origin(
        self,
        fact_id: str,
        *,
        tenant_context: str | None = None,
    ) -> TemporalFact | None:
        """Trace to the original (oldest) version of a fact.

        Follows the ``parent_fact_id`` chain backward to the first fact
        that has no parent — the origin of this lineage.

        Args:
            fact_id:         The current fact ID.
            tenant_context:  Required tenant ID for isolation.

        Returns:
            The root/origin :class:`TemporalFact`, or None.
        """
        if not tenant_context:
            return None

        visited: set[str] = set()
        current_id = fact_id

        while current_id and current_id not in visited:
            visited.add(current_id)
            fact = await self._fact_store.get_fact(
                current_id,
                tenant_context=tenant_context,
            )
            if fact is None:
                return None
            if fact.parent_fact_id is None:
                return fact
            current_id = fact.parent_fact_id

        return None

    async def find_descendants(
        self,
        fact_id: str,
        *,
        tenant_context: str | None = None,
        max_depth: int = 10,
    ) -> list[TemporalFact]:
        """Find all facts that descend (directly or indirectly) from this fact.

        BFS traversal following ``parent_fact_id`` links forward.

        Args:
            fact_id:         The ancestor fact ID.
            tenant_context:  Required tenant ID.
            max_depth:       Maximum BFS depth.

        Returns:
            Descendant facts, ordered by depth (closest first).
        """
        if not tenant_context:
            return []

        async with self._storage.session() as session:
            results: list[TemporalFact] = []
            visited: set[str] = {fact_id}
            bfs_queue: deque[tuple[str, int]] = deque([(fact_id, 0)])

            while bfs_queue:
                current_id, depth = bfs_queue.popleft()
                if depth >= max_depth:
                    continue

                stmt = select(FactRow).where(
                    FactRow.parent_fact_id == current_id,
                    FactRow.tenant_id == tenant_context,
                )
                result = await session.execute(stmt)
                for row in result.scalars().all():
                    if row.fact_id not in visited:
                        visited.add(row.fact_id)
                        fact = self._fact_store._row_to_fact(row)
                        results.append(fact)
                        bfs_queue.append((row.fact_id, depth + 1))

            return results

    async def find_fact_history(
        self,
        fact_id: str,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Get the full version history of a fact lineage.

        Walks both backward (to origin) and forward (to latest descendant)
        to produce a complete chronological history.

        Args:
            fact_id:         The fact ID to center the history on.
            tenant_context:  Required tenant ID.

        Returns:
            Chronologically ordered list of all facts in this lineage.
        """
        if not tenant_context:
            return []

        all_facts: dict[str, TemporalFact] = {}

        # Walk backward to origin
        current_id = fact_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            fact = await self._fact_store.get_fact(
                current_id,
                tenant_context=tenant_context,
            )
            if fact is None:
                break
            all_facts[fact.fact_id] = fact
            current_id = fact.parent_fact_id or ""

        # Walk forward to descendants
        descendants = await self.find_descendants(
            fact_id,
            tenant_context=tenant_context,
        )
        for d in descendants:
            all_facts[d.fact_id] = d

        # Sort chronologically
        return sorted(all_facts.values(), key=lambda f: f.created_at)

    async def find_superseded_chain(
        self,
        fact_id: str,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Get the supersession chain starting from this fact.

        Follows ``superseded_by`` links to produce a chain of
        fact → superseder → superseder's superseder → ...

        Args:
            fact_id:         The starting fact ID.
            tenant_context:  Required tenant ID.

        Returns:
            The supersession chain (first = current fact, last = latest).
        """
        if not tenant_context:
            return []

        chain: list[TemporalFact] = []
        visited: set[str] = set()
        current_id = fact_id

        while current_id and current_id not in visited:
            visited.add(current_id)
            fact = await self._fact_store.get_fact(
                current_id,
                tenant_context=tenant_context,
            )
            if fact is None:
                break
            chain.append(fact)

            if fact.superseded_by and fact.superseded_by != "__DELETED__":
                current_id = fact.superseded_by
            else:
                break

        return chain

    async def find_related_facts(
        self,
        fact_id: str,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Find facts related to this fact by document or subject.

        Args:
            fact_id:         The reference fact ID.
            tenant_context:  Required tenant ID.

        Returns:
            Related facts (same document or same subject).
        """
        if not tenant_context:
            return []

        source = await self._fact_store.get_fact(
            fact_id,
            tenant_context=tenant_context,
        )
        if source is None:
            return []

        related: dict[str, TemporalFact] = {}

        # Same document
        doc_facts = await self._fact_store.get_facts_by_document(
            source.source_document_id,
            tenant_context=tenant_context,
        )
        for f in doc_facts:
            related[f.fact_id] = f

        # Same subject (if not too broad)
        if source.subject and len(source.subject) > 3:
            async with self._storage.session() as session:
                stmt = (
                    select(FactRow)
                    .where(
                        FactRow.subject == source.subject,
                        FactRow.tenant_id == tenant_context,
                    )
                    .limit(20)
                )
                result = await session.execute(stmt)
                for row in result.scalars().all():
                    related[row.fact_id] = self._fact_store._row_to_fact(row)

        # Remove self
        related.pop(fact_id, None)
        return list(related.values())
