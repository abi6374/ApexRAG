# conftest.py — shared pytest fixtures for ApexRAG test suite.

import pytest

# Configure asyncio mode globally (also set in pyproject.toml for redundancy)
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(autouse=True)
async def _dispose_apex_indexes(monkeypatch):
    """Auto-dispose every ApexIndex created during a test.

    Without this, each ``ApexIndex.create(...)`` call opens a DB engine/
    connection -- for the ``sqlite+aiosqlite`` URLs almost every test uses,
    that's an aiosqlite connection backed by its own dedicated background
    OS thread (see ``aiosqlite/core.py::_connection_worker_thread``) -- that
    is never released unless the caller explicitly awaits ``index.close()``.
    Across this ~900-test suite, essentially no fixture does that, so every
    test leaks one more idle thread.

    In isolation this is invisible (a handful of leaked threads do nothing
    observable). Deep into the full suite, with hundreds of leaked threads
    accumulated, the process reliably hangs -- confirmed with
    ``pytest-timeout``'s thread-dump: the hang sits inside asyncio's
    event-loop-close / task-cancellation path (Windows ProactorEventLoop
    waiting on ``GetQueuedCompletionStatus``), not in any single test's
    logic. No test file hangs when run alone.

    This fixture makes cleanup automatic and centralized -- it wraps
    ``ApexIndex.create`` for the duration of each test, tracks every
    instance created, and disposes them all in teardown -- instead of
    requiring an audit of every test/fixture in the suite to remember to
    call ``.close()``.
    """
    from apex_rag.client import ApexIndex

    created: list[ApexIndex] = []
    original_create = ApexIndex.create.__func__

    async def _tracking_create(cls, *args, **kwargs):
        index = await original_create(cls, *args, **kwargs)
        created.append(index)
        return index

    monkeypatch.setattr(ApexIndex, "create", classmethod(_tracking_create))

    yield

    for index in created:
        try:
            await index.close()
        except Exception:
            pass  # best-effort cleanup -- a close() failure shouldn't fail the test
