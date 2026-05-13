# conftest.py — shared pytest fixtures for ApexRAG test suite.
import pytest

# Configure asyncio mode globally (also set in pyproject.toml for redundancy)
pytest_plugins = ["pytest_asyncio"]
