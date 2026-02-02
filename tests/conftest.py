"""Pytest configuration and shared fixtures."""

import pytest

from frame.memory_cache import reset_memory_cache


@pytest.fixture(autouse=True)
def reset_memory_cache_for_all_tests():
    """Reset the memory cache before and after each test for isolation.

    The memory cache is a module-level singleton that persists across tests.
    This fixture ensures each test starts with a clean cache state.
    """
    reset_memory_cache()
    yield
    reset_memory_cache()
