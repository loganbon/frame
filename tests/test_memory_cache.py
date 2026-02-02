"""Tests for in-memory cache functionality."""

import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from frame import (
    Frame,
    CacheConfig,
    CacheStats,
    clear_memory_cache,
    configure_memory_cache,
    get_memory_cache_stats,
)
from frame.memory_cache import CacheKey, CacheEntry, MemoryCache, get_memory_cache


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


@pytest.fixture
def call_counter():
    """Track function call counts."""
    return {"count": 0}


@pytest.fixture
def sample_data_func(call_counter):
    """Sample data function that tracks calls."""
    def fetch_data(start_dt: datetime, end_dt: datetime, multiplier: int = 1):
        call_counter["count"] += 1
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for dt in dates:
            for id_ in range(3):
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "value": (id_ + 1) * multiplier,
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return fetch_data


class TestCacheKey:
    """Test CacheKey functionality."""

    def test_make_with_no_columns_or_filters(self):
        """CacheKey.make creates key with None for optional params."""
        path = Path("/tmp/test.prq")
        key = CacheKey.make(path)

        assert key.path == path
        assert key.columns is None
        assert key.filters is None

    def test_make_with_columns(self):
        """CacheKey.make sorts columns for consistency."""
        path = Path("/tmp/test.prq")
        key1 = CacheKey.make(path, columns=["b", "a", "c"])
        key2 = CacheKey.make(path, columns=["a", "b", "c"])
        key3 = CacheKey.make(path, columns=["c", "a", "b"])

        assert key1 == key2 == key3
        assert key1.columns == ("a", "b", "c")

    def test_make_with_filters(self):
        """CacheKey.make converts filters to tuples."""
        path = Path("/tmp/test.prq")
        key = CacheKey.make(path, filters=[("value", ">", 1), ("id", "=", 0)])

        assert key.filters == (("value", ">", 1), ("id", "=", 0))

    def test_different_columns_different_keys(self):
        """Different columns produce different keys."""
        path = Path("/tmp/test.prq")
        key1 = CacheKey.make(path, columns=["a"])
        key2 = CacheKey.make(path, columns=["b"])

        assert key1 != key2

    def test_different_filters_different_keys(self):
        """Different filters produce different keys."""
        path = Path("/tmp/test.prq")
        key1 = CacheKey.make(path, filters=[("value", ">", 1)])
        key2 = CacheKey.make(path, filters=[("value", ">", 2)])

        assert key1 != key2


class TestMemoryCache:
    """Test MemoryCache class."""

    def test_put_and_get(self):
        """Basic put and get operations work."""
        cache = MemoryCache()
        path = Path("/tmp/test.prq")
        df = pd.DataFrame({"a": [1, 2, 3]})

        cache.put(path, df)
        result = cache.get(path)

        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_get_miss_returns_none(self):
        """Get on missing key returns None."""
        cache = MemoryCache()
        path = Path("/tmp/nonexistent.prq")

        result = cache.get(path)

        assert result is None

    def test_put_with_columns_and_filters(self):
        """Put with columns/filters creates composite key."""
        cache = MemoryCache()
        path = Path("/tmp/test.prq")
        df_full = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_filtered = pd.DataFrame({"a": [1]})

        cache.put(path, df_full)
        cache.put(path, df_filtered, columns=["a"], filters=[("a", "=", 1)])

        # Different keys, different data
        result_full = cache.get(path)
        result_filtered = cache.get(path, columns=["a"], filters=[("a", "=", 1)])

        pd.testing.assert_frame_equal(result_full, df_full)
        pd.testing.assert_frame_equal(result_filtered, df_filtered)

    def test_lru_eviction(self):
        """LRU eviction removes oldest entries."""
        config = CacheConfig(max_entries=3)
        cache = MemoryCache(config)

        # Add 3 entries
        for i in range(3):
            cache.put(Path(f"/tmp/{i}.prq"), pd.DataFrame({"x": [i]}))

        # Access first entry to make it most recent
        cache.get(Path("/tmp/0.prq"))

        # Add 4th entry - should evict entry 1 (oldest unused)
        cache.put(Path("/tmp/3.prq"), pd.DataFrame({"x": [3]}))

        assert cache.get(Path("/tmp/0.prq")) is not None  # Kept (accessed)
        assert cache.get(Path("/tmp/1.prq")) is None      # Evicted
        assert cache.get(Path("/tmp/2.prq")) is not None  # Kept
        assert cache.get(Path("/tmp/3.prq")) is not None  # Kept

    def test_invalidate_by_path(self):
        """Invalidate removes all entries for a path."""
        cache = MemoryCache()
        path = Path("/tmp/test.prq")
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Add multiple entries for same path with different columns/filters
        cache.put(path, df)
        cache.put(path, df[["a"]], columns=["a"])
        cache.put(path, df[df["a"] > 1], filters=[("a", ">", 1)])

        count = cache.invalidate(path)

        assert count == 3
        assert cache.get(path) is None
        assert cache.get(path, columns=["a"]) is None
        assert cache.get(path, filters=[("a", ">", 1)]) is None

    def test_clear(self):
        """Clear removes all entries."""
        cache = MemoryCache()

        for i in range(5):
            cache.put(Path(f"/tmp/{i}.prq"), pd.DataFrame({"x": [i]}))

        count = cache.clear()

        assert count == 5
        assert cache.get_stats().current_entries == 0

    def test_disabled_cache(self):
        """Disabled cache returns None and doesn't store."""
        config = CacheConfig(enabled=False)
        cache = MemoryCache(config)
        path = Path("/tmp/test.prq")
        df = pd.DataFrame({"a": [1, 2, 3]})

        cache.put(path, df)
        result = cache.get(path)

        assert result is None
        assert cache.get_stats().current_entries == 0

    def test_stats_tracking(self):
        """Stats correctly track hits, misses, evictions."""
        config = CacheConfig(max_entries=2)
        cache = MemoryCache(config)

        cache.put(Path("/tmp/0.prq"), pd.DataFrame({"x": [0]}))
        cache.put(Path("/tmp/1.prq"), pd.DataFrame({"x": [1]}))

        cache.get(Path("/tmp/0.prq"))  # Hit
        cache.get(Path("/tmp/nonexistent.prq"))  # Miss
        cache.put(Path("/tmp/2.prq"), pd.DataFrame({"x": [2]}))  # Evict

        stats = cache.get_stats()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.evictions == 1
        assert stats.current_entries == 2

    def test_configure_changes_limits(self):
        """Configure method changes cache limits."""
        cache = MemoryCache()

        for i in range(5):
            cache.put(Path(f"/tmp/{i}.prq"), pd.DataFrame({"x": [i]}))

        assert cache.get_stats().current_entries == 5

        # Reduce max entries - should trigger eviction
        cache.configure(max_entries=2)

        assert cache.get_stats().current_entries == 2

    def test_memory_limit_eviction(self):
        """Memory limit triggers eviction of old entries."""
        config = CacheConfig(max_memory_bytes=1000)  # Very small limit
        cache = MemoryCache(config)

        # Add a small dataframe
        df_small = pd.DataFrame({"x": [1]})
        cache.put(Path("/tmp/0.prq"), df_small)

        # Add a larger dataframe that exceeds limit
        df_large = pd.DataFrame({"x": list(range(1000))})
        cache.put(Path("/tmp/1.prq"), df_large)

        # Small one should be evicted
        stats = cache.get_stats()
        assert stats.evictions >= 1


class TestMemoryCacheThreadSafety:
    """Test thread safety of MemoryCache."""

    def test_concurrent_reads_and_writes(self):
        """Concurrent reads and writes don't corrupt cache."""
        cache = MemoryCache()
        errors = []

        def writer(thread_id):
            try:
                for i in range(100):
                    cache.put(
                        Path(f"/tmp/thread{thread_id}_{i}.prq"),
                        pd.DataFrame({"x": [i]}),
                    )
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            try:
                for i in range(100):
                    cache.get(Path(f"/tmp/thread{thread_id}_{i}.prq"))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_invalidation(self):
        """Concurrent invalidation is safe."""
        cache = MemoryCache()
        path = Path("/tmp/test.prq")

        # Pre-populate
        for i in range(10):
            cache.put(path, pd.DataFrame({"x": [i]}), columns=[f"col{i}"])

        errors = []

        def invalidator():
            try:
                for _ in range(50):
                    cache.invalidate(path)
                    cache.put(path, pd.DataFrame({"x": [1]}))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=invalidator) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestModuleLevelFunctions:
    """Test module-level cache functions."""

    def test_get_memory_cache_returns_singleton(self):
        """get_memory_cache returns same instance."""
        cache1 = get_memory_cache()
        cache2 = get_memory_cache()

        assert cache1 is cache2

    def test_configure_memory_cache(self):
        """configure_memory_cache updates singleton."""
        configure_memory_cache(max_entries=50)

        cache = get_memory_cache()
        assert cache._config.max_entries == 50

    def test_clear_memory_cache(self):
        """clear_memory_cache clears singleton."""
        cache = get_memory_cache()
        cache.put(Path("/tmp/test.prq"), pd.DataFrame({"x": [1]}))

        count = clear_memory_cache()

        assert count == 1
        assert cache.get_stats().current_entries == 0

    def test_get_memory_cache_stats(self):
        """get_memory_cache_stats returns singleton stats."""
        clear_memory_cache()
        cache = get_memory_cache()

        cache.put(Path("/tmp/test.prq"), pd.DataFrame({"x": [1]}))
        cache.get(Path("/tmp/test.prq"))
        cache.get(Path("/tmp/nonexistent.prq"))

        stats = get_memory_cache_stats()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.current_entries == 1


class TestFrameMemoryCacheIntegration:
    """Test memory cache integration with Frame.

    Note: These tests directly manipulate the memory cache to verify
    integration, since the disk cache has a pre-existing bug where
    _resolve_chunk_hierarchically uses .prq extension but write_chunk
    uses .prq extension.
    """

    def test_read_chunk_uses_memory_cache(self, cache_dir):
        """read_chunk() integrates with memory cache correctly."""
        from frame.cache import CacheManager
        from frame.backends.pandas import PandasBackend

        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [{"as_of_date": dt.to_pydatetime(), "id": 1, "value": 100}
                       for dt in dates]
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        backend = PandasBackend()
        cache = CacheManager(fetch_data, {}, backend, cache_dir)

        chunk_start = datetime(2024, 1, 1)
        chunk_end = datetime(2024, 1, 31)

        # Write a chunk directly
        df = fetch_data(chunk_start, chunk_end)
        cache.write_chunk(df, chunk_start, chunk_end)

        # First read - memory miss, disk read, populates memory cache
        result1 = cache.read_chunk(chunk_start, chunk_end)
        stats1 = get_memory_cache_stats()

        # Second read - memory hit
        result2 = cache.read_chunk(chunk_start, chunk_end)
        stats2 = get_memory_cache_stats()

        pd.testing.assert_frame_equal(result1, result2)
        assert stats2.hits > stats1.hits

    def test_write_chunk_invalidates_memory_cache(self, cache_dir):
        """write_chunk() invalidates memory cache entries."""
        from frame.cache import CacheManager
        from frame.backends.pandas import PandasBackend

        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [{"as_of_date": dt.to_pydatetime(), "id": 1, "value": 100}
                       for dt in dates]
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        backend = PandasBackend()
        cache = CacheManager(fetch_data, {}, backend, cache_dir)

        chunk_start = datetime(2024, 1, 1)
        chunk_end = datetime(2024, 1, 31)

        # Write and read to populate memory cache
        df = fetch_data(chunk_start, chunk_end)
        cache.write_chunk(df, chunk_start, chunk_end)
        cache.read_chunk(chunk_start, chunk_end)

        stats_before = get_memory_cache_stats()
        assert stats_before.current_entries >= 1

        # Write again - should invalidate
        cache.write_chunk(df, chunk_start, chunk_end)

        stats_after = get_memory_cache_stats()
        assert stats_after.invalidations > stats_before.invalidations

    def test_memory_cache_with_columns_via_read_chunk(self, cache_dir):
        """Memory cache respects column selection in key."""
        from frame.cache import CacheManager
        from frame.backends.pandas import PandasBackend

        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [{"as_of_date": dt.to_pydatetime(), "id": 1, "value": 100, "price": 50.0}
                       for dt in dates]
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        backend = PandasBackend()
        cache = CacheManager(fetch_data, {}, backend, cache_dir)

        chunk_start = datetime(2024, 1, 1)
        chunk_end = datetime(2024, 1, 31)

        # Write a chunk
        df = fetch_data(chunk_start, chunk_end)
        cache.write_chunk(df, chunk_start, chunk_end)

        # Read with column selection
        result1 = cache.read_chunk(chunk_start, chunk_end, columns=["value"])
        result2 = cache.read_chunk(chunk_start, chunk_end, columns=["value"])

        stats = get_memory_cache_stats()
        assert stats.hits >= 1
        assert list(result1.columns) == ["value"]
        pd.testing.assert_frame_equal(result1, result2)

        # Different columns should be a separate cache entry
        result3 = cache.read_chunk(chunk_start, chunk_end, columns=["price"])
        assert list(result3.columns) == ["price"]

    def test_memory_cache_with_filters_via_read_chunk(self, cache_dir):
        """Memory cache respects filters in key."""
        from frame.cache import CacheManager
        from frame.backends.pandas import PandasBackend

        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for i in range(3):
                    records.append({"as_of_date": dt.to_pydatetime(), "id": i, "value": i + 1})
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        backend = PandasBackend()
        cache = CacheManager(fetch_data, {}, backend, cache_dir)

        chunk_start = datetime(2024, 1, 1)
        chunk_end = datetime(2024, 1, 31)

        # Write a chunk
        df = fetch_data(chunk_start, chunk_end)
        cache.write_chunk(df, chunk_start, chunk_end)

        # Read with filter
        result1 = cache.read_chunk(chunk_start, chunk_end, filters=[("value", ">", 1)])
        result2 = cache.read_chunk(chunk_start, chunk_end, filters=[("value", ">", 1)])

        stats = get_memory_cache_stats()
        assert stats.hits >= 1
        pd.testing.assert_frame_equal(result1, result2)

    def test_memory_cache_persists_across_manager_instances(self, cache_dir):
        """Memory cache persists beyond CacheManager instance lifetime."""
        from frame.cache import CacheManager
        from frame.backends.pandas import PandasBackend

        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [{"as_of_date": dt.to_pydatetime(), "id": 1, "value": 100}
                       for dt in dates]
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        backend = PandasBackend()
        chunk_start = datetime(2024, 1, 1)
        chunk_end = datetime(2024, 1, 31)

        # First manager - write and read to populate memory cache
        cache1 = CacheManager(fetch_data, {}, backend, cache_dir)
        df = fetch_data(chunk_start, chunk_end)
        cache1.write_chunk(df, chunk_start, chunk_end)
        result1 = cache1.read_chunk(chunk_start, chunk_end)
        del cache1

        # New manager instance - should hit memory cache
        cache2 = CacheManager(fetch_data, {}, backend, cache_dir)
        result2 = cache2.read_chunk(chunk_start, chunk_end)

        pd.testing.assert_frame_equal(result1, result2)
        stats = get_memory_cache_stats()
        assert stats.hits >= 1

    def test_hit_rate_via_read_chunk(self, cache_dir):
        """Hit rate is calculated correctly via read_chunk."""
        from frame.cache import CacheManager
        from frame.backends.pandas import PandasBackend

        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [{"as_of_date": dt.to_pydatetime(), "id": 1, "value": 100}
                       for dt in dates]
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        backend = PandasBackend()
        cache = CacheManager(fetch_data, {}, backend, cache_dir)

        chunk_start = datetime(2024, 1, 1)
        chunk_end = datetime(2024, 1, 31)

        # Write a chunk
        df = fetch_data(chunk_start, chunk_end)
        cache.write_chunk(df, chunk_start, chunk_end)

        # First call - miss
        cache.read_chunk(chunk_start, chunk_end)

        # Second call - hit
        cache.read_chunk(chunk_start, chunk_end)

        stats = get_memory_cache_stats()
        assert stats.hits >= 1
        assert stats.hit_rate > 0


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_hit_rate_with_no_accesses(self):
        """Hit rate is 0 when no accesses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Hit rate is calculated correctly."""
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 75.0

    def test_hit_rate_100_percent(self):
        """Hit rate can be 100%."""
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 100.0


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_defaults(self):
        """Default values are correct."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.max_entries == 128
        assert config.max_memory_bytes == 0
        assert config.track_stats is True
