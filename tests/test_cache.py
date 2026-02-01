"""Tests for caching functionality."""

import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from frame import Frame
from frame.cache import CacheManager, CacheMissError
from frame.backends.pandas import PandasBackend


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


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_compute_cache_key(self, sample_data_func, cache_dir):
        """Test cache key generation is consistent."""
        backend = PandasBackend()

        cache1 = CacheManager(
            sample_data_func, {"multiplier": 2}, backend, cache_dir
        )
        cache2 = CacheManager(
            sample_data_func, {"multiplier": 2}, backend, cache_dir
        )
        cache3 = CacheManager(
            sample_data_func, {"multiplier": 3}, backend, cache_dir
        )

        assert cache1._cache_key == cache2._cache_key
        assert cache1._cache_key != cache3._cache_key

    def test_get_chunk_ranges_month(self, sample_data_func, cache_dir):
        """Test date range chunking with month granularity."""
        backend = PandasBackend()
        cache = CacheManager(
            sample_data_func, {}, backend, cache_dir, chunk_granularity="month"
        )

        start = datetime(2024, 1, 15)
        end = datetime(2024, 3, 15)

        chunks = cache.get_chunk_ranges(start, end)

        assert len(chunks) == 3
        # First chunk should be Jan 1 - Jan 31
        assert chunks[0][0] == datetime(2024, 1, 1)
        assert chunks[0][1].date() == datetime(2024, 1, 31).date()
        # Second chunk should be Feb 1 - Feb 29 (2024 is leap year)
        assert chunks[1][0] == datetime(2024, 2, 1)
        assert chunks[1][1].date() == datetime(2024, 2, 29).date()

    def test_get_chunk_ranges_day(self, sample_data_func, cache_dir):
        """Test date range chunking with day granularity."""
        backend = PandasBackend()
        cache = CacheManager(
            sample_data_func, {}, backend, cache_dir, chunk_granularity="day"
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        chunks = cache.get_chunk_ranges(start, end)

        assert len(chunks) == 3
        assert chunks[0][0] == datetime(2024, 1, 1)
        assert chunks[0][1].date() == datetime(2024, 1, 1).date()
        assert chunks[1][0] == datetime(2024, 1, 2)
        assert chunks[2][0] == datetime(2024, 1, 3)


class TestFrameCaching:
    """Test Frame caching behavior."""

    def test_cache_hit(self, sample_data_func, call_counter, cache_dir):
        """Test that second call uses cache."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        result1 = frame.get_range(start, end)
        first_call_count = call_counter["count"]

        result2 = frame.get_range(start, end)
        second_call_count = call_counter["count"]

        pd.testing.assert_frame_equal(result1, result2)
        assert second_call_count == first_call_count

    def test_cache_files_created(self, sample_data_func, cache_dir):
        """Test that parquet files are created in cache directory."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        frame.get_range(start, end)

        assert cache_dir.exists()
        parquet_files = list(cache_dir.rglob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_partial_cache(self, sample_data_func, call_counter, cache_dir):
        """Test that only uncached chunks are fetched."""
        frame = Frame(
            sample_data_func,
            {"multiplier": 1},
            cache_dir=cache_dir,
            chunk_granularity="day"
        )

        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 5))
        calls_after_first = call_counter["count"]

        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 10))
        calls_after_second = call_counter["count"]

        # Should have fetched only the missing days (Jan 6-10)
        assert calls_after_second > calls_after_first

    def test_different_kwargs_different_cache(self, sample_data_func, call_counter, cache_dir):
        """Test that different kwargs use different caches."""
        frame1 = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        frame2 = Frame(sample_data_func, {"multiplier": 2}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        result1 = frame1.get_range(start, end)
        result2 = frame2.get_range(start, end)

        assert not result1["value"].equals(result2["value"])
        assert call_counter["count"] == 2


class TestColumnSelection:
    """Test column selection functionality."""

    def test_column_selection_get_range(self, sample_data_func, cache_dir):
        """Test get_range with column selection."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(start, end, columns=["value"])

        assert list(result.columns) == ["value"]

    def test_column_selection_get(self, sample_data_func, cache_dir):
        """Test get with column selection."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        dt = datetime(2024, 1, 5)

        result = frame.get(dt, columns=["value"])

        assert list(result.columns) == ["value"]

    def test_multiple_columns(self, sample_data_func, cache_dir):
        """Test selecting multiple columns."""
        # Add price column to the data func
        def fetch_with_price(start_dt, end_dt, multiplier=1):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in range(3):
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": (id_ + 1) * multiplier,
                        "price": 100.0 + id_ * 10,
                    })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(fetch_with_price, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(start, end, columns=["value", "price"])

        assert list(result.columns) == ["value", "price"]


class TestRowFiltering:
    """Test row filtering functionality."""

    def test_filter_greater_than(self, sample_data_func, cache_dir):
        """Test filtering with > operator."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(start, end, filters=[("value", ">", 1)])

        assert (result["value"] > 1).all()

    def test_filter_equal(self, sample_data_func, cache_dir):
        """Test filtering with = operator."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(start, end, filters=[("value", "=", 2)])

        assert (result["value"] == 2).all()

    def test_filter_in(self, sample_data_func, cache_dir):
        """Test filtering with 'in' operator."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(start, end, filters=[("value", "in", [1, 3])])

        assert result["value"].isin([1, 3]).all()

    def test_multiple_filters(self, sample_data_func, cache_dir):
        """Test multiple filter conditions (AND)."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(
            start, end, filters=[("value", ">", 1), ("value", "<", 3)]
        )

        assert (result["value"] == 2).all()


class TestCacheFullDataAfterFilter:
    """Test that cache stores full data after filtered read."""

    def test_cache_full_data_after_filter(self, sample_data_func, call_counter, cache_dir):
        """Test cache contains full data after a filtered read."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # First read with filter
        result1 = frame.get_range(
            start, end, columns=["value"], filters=[("value", ">", 1)]
        )
        first_call_count = call_counter["count"]

        # Second read without filter should have all data (from cache)
        result2 = frame.get_range(start, end)
        second_call_count = call_counter["count"]

        # Full data has more rows than filtered
        assert len(result2) > len(result1)
        # Cache was used (no new fetch)
        assert second_call_count == first_call_count

    def test_different_filters_same_cache(self, sample_data_func, call_counter, cache_dir):
        """Test different filters on same data use the same cache."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # First read with one filter
        result1 = frame.get_range(start, end, filters=[("value", ">", 1)])
        first_call_count = call_counter["count"]

        # Second read with different filter
        result2 = frame.get_range(start, end, filters=[("value", "=", 1)])
        second_call_count = call_counter["count"]

        # Different filter results
        assert len(result1) != len(result2)
        # Cache was used (no new fetch)
        assert second_call_count == first_call_count


class TestCacheMode:
    """Test cache mode functionality."""

    def test_live_mode_bypasses_cache(self, sample_data_func, call_counter, cache_dir):
        """Test that live mode always fetches fresh data."""
        frame = Frame(sample_data_func, cache_dir=cache_dir)
        start, end = datetime(2024, 1, 1), datetime(2024, 1, 5)

        # First call caches data
        frame.get_range(start, end)
        assert call_counter["count"] == 1

        # Live mode should call function again
        frame.get_range(start, end, cache_mode="l")
        assert call_counter["count"] == 2

        # Another live call should also fetch
        frame.get_range(start, end, cache_mode="l")
        assert call_counter["count"] == 3

    def test_read_mode_raises_if_not_cached(self, sample_data_func, cache_dir):
        """Test that read mode raises CacheMissError if data not cached."""
        frame = Frame(sample_data_func, cache_dir=cache_dir)
        start, end = datetime(2024, 1, 1), datetime(2024, 1, 5)

        with pytest.raises(CacheMissError):
            frame.get_range(start, end, cache_mode="r")

    def test_read_mode_returns_cached_data(self, sample_data_func, call_counter, cache_dir):
        """Test that read mode returns cached data without fetching."""
        frame = Frame(sample_data_func, cache_dir=cache_dir)
        start, end = datetime(2024, 1, 1), datetime(2024, 1, 5)

        # First, cache the data
        result1 = frame.get_range(start, end)
        assert call_counter["count"] == 1

        # Read mode should return cached data without fetching
        result2 = frame.get_range(start, end, cache_mode="r")
        assert call_counter["count"] == 1  # No additional fetch
        assert len(result2) == len(result1)

    def test_write_mode_overwrites_cache(self, sample_data_func, call_counter, cache_dir):
        """Test that write mode always fetches and overwrites cache."""
        frame = Frame(sample_data_func, cache_dir=cache_dir)
        start, end = datetime(2024, 1, 1), datetime(2024, 1, 5)

        # First call caches
        frame.get_range(start, end)
        assert call_counter["count"] == 1

        # Write mode forces re-fetch
        frame.get_range(start, end, cache_mode="w")
        assert call_counter["count"] == 2

    def test_append_mode_is_default(self, sample_data_func, call_counter, cache_dir):
        """Test that append mode is the default behavior."""
        frame = Frame(sample_data_func, cache_dir=cache_dir)
        start, end = datetime(2024, 1, 1), datetime(2024, 1, 5)

        # First call should fetch
        frame.get_range(start, end)
        assert call_counter["count"] == 1

        # Second call with explicit append mode should use cache
        frame.get_range(start, end, cache_mode="a")
        assert call_counter["count"] == 1

        # Third call with default should also use cache
        frame.get_range(start, end)
        assert call_counter["count"] == 1


class TestHierarchicalCache:
    """Test hierarchical cache functionality."""

    def test_reads_from_parent_cache(self, tmp_path):
        """Test that data is read from parent cache when not in primary."""
        parent_cache = tmp_path / "parent"
        primary_cache = tmp_path / "primary"

        # Create a call counter to track function invocations
        call_counter = {"count": 0}

        def shared_func(start_dt, end_dt, multiplier=1):
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

        # First, populate parent cache
        parent_frame = Frame(shared_func, cache_dir=parent_cache)
        parent_frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 5))
        assert call_counter["count"] == 1  # Fetched once

        # Create frame with same function but different cache directory
        frame = Frame(
            shared_func,
            cache_dir=primary_cache,
            parent_cache_dirs=[parent_cache],
        )

        # Should read from parent, not call function
        result = frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 5))
        assert call_counter["count"] == 1  # No additional calls
        assert len(result) > 0

    def test_writes_to_primary_only(self, sample_data_func, tmp_path):
        """Test that new data is written to primary cache only."""
        parent_cache = tmp_path / "parent"
        primary_cache = tmp_path / "primary"
        parent_cache.mkdir()

        frame = Frame(
            sample_data_func,
            cache_dir=primary_cache,
            parent_cache_dirs=[parent_cache],
        )

        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 5))

        # Data should be in primary, not parent
        assert len(list(primary_cache.rglob("*.parquet"))) > 0
        assert len(list(parent_cache.rglob("*.parquet"))) == 0

    def test_mixed_source_resolution(self, tmp_path):
        """Test that different chunks can come from different cache levels."""
        parent_cache = tmp_path / "parent"
        primary_cache = tmp_path / "primary"

        # Create a call counter to track function invocations
        call_counter = {"count": 0}

        def shared_func(start_dt, end_dt, multiplier=1):
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

        # Populate parent cache with January only
        parent_frame = Frame(
            shared_func, cache_dir=parent_cache, chunk_granularity="month"
        )
        parent_frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 31))
        assert call_counter["count"] == 1  # Fetched January

        # Create frame with same function and both caches
        frame = Frame(
            shared_func,
            cache_dir=primary_cache,
            parent_cache_dirs=[parent_cache],
            chunk_granularity="month",
        )

        # Request range spanning 2 months
        # January: should come from parent cache
        # February: should be fetched live
        result = frame.get_range(datetime(2024, 1, 15), datetime(2024, 2, 15))

        # Only 1 additional call (for February chunk)
        assert call_counter["count"] == 2
        # But we got data for the full range
        assert len(result) > 0

    def test_partial_chunk_resolution(self, tmp_path):
        """Test that partial data in parent cache is augmented with live data."""
        parent_cache = tmp_path / "parent"
        primary_cache = tmp_path / "primary"

        # Track function invocations
        call_counter = {"dates_requested": []}

        def tracking_func(start_dt, end_dt, multiplier=1):
            call_counter["dates_requested"].append((start_dt.date(), end_dt.date()))
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

        # Populate parent cache with only first half of January
        parent_frame = Frame(
            tracking_func, cache_dir=parent_cache, chunk_granularity="month"
        )
        parent_frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 15))
        assert len(call_counter["dates_requested"]) == 1  # Fetched once

        # Reset counter for child frame
        call_counter["dates_requested"] = []

        # Create frame with same function and both caches
        frame = Frame(
            tracking_func,
            cache_dir=primary_cache,
            parent_cache_dirs=[parent_cache],
            chunk_granularity="month",
        )

        # Request full January - should get Jan 1-15 from parent, Jan 16-31 live
        result = frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 31))

        # Should have fetched only the missing dates
        assert len(call_counter["dates_requested"]) == 1
        start_fetched, _ = call_counter["dates_requested"][0]
        assert start_fetched >= datetime(2024, 1, 16).date()

        # Result should contain all dates
        assert len(result) > 0


class TestChunkGranularity:
    """Test different chunk granularity options."""

    def test_day_granularity(self, sample_data_func, cache_dir):
        """Test day-level chunking."""
        frame = Frame(
            sample_data_func,
            cache_dir=cache_dir,
            chunk_granularity="day",
        )

        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 3))

        # Should create 3 parquet files, one per day
        parquet_files = list(cache_dir.rglob("*.parquet"))
        assert len(parquet_files) == 3

    def test_week_granularity(self, sample_data_func, cache_dir):
        """Test week-level chunking."""
        frame = Frame(
            sample_data_func,
            cache_dir=cache_dir,
            chunk_granularity="week",
        )

        # Request 2 weeks of data
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 14))

        parquet_files = list(cache_dir.rglob("*.parquet"))
        # Should create 2-3 parquet files (depends on week boundaries)
        assert len(parquet_files) >= 2

    def test_month_granularity(self, sample_data_func, cache_dir):
        """Test month-level chunking (default)."""
        frame = Frame(
            sample_data_func,
            cache_dir=cache_dir,
            chunk_granularity="month",
        )

        frame.get_range(datetime(2024, 1, 15), datetime(2024, 3, 15))

        # Should create 3 parquet files, one per month
        parquet_files = list(cache_dir.rglob("*.parquet"))
        assert len(parquet_files) == 3

    def test_year_granularity(self, sample_data_func, cache_dir):
        """Test year-level chunking."""
        frame = Frame(
            sample_data_func,
            cache_dir=cache_dir,
            chunk_granularity="year",
        )

        frame.get_range(datetime(2024, 6, 1), datetime(2025, 6, 1))

        # Should create 2 parquet files, one per year
        parquet_files = list(cache_dir.rglob("*.parquet"))
        assert len(parquet_files) == 2


class TestConcurrentChunkOperations:
    """Test concurrent chunk reading and fetching."""

    def test_concurrent_read_multiple_chunks(self, tmp_path):
        """Reading multiple cached chunks uses concurrency."""
        cache_dir = tmp_path / "cache"

        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
        )

        # First request: populate cache for 3 months
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 3, 31))

        # Verify cache files created
        parquet_files = list(cache_dir.rglob("*.parquet"))
        assert len(parquet_files) == 3

        # Second request: read from cache (uses concurrent reads)
        result = frame.get_range(datetime(2024, 1, 1), datetime(2024, 3, 31))

        # Verify data returned correctly
        assert len(result) > 0

    def test_concurrent_read_partial_cache(self, tmp_path):
        """Mixed cached/uncached chunks work correctly."""
        cache_dir = tmp_path / "cache"
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
        )

        # Cache only January
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 31))
        initial_calls = call_counter["count"]

        # Request January through March (Jan cached, Feb/Mar need fetch)
        result = frame.get_range(datetime(2024, 1, 1), datetime(2024, 3, 31))

        # Should have made additional calls for Feb and Mar
        assert call_counter["count"] > initial_calls
        assert len(result) > 0

    def test_concurrent_read_preserves_order(self, tmp_path):
        """Data returned in correct chronological order."""
        cache_dir = tmp_path / "cache"

        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": dt.day,  # Use day as value for order verification
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
        )

        # Request 3 months of data
        result = frame.get_range(datetime(2024, 1, 1), datetime(2024, 3, 31))

        # Result should be sortable by date and sorted
        dates = result.index.get_level_values("as_of_date")
        assert list(dates) == sorted(dates)

    def test_read_workers_limits_read_concurrency(self, tmp_path):
        """read_workers parameter limits parallel cache reads."""
        cache_dir = tmp_path / "cache"
        concurrent_reads = {"max": 0, "current": 0}
        lock = threading.Lock()

        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
            read_workers=2,  # Limit to 2 concurrent reads
        )

        # Populate cache
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 6, 30))

        # Verify cache populated
        parquet_files = list(cache_dir.rglob("*.parquet"))
        assert len(parquet_files) == 6

    def test_fetch_workers_limits_fetch_concurrency(self, tmp_path):
        """fetch_workers parameter limits parallel live fetches."""
        cache_dir = tmp_path / "cache"
        concurrent_fetches = {"max": 0, "current": 0}
        lock = threading.Lock()

        def fetch_data(start_dt, end_dt):
            with lock:
                concurrent_fetches["current"] += 1
                concurrent_fetches["max"] = max(
                    concurrent_fetches["max"], concurrent_fetches["current"]
                )
            time.sleep(0.05)  # Small delay to allow concurrency detection
            with lock:
                concurrent_fetches["current"] -= 1

            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
            fetch_workers=2,  # Limit to 2 concurrent fetches
        )

        # Request 4 months (all uncached, should trigger concurrent fetches)
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 4, 30))

        # Max concurrent should be limited to 2
        assert concurrent_fetches["max"] <= 2

    def test_fetch_workers_one_is_sequential(self, tmp_path):
        """fetch_workers=1 (default) processes fetches sequentially."""
        cache_dir = tmp_path / "cache"
        concurrent_fetches = {"max": 0, "current": 0}
        lock = threading.Lock()

        def fetch_data(start_dt, end_dt):
            with lock:
                concurrent_fetches["current"] += 1
                concurrent_fetches["max"] = max(
                    concurrent_fetches["max"], concurrent_fetches["current"]
                )
            time.sleep(0.05)
            with lock:
                concurrent_fetches["current"] -= 1

            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
            fetch_workers=1,  # Sequential (default)
        )

        # Request multiple months
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 4, 30))

        # Should never exceed 1 concurrent fetch
        assert concurrent_fetches["max"] == 1

    def test_single_chunk_skips_threading(self, tmp_path):
        """Single chunk request doesn't use threading overhead."""
        cache_dir = tmp_path / "cache"
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
        )

        # Request within single month
        result = frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 15))

        # Should have made exactly 1 call and returned data
        assert call_counter["count"] == 1
        assert len(result) > 0

    def test_default_fetch_workers_is_one(self, tmp_path):
        """Default fetch_workers is 1 (sequential, safe for APIs)."""
        cache_dir = tmp_path / "cache"

        def fetch_data(start_dt, end_dt):
            return pd.DataFrame()

        frame = Frame(fetch_data, cache_dir=cache_dir)

        assert frame._fetch_workers == 1

    def test_default_read_workers_is_none(self, tmp_path):
        """Default read_workers is None (system default)."""
        cache_dir = tmp_path / "cache"

        def fetch_data(start_dt, end_dt):
            return pd.DataFrame()

        frame = Frame(fetch_data, cache_dir=cache_dir)

        assert frame._read_workers is None

    def test_concurrent_with_high_fetch_workers(self, tmp_path):
        """High fetch_workers allows more parallel fetches."""
        cache_dir = tmp_path / "cache"
        concurrent_fetches = {"max": 0, "current": 0}
        lock = threading.Lock()

        def fetch_data(start_dt, end_dt):
            with lock:
                concurrent_fetches["current"] += 1
                concurrent_fetches["max"] = max(
                    concurrent_fetches["max"], concurrent_fetches["current"]
                )
            time.sleep(0.05)
            with lock:
                concurrent_fetches["current"] -= 1

            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
            fetch_workers=4,  # Allow 4 concurrent fetches
        )

        # Request 4 months (all uncached)
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 4, 30))

        # Should allow up to 4 concurrent fetches
        assert concurrent_fetches["max"] <= 4
        assert concurrent_fetches["max"] >= 2  # Should use some concurrency

    def test_concurrent_cache_miss_error_aggregates_missing(self, tmp_path):
        """cache_mode='r' with concurrent reads aggregates all missing dates."""
        cache_dir = tmp_path / "cache"

        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(
            fetch_data,
            cache_dir=cache_dir,
            chunk_granularity="month",
        )

        # Request multiple uncached months with read-only mode
        with pytest.raises(CacheMissError):
            frame.get_range(
                datetime(2024, 1, 1), datetime(2024, 3, 31), cache_mode="r"
            )
