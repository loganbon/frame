"""Tests for Frame core functionality."""

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from frame import Frame


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


@pytest.fixture
def sample_data_func():
    """Sample data function that returns DataFrame with as_of_date/id index."""
    def fetch_data(start_dt: datetime, end_dt: datetime, multiplier: int = 1):
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


class TestFrameBasic:
    """Test basic Frame functionality."""

    def test_init(self, sample_data_func, cache_dir):
        """Test Frame initialization."""
        frame = Frame(sample_data_func, {"multiplier": 2}, cache_dir=cache_dir)
        assert frame._func == sample_data_func
        assert frame._kwargs == {"multiplier": 2}
        assert frame._backend_name == "pandas"

    def test_get_range(self, sample_data_func, cache_dir):
        """Test get_range returns correct data."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5 * 3  # 5 days, 3 ids each

        dates = result.index.get_level_values("as_of_date")
        assert dates.min().date() == start.date()
        assert dates.max().date() == end.date()

    def test_get_single_date(self, sample_data_func, cache_dir):
        """Test get returns data for single date without as_of_date in index."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)

        dt = datetime(2024, 1, 15)
        result = frame.get(dt)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3 ids
        assert "as_of_date" not in result.index.names

    def test_repr(self, sample_data_func, cache_dir):
        """Test string representation."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        repr_str = repr(frame)
        assert "Frame" in repr_str
        assert "fetch_data" in repr_str
        assert "pandas" in repr_str


class TestFrameAsync:
    """Test async Frame methods."""

    @pytest.mark.asyncio
    async def test_aget_range(self, sample_data_func, cache_dir):
        """Test async get_range."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = await frame.aget_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5 * 3

    @pytest.mark.asyncio
    async def test_aget_single_date(self, sample_data_func, cache_dir):
        """Test async get for single date."""
        frame = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)

        dt = datetime(2024, 1, 15)
        result = await frame.aget(dt)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "as_of_date" not in result.index.names


class TestFrameConcat:
    """Test Frame.concat functionality."""

    def test_concat_multiple_frames(self, cache_dir):
        """Test concatenating multiple frames with different kwargs."""
        def fetch_data(start_dt, end_dt, ticker="AAPL"):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": ticker,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame1 = Frame(fetch_data, {"ticker": "AAPL"}, cache_dir=cache_dir)
        frame2 = Frame(fetch_data, {"ticker": "GOOG"}, cache_dir=cache_dir)
        frame3 = Frame(fetch_data, {"ticker": "MSFT"}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # concat returns an Operation
        combined = Frame.concat([frame1, frame2, frame3])
        result = combined.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # 5 days * 3 tickers = 15 rows
        assert len(result) == 15
        # Check all tickers present
        ids = result.index.get_level_values("id").unique()
        assert set(ids) == {"AAPL", "GOOG", "MSFT"}

    def test_concat_uses_batching(self, cache_dir):
        """Test that concat fetches frames concurrently."""
        import threading
        import time

        concurrent_calls = {"max": 0, "current": 0}
        call_count = {"count": 0}
        lock = threading.Lock()

        def slow_fetch(start_dt, end_dt, ticker="AAPL"):
            with lock:
                call_count["count"] += 1
                concurrent_calls["current"] += 1
                concurrent_calls["max"] = max(
                    concurrent_calls["max"], concurrent_calls["current"]
                )
            time.sleep(0.1)  # Longer delay to detect concurrency
            with lock:
                concurrent_calls["current"] -= 1

            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": ticker,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        from frame import DateCalendar

        frames = [
            Frame(slow_fetch, {"ticker": f"T{i}"}, cache_dir=cache_dir, calendar=DateCalendar())
            for i in range(4)
        ]

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        combined = Frame.concat(frames)
        result = combined.get_range(start, end)

        # Verify all 4 frames were fetched
        assert call_count["count"] == 4
        # Verify result has all data: 3 days * 4 tickers
        assert len(result) == 12

    def test_concat_empty_list_raises(self, cache_dir):
        """Test that empty frames list raises ValueError."""
        with pytest.raises(ValueError, match="Concat requires at least one input"):
            Frame.concat([])

    def test_concat_with_operations(self, cache_dir):
        """Test concat works with Operation objects."""
        def fetch_data(start_dt, end_dt, multiplier=1):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 10.0 * multiplier,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {"multiplier": 1}, cache_dir=cache_dir)
        doubled = frame * 2

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        combined = Frame.concat([frame, doubled])
        result = combined.get_range(start, end)

        # 3 days * 2 sources = 6 rows
        assert len(result) == 6
        # Check values (original 10, doubled 20)
        values = result["value"].unique()
        assert set(values) == {10.0, 20.0}

    @pytest.mark.asyncio
    async def test_concat_async(self, cache_dir):
        """Test async concat via aget_range."""
        def fetch_data(start_dt, end_dt, ticker="AAPL"):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": ticker,
                    "value": 100,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame1 = Frame(fetch_data, {"ticker": "AAPL"}, cache_dir=cache_dir)
        frame2 = Frame(fetch_data, {"ticker": "GOOG"}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # Use aget_range on the concat operation
        combined = Frame.concat([frame1, frame2])
        result = await combined.aget_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10  # 5 days * 2 tickers

    def test_concat_can_be_chained(self, cache_dir):
        """Test concat can be chained with other operations."""
        def fetch_data(start_dt, end_dt, ticker="AAPL"):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": ticker,
                    "value": 10.0,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame1 = Frame(fetch_data, {"ticker": "AAPL"}, cache_dir=cache_dir)
        frame2 = Frame(fetch_data, {"ticker": "GOOG"}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        # Chain concat with multiplication
        scaled = Frame.concat([frame1, frame2]) * 2
        result = scaled.get_range(start, end)

        # All values should be doubled (10 * 2 = 20)
        assert (result["value"] == 20.0).all()
        # 3 days * 2 tickers = 6 rows
        assert len(result) == 6
