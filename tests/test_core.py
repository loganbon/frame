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
