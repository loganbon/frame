"""Tests for backend implementations."""

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from frame import Frame, PandasBackend, PolarsBackend


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


class TestPandasBackend:
    """Test PandasBackend functionality."""

    def test_concat(self):
        """Test concat operation."""
        backend = PandasBackend()

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        result = backend.concat([df1, df2])

        assert len(result) == 4

    def test_concat_empty(self):
        """Test concat with empty list."""
        backend = PandasBackend()
        result = backend.concat([])
        assert len(result) == 0

    def test_filter_date_range(self):
        """Test date range filtering."""
        backend = PandasBackend()

        df = pd.DataFrame({
            "as_of_date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "id": range(10),
            "value": range(10),
        })
        df = df.set_index(["as_of_date", "id"])

        result = backend.filter_date_range(
            df, datetime(2024, 1, 3), datetime(2024, 1, 5)
        )

        assert len(result) == 3

    def test_drop_index_level(self):
        """Test dropping index level."""
        backend = PandasBackend()

        df = pd.DataFrame({
            "as_of_date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        df = df.set_index(["as_of_date", "id"])

        result = backend.drop_index_level(df, "as_of_date")

        assert "as_of_date" not in result.index.names
        assert result.index.name == "id"

    def test_parquet_roundtrip(self, tmp_path):
        """Test parquet write and read."""
        backend = PandasBackend()

        df = pd.DataFrame({
            "as_of_date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        df = df.set_index(["as_of_date", "id"])

        path = tmp_path / "test.parquet"
        backend.to_parquet(df, path)
        result = backend.read_parquet(path)

        pd.testing.assert_frame_equal(df, result)

    def test_read_parquet_with_columns(self, tmp_path):
        """Test read_parquet with column selection."""
        backend = PandasBackend()

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })

        path = tmp_path / "test.parquet"
        backend.to_parquet(df, path)
        result = backend.read_parquet(path, columns=["a", "c"])

        assert list(result.columns) == ["a", "c"]

    def test_read_parquet_with_filters(self, tmp_path):
        """Test read_parquet with row filtering."""
        backend = PandasBackend()

        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
        })

        path = tmp_path / "test.parquet"
        backend.to_parquet(df, path)
        result = backend.read_parquet(path, filters=[("value", ">", 2)])

        assert len(result) == 3
        assert (result["value"] > 2).all()

    def test_get_date_range(self):
        """Test getting date range from DataFrame."""
        backend = PandasBackend()

        df = pd.DataFrame({
            "as_of_date": pd.date_range("2024-01-05", periods=3, freq="D"),
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        df = df.set_index(["as_of_date", "id"])

        min_date, max_date = backend.get_date_range(df)

        assert min_date.date() == datetime(2024, 1, 5).date()
        assert max_date.date() == datetime(2024, 1, 7).date()


class TestPolarsBackend:
    """Test PolarsBackend functionality."""

    def test_concat(self):
        """Test concat operation."""
        backend = PolarsBackend()

        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})

        result = backend.concat([df1, df2])

        assert len(result) == 4

    def test_concat_empty(self):
        """Test concat with empty list."""
        backend = PolarsBackend()
        result = backend.concat([])
        assert len(result) == 0

    def test_filter_date_range(self):
        """Test date range filtering."""
        backend = PolarsBackend()

        df = pl.DataFrame({
            "as_of_date": pd.date_range("2024-01-01", periods=10, freq="D").tolist(),
            "id": list(range(10)),
            "value": list(range(10)),
        })

        result = backend.filter_date_range(
            df, datetime(2024, 1, 3), datetime(2024, 1, 5)
        )

        assert len(result) == 3

    def test_drop_index_level(self):
        """Test dropping column (Polars equivalent of index level)."""
        backend = PolarsBackend()

        df = pl.DataFrame({
            "as_of_date": pd.date_range("2024-01-01", periods=3, freq="D").tolist(),
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })

        result = backend.drop_index_level(df, "as_of_date")

        assert "as_of_date" not in result.columns

    def test_parquet_roundtrip(self, tmp_path):
        """Test parquet write and read."""
        backend = PolarsBackend()

        df = pl.DataFrame({
            "as_of_date": pd.date_range("2024-01-01", periods=3, freq="D").tolist(),
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })

        path = tmp_path / "test.parquet"
        backend.to_parquet(df, path)
        result = backend.read_parquet(path)

        assert df.equals(result)

    def test_read_parquet_with_columns(self, tmp_path):
        """Test read_parquet with column selection."""
        backend = PolarsBackend()

        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })

        path = tmp_path / "test.parquet"
        backend.to_parquet(df, path)
        result = backend.read_parquet(path, columns=["a", "c"])

        assert result.columns == ["a", "c"]

    def test_read_parquet_with_filters(self, tmp_path):
        """Test read_parquet with row filtering."""
        backend = PolarsBackend()

        df = pl.DataFrame({
            "value": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
        })

        path = tmp_path / "test.parquet"
        backend.to_parquet(df, path)
        result = backend.read_parquet(path, filters=[("value", ">", 2)])

        assert len(result) == 3
        assert (result["value"] > 2).all()

    def test_get_date_range(self):
        """Test getting date range from DataFrame."""
        backend = PolarsBackend()

        df = pl.DataFrame({
            "as_of_date": pd.date_range("2024-01-05", periods=3, freq="D").tolist(),
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })

        min_date, max_date = backend.get_date_range(df)

        assert min_date.date() == datetime(2024, 1, 5).date()
        assert max_date.date() == datetime(2024, 1, 7).date()


class TestPolarsFrame:
    """Test Frame with Polars backend."""

    @pytest.fixture
    def polars_data_func(self):
        """Sample data function returning Polars DataFrame."""
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
            return pl.DataFrame(records)
        return fetch_data

    def test_polars_get_range(self, polars_data_func, cache_dir):
        """Test get_range with Polars backend."""
        frame = Frame(
            polars_data_func,
            {"multiplier": 1},
            backend="polars",
            cache_dir=cache_dir
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = frame.get_range(start, end)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5 * 3

    def test_polars_get_single_date(self, polars_data_func, cache_dir):
        """Test get with Polars backend."""
        frame = Frame(
            polars_data_func,
            {"multiplier": 1},
            backend="polars",
            cache_dir=cache_dir
        )

        dt = datetime(2024, 1, 15)
        result = frame.get(dt)

        assert isinstance(result, pl.DataFrame)
        assert "as_of_date" not in result.columns

    def test_polars_caching(self, polars_data_func, cache_dir):
        """Test caching works with Polars backend."""
        call_count = [0]

        def counting_func(start_dt, end_dt, multiplier=1):
            call_count[0] += 1
            return polars_data_func(start_dt, end_dt, multiplier)

        frame = Frame(
            counting_func,
            {"multiplier": 1},
            backend="polars",
            cache_dir=cache_dir
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        frame.get_range(start, end)
        first_count = call_count[0]

        frame.get_range(start, end)
        second_count = call_count[0]

        assert second_count == first_count
