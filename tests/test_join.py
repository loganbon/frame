"""Tests for Join operation."""

import shutil
from datetime import datetime

import pandas as pd
import pytest

from frame import AsofJoin, Frame, Join


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


@pytest.fixture
def prices_func():
    """Sample prices data function."""
    def fetch_data(start_dt: datetime, end_dt: datetime):
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for dt in dates:
            for id_ in [1, 2, 3]:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "price": 100.0 + id_ * 10,
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return fetch_data


@pytest.fixture
def volumes_func():
    """Sample volumes data function."""
    def fetch_data(start_dt: datetime, end_dt: datetime):
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for dt in dates:
            for id_ in [1, 2, 3]:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "volume": 1000 * id_,
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return fetch_data


@pytest.fixture
def prices_frame(prices_func, cache_dir):
    """Create a prices Frame for testing."""
    return Frame(prices_func, cache_dir=cache_dir / "prices")


@pytest.fixture
def volumes_frame(volumes_func, cache_dir):
    """Create a volumes Frame for testing."""
    return Frame(volumes_func, cache_dir=cache_dir / "volumes")


class TestJoinBasic:
    """Test basic join functionality."""

    def test_join_default_on_both_keys(self, prices_frame, volumes_frame):
        """Test default join on both as_of_date and id."""
        joined = Join(prices_frame, volumes_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = joined.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert "price" in result.columns
        assert "volume" in result.columns
        # Index should be as_of_date, id
        assert list(result.index.names) == ["as_of_date", "id"]

    def test_join_on_id_only(self, cache_dir):
        """Test join on id only (static metadata use case)."""
        # Prices with time series data
        def fetch_prices(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in [1, 2]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "price": 100.0 + id_,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        # Static metadata (no time dimension in data, just id)
        def fetch_sectors(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [1, 2],
                "sector": ["Tech", "Finance"],
            }).set_index(["as_of_date", "id"])

        prices = Frame(fetch_prices, cache_dir=cache_dir / "prices")
        sectors = Frame(fetch_sectors, cache_dir=cache_dir / "sectors")

        joined = Join(prices, sectors, on="id")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        result = joined.get_range(start, end)

        assert "price" in result.columns
        assert "sector" in result.columns
        # Index should be just id since we joined on id
        assert list(result.index.names) == ["id"]

    def test_join_on_as_of_date_only(self, cache_dir):
        """Test join on as_of_date only (market benchmark use case)."""
        # Individual stock prices
        def fetch_prices(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in [1, 2]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "price": 100.0,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        # Market benchmark (one value per date)
        def fetch_benchmark(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "SPY",
                    "market_return": 0.01,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        prices = Frame(fetch_prices, cache_dir=cache_dir / "prices")
        benchmark = Frame(fetch_benchmark, cache_dir=cache_dir / "benchmark")

        joined = Join(prices, benchmark, on="as_of_date")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        result = joined.get_range(start, end)

        assert "price" in result.columns
        assert "market_return" in result.columns
        assert list(result.index.names) == ["as_of_date"]

    def test_join_custom_columns(self, cache_dir):
        """Test join on custom columns."""
        def fetch_left(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [1, 2],
                "ticker": ["AAPL", "GOOGL"],
                "price": [150.0, 140.0],
            }).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [10, 20],
                "ticker": ["AAPL", "GOOGL"],
                "rating": ["A", "A+"],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        joined = Join(left, right, on=["ticker"])
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        assert "price" in result.columns
        assert "rating" in result.columns
        assert list(result.index.names) == ["ticker"]


class TestJoinTypes:
    """Test different join types."""

    def test_inner_join(self, cache_dir):
        """Test inner join excludes non-matching rows."""
        def fetch_left(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt, start_dt],
                "id": [1, 2, 3],
                "left_val": [10, 20, 30],
            }).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [1, 2],
                "right_val": [100, 200],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        joined = Join(left, right, how="inner")
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        # Only ids 1 and 2 should be present (inner join)
        ids = result.index.get_level_values("id").unique().tolist()
        assert set(ids) == {1, 2}

    def test_left_join(self, cache_dir):
        """Test left join keeps all left rows."""
        def fetch_left(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt, start_dt],
                "id": [1, 2, 3],
                "left_val": [10, 20, 30],
            }).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [1, 2],
                "right_val": [100, 200],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        joined = Join(left, right, how="left")
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        # All ids from left should be present
        ids = result.index.get_level_values("id").unique().tolist()
        assert set(ids) == {1, 2, 3}
        # id=3 should have NaN for right_val
        row_3 = result.loc[(slice(None), 3), "right_val"].iloc[0]
        assert pd.isna(row_3)

    def test_right_join(self, cache_dir):
        """Test right join keeps all right rows."""
        def fetch_left(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [1, 2],
                "left_val": [10, 20],
            }).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt, start_dt],
                "id": [1, 2, 3],
                "right_val": [100, 200, 300],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        joined = Join(left, right, how="right")
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        # All ids from right should be present
        ids = result.index.get_level_values("id").unique().tolist()
        assert set(ids) == {1, 2, 3}
        # id=3 should have NaN for left_val
        row_3 = result.loc[(slice(None), 3), "left_val"].iloc[0]
        assert pd.isna(row_3)

    def test_outer_join(self, cache_dir):
        """Test outer join keeps all rows from both sides."""
        def fetch_left(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [1, 2],
                "left_val": [10, 20],
            }).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [2, 3],
                "right_val": [200, 300],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        joined = Join(left, right, how="outer")
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        # All ids from both sides should be present
        ids = result.index.get_level_values("id").unique().tolist()
        assert set(ids) == {1, 2, 3}


class TestJoinSuffixes:
    """Test suffix handling for overlapping columns."""

    def test_overlapping_columns_default_suffix(self, cache_dir):
        """Test default suffixes for overlapping columns."""
        def fetch_left(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "value": [10],
            }).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "value": [100],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        joined = Join(left, right)
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        # Should have value_left and value_right
        assert "value_left" in result.columns
        assert "value_right" in result.columns

    def test_custom_suffixes(self, cache_dir):
        """Test custom suffixes for overlapping columns."""
        def fetch_left(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "value": [10],
            }).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "value": [100],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        joined = Join(left, right, suffix=("_price", "_vol"))
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        assert "value_price" in result.columns
        assert "value_vol" in result.columns


class TestJoinEdgeCases:
    """Test edge cases for Join operation."""

    def test_invalid_join_type(self, prices_frame, volumes_frame):
        """Test invalid join type raises ValueError."""
        with pytest.raises(ValueError, match="how must be one of"):
            Join(prices_frame, volumes_frame, how="invalid")

    def test_join_repr(self, prices_frame, volumes_frame):
        """Test Join string representation."""
        joined = Join(prices_frame, volumes_frame, on="id", how="inner")
        repr_str = repr(joined)

        assert "Join" in repr_str


class TestJoinPolars:
    """Test Join operation with polars backend."""

    def test_polars_basic_join(self, cache_dir):
        """Test basic join with polars backend."""
        import polars as pl

        def fetch_prices_polars(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            return pl.DataFrame({
                "as_of_date": dates.to_list(),
                "id": [1] * len(dates),
                "price": [100.0] * len(dates),
            })

        def fetch_volumes_polars(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            return pl.DataFrame({
                "as_of_date": dates.to_list(),
                "id": [1] * len(dates),
                "volume": [1000] * len(dates),
            })

        prices = Frame(fetch_prices_polars, backend="polars", cache_dir=cache_dir / "prices")
        volumes = Frame(fetch_volumes_polars, backend="polars", cache_dir=cache_dir / "volumes")

        joined = Join(prices, volumes)
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 3))

        assert isinstance(result, pl.DataFrame)
        assert "price" in result.columns
        assert "volume" in result.columns

    def test_polars_outer_join(self, cache_dir):
        """Test outer join with polars (maps to 'full')."""
        import polars as pl

        def fetch_left(start_dt, end_dt):
            return pl.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [1, 2],
                "left_val": [10, 20],
            })

        def fetch_right(start_dt, end_dt):
            return pl.DataFrame({
                "as_of_date": [start_dt, start_dt],
                "id": [2, 3],
                "right_val": [200, 300],
            })

        left = Frame(fetch_left, backend="polars", cache_dir=cache_dir / "left")
        right = Frame(fetch_right, backend="polars", cache_dir=cache_dir / "right")

        joined = Join(left, right, how="outer")
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        assert isinstance(result, pl.DataFrame)
        # In polars full join, non-matching keys create id_right column
        # We verify both data columns are present and result has correct row count
        assert "left_val" in result.columns
        assert "right_val" in result.columns
        assert len(result) == 3  # id=1 (left only), id=2 (both), id=3 (right only)

    def test_polars_custom_suffix(self, cache_dir):
        """Test custom suffix with polars (only right suffix used)."""
        import polars as pl

        def fetch_left(start_dt, end_dt):
            return pl.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "value": [10],
            })

        def fetch_right(start_dt, end_dt):
            return pl.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "value": [100],
            })

        left = Frame(fetch_left, backend="polars", cache_dir=cache_dir / "left")
        right = Frame(fetch_right, backend="polars", cache_dir=cache_dir / "right")

        joined = Join(left, right, suffix=("_x", "_y"))
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        # Polars applies right suffix to duplicate columns from right frame
        assert "value" in result.columns
        assert "value_y" in result.columns


class TestJoinChaining:
    """Test Join with other operations."""

    def test_join_then_rolling(self, prices_frame, volumes_frame):
        """Test join followed by rolling."""
        joined = Join(prices_frame, volumes_frame)
        result = joined.rolling(window=3).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        assert isinstance(result, pd.DataFrame)
        assert "price" in result.columns
        assert "volume" in result.columns

    def test_multiple_joins(self, cache_dir):
        """Test chaining multiple joins."""
        def fetch_a(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "a_val": [10],
            }).set_index(["as_of_date", "id"])

        def fetch_b(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "b_val": [20],
            }).set_index(["as_of_date", "id"])

        def fetch_c(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": [1],
                "c_val": [30],
            }).set_index(["as_of_date", "id"])

        frame_a = Frame(fetch_a, cache_dir=cache_dir / "a")
        frame_b = Frame(fetch_b, cache_dir=cache_dir / "b")
        frame_c = Frame(fetch_c, cache_dir=cache_dir / "c")

        # Chain: a.join(b).join(c)
        result = Join(Join(frame_a, frame_b), frame_c).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )

        assert "a_val" in result.columns
        assert "b_val" in result.columns
        assert "c_val" in result.columns

    def test_join_with_select(self, prices_frame, volumes_frame):
        """Test join with column selection."""
        joined = Join(prices_frame, volumes_frame)
        result = joined.select(["price"]).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 5)
        )

        assert list(result.columns) == ["price"]


class TestJoinFluentAPI:
    """Test fluent API for join."""

    def test_frame_join_method(self, prices_frame, volumes_frame):
        """Test Frame.join() fluent method."""
        result = prices_frame.join(volumes_frame).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 5)
        )

        assert isinstance(result, pd.DataFrame)
        assert "price" in result.columns
        assert "volume" in result.columns

    def test_join_method_with_params(self, prices_frame, volumes_frame):
        """Test Frame.join() with parameters."""
        result = prices_frame.join(
            volumes_frame, on="id", how="inner", suffix=("_p", "_v")
        ).get_range(datetime(2024, 1, 1), datetime(2024, 1, 5))

        assert isinstance(result, pd.DataFrame)

    def test_join_method_chaining(self, prices_frame, volumes_frame):
        """Test chaining from join result."""
        result = (
            prices_frame
            .join(volumes_frame)
            .rolling(window=2)
            .get_range(datetime(2024, 1, 1), datetime(2024, 1, 10))
        )

        assert isinstance(result, pd.DataFrame)


class TestAsofJoin:
    """Test AsofJoin operation for point-in-time joins."""

    def test_asof_join_basic(self, cache_dir):
        """Test basic as-of join."""
        # Daily price data
        def fetch_prices(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in [1, 2]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "price": 100.0 + id_,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        # Quarterly fundamental data (sparse)
        def fetch_fundamentals(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 15),
                    datetime(2024, 1, 15),
                ],
                "id": [1, 2, 1, 2],
                "earnings": [10.0, 20.0, 12.0, 22.0],
            }).set_index(["as_of_date", "id"])

        prices = Frame(fetch_prices, cache_dir=cache_dir / "prices")
        fundamentals = Frame(fetch_fundamentals, cache_dir=cache_dir / "fundamentals")

        joined = AsofJoin(prices, fundamentals, on="as_of_date", by="id")
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 20))

        assert isinstance(result, pd.DataFrame)
        assert "price" in result.columns
        assert "earnings" in result.columns

    def test_asof_join_uses_prior_data(self, cache_dir):
        """Test that as-of join uses most recent prior data."""
        # Daily prices
        def fetch_prices(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "price": 100.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        # Sparse fundamentals - only on Jan 1 and Jan 15
        def fetch_fundamentals(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [datetime(2024, 1, 1), datetime(2024, 1, 15)],
                "id": [1, 1],
                "earnings": [10.0, 15.0],
            }).set_index(["as_of_date", "id"])

        prices = Frame(fetch_prices, cache_dir=cache_dir / "prices")
        fundamentals = Frame(fetch_fundamentals, cache_dir=cache_dir / "fundamentals")

        joined = AsofJoin(prices, fundamentals, on="as_of_date", by="id")
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 20))

        # On Jan 10, should use Jan 1 earnings (10.0)
        # On Jan 20, should use Jan 15 earnings (15.0)
        result_reset = result.reset_index()
        jan_10 = result_reset[result_reset["as_of_date"] == datetime(2024, 1, 10)]
        jan_20 = result_reset[result_reset["as_of_date"] == datetime(2024, 1, 20)]

        if not jan_10.empty:
            assert jan_10["earnings"].iloc[0] == 10.0
        if not jan_20.empty:
            assert jan_20["earnings"].iloc[0] == 15.0

    def test_asof_join_fluent_api(self, cache_dir):
        """Test Frame.asof_join() fluent method."""
        def fetch_left(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [datetime(2024, 1, 1)],
                "id": [1],
                "other": [50.0],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        result = left.asof_join(right).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 5)
        )

        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns
        assert "other" in result.columns

    def test_asof_join_no_by(self, cache_dir):
        """Test as-of join without by grouping."""
        def fetch_left(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "value": 100.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        def fetch_right(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [datetime(2024, 1, 1), datetime(2024, 1, 10)],
                "id": [99, 99],  # Different ids
                "benchmark": [1.0, 1.5],
            }).set_index(["as_of_date", "id"])

        left = Frame(fetch_left, cache_dir=cache_dir / "left")
        right = Frame(fetch_right, cache_dir=cache_dir / "right")

        # Join without by - should match purely on date
        joined = AsofJoin(left, right, on="as_of_date", by=None)
        result = joined.get_range(datetime(2024, 1, 1), datetime(2024, 1, 15))

        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns
        assert "benchmark" in result.columns

    def test_asof_join_chaining(self, cache_dir):
        """Test chaining operations after as-of join."""
        def fetch_prices(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 1,
                    "price": 100.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        def fetch_fundamentals(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [datetime(2024, 1, 1)],
                "id": [1],
                "earnings": [10.0],
            }).set_index(["as_of_date", "id"])

        prices = Frame(fetch_prices, cache_dir=cache_dir / "prices")
        fundamentals = Frame(fetch_fundamentals, cache_dir=cache_dir / "fundamentals")

        # Chain: asof_join -> rolling
        result = (
            prices
            .asof_join(fundamentals)
            .rolling(window=3)
            .get_range(datetime(2024, 1, 1), datetime(2024, 1, 10))
        )

        assert isinstance(result, pd.DataFrame)
