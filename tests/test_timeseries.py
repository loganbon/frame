"""Tests for time series operations."""

import shutil
from datetime import datetime

import pandas as pd
import pytest

from frame import (
    Cummax,
    Cummin,
    Cumprod,
    Cumsum,
    Ewm,
    Expanding,
    Frame,
    Resample,
)


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
    def fetch_data(start_dt: datetime, end_dt: datetime):
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for i, dt in enumerate(dates):
            for id_ in range(3):
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "value": float(i + 1 + id_),  # 1, 2, 3, 4... varies over time
                    "price": 100.0 + i * 10,
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return fetch_data


@pytest.fixture
def prices_frame(sample_data_func, cache_dir):
    """Create a prices Frame for testing."""
    return Frame(sample_data_func, cache_dir=cache_dir)


class TestEwm:
    """Test Ewm (exponentially weighted moving) operation."""

    def test_ewm_mean_basic(self, prices_frame):
        """Test basic EWM mean."""
        ewm = Ewm(prices_frame, span=3, func="mean")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = ewm.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns

    def test_ewm_std(self, prices_frame):
        """Test EWM standard deviation."""
        ewm = Ewm(prices_frame, span=3, func="std")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = ewm.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_ewm_var(self, prices_frame):
        """Test EWM variance."""
        ewm = Ewm(prices_frame, span=3, func="var")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = ewm.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_ewm_invalid_func(self, prices_frame):
        """Test that invalid func raises error."""
        with pytest.raises(ValueError, match="func must be"):
            Ewm(prices_frame, span=3, func="invalid")

    def test_ewm_fluent_api(self, prices_frame):
        """Test Frame.ewm() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.ewm(span=3).get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestExpanding:
    """Test Expanding operation."""

    def test_expanding_mean(self, prices_frame):
        """Test expanding mean."""
        expanding = Expanding(prices_frame, func="mean")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = expanding.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_expanding_sum(self, prices_frame):
        """Test expanding sum."""
        expanding = Expanding(prices_frame, func="sum")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = expanding.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_expanding_min_periods(self, prices_frame):
        """Test expanding with min_periods."""
        expanding = Expanding(prices_frame, func="mean", min_periods=3)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = expanding.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # First 2 rows per group should be NaN with min_periods=3
        assert result["value"].isna().any()

    def test_expanding_fluent_api(self, prices_frame):
        """Test Frame.expanding() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.expanding(func="sum").get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestCumsum:
    """Test Cumsum operation."""

    def test_cumsum_basic(self, prices_frame):
        """Test basic cumulative sum."""
        cumsum = Cumsum(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = cumsum.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Values should be increasing for each id
        for id_ in range(3):
            id_values = result.loc[(slice(None), id_), "value"].values
            assert all(id_values[i] <= id_values[i + 1] for i in range(len(id_values) - 1))

    def test_cumsum_fluent_api(self, prices_frame):
        """Test Frame.cumsum() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cumsum().get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestCumprod:
    """Test Cumprod operation."""

    def test_cumprod_basic(self, cache_dir):
        """Test basic cumulative product."""
        # Use small positive values to avoid overflow
        def fetch_small_values(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in range(2):
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": 1.1,  # Small multiplier
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_small_values, cache_dir=cache_dir)
        cumprod = Cumprod(frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = cumprod.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Values should be increasing (since all multipliers > 1)
        for id_ in range(2):
            id_values = result.loc[(slice(None), id_), "value"].values
            assert all(id_values[i] <= id_values[i + 1] for i in range(len(id_values) - 1))

    def test_cumprod_fluent_api(self, prices_frame):
        """Test Frame.cumprod() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cumprod().get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestCummax:
    """Test Cummax operation."""

    def test_cummax_basic(self, prices_frame):
        """Test basic cumulative maximum."""
        cummax = Cummax(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = cummax.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Cumulative max should be non-decreasing
        for id_ in range(3):
            id_values = result.loc[(slice(None), id_), "value"].values
            assert all(id_values[i] <= id_values[i + 1] for i in range(len(id_values) - 1))

    def test_cummax_fluent_api(self, prices_frame):
        """Test Frame.cummax() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.cummax().get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestCummin:
    """Test Cummin operation."""

    def test_cummin_basic(self, cache_dir):
        """Test basic cumulative minimum."""
        # Use decreasing values
        def fetch_decreasing(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for i, dt in enumerate(dates):
                for id_ in range(2):
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": 100.0 - i,  # Decreasing
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_decreasing, cache_dir=cache_dir)
        cummin = Cummin(frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = cummin.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Cumulative min should be non-increasing (since values decrease)
        for id_ in range(2):
            id_values = result.loc[(slice(None), id_), "value"].values
            assert all(id_values[i] >= id_values[i + 1] for i in range(len(id_values) - 1))

    def test_cummin_fluent_api(self, prices_frame):
        """Test Frame.cummin() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.cummin().get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestResample:
    """Test Resample operation."""

    def test_resample_daily_to_weekly(self, prices_frame):
        """Test resampling from daily to weekly."""
        resampled = Resample(prices_frame, freq="W", func="last")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        result = resampled.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Should have fewer rows than original
        original = prices_frame.get_range(start, end)
        assert len(result) < len(original)

    def test_resample_daily_to_monthly(self, prices_frame):
        """Test resampling from daily to monthly."""
        resampled = Resample(prices_frame, freq="M", func="mean")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)

        result = resampled.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_resample_func_sum(self, prices_frame):
        """Test resampling with sum aggregation."""
        resampled = Resample(prices_frame, freq="W", func="sum")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        result = resampled.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_resample_func_first(self, prices_frame):
        """Test resampling with first aggregation."""
        resampled = Resample(prices_frame, freq="W", func="first")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        result = resampled.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_resample_fluent_api(self, prices_frame):
        """Test Frame.resample() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        result = prices_frame.resample(freq="W").get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestTimeseriesChaining:
    """Test chaining time series operations."""

    def test_cumsum_then_diff(self, prices_frame):
        """Test chaining cumsum and diff (should recover original-ish)."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.cumsum().diff().get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_ewm_then_rolling(self, prices_frame):
        """Test chaining EWM and rolling."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        result = prices_frame.ewm(span=3).rolling(window=3).get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_expanding_then_zscore(self, prices_frame):
        """Test chaining expanding and zscore."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        result = prices_frame.expanding(func="mean").zscore(window=5).get_range(start, end)

        assert isinstance(result, pd.DataFrame)
