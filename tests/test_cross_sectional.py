"""Tests for cross-sectional operations."""

import shutil
from datetime import datetime

import pandas as pd
import pytest

from frame import (
    CsDemean,
    CsRank,
    CsWinsorize,
    CsZscore,
    Frame,
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
    """Sample data function with multiple ids per date for cross-sectional ops."""
    def fetch_data(start_dt: datetime, end_dt: datetime):
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for dt in dates:
            for id_ in range(5):
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "value": (id_ + 1) * 10.0,  # 10, 20, 30, 40, 50
                    "price": 100.0 + id_ * 5,   # 100, 105, 110, 115, 120
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return fetch_data


@pytest.fixture
def prices_frame(sample_data_func, cache_dir):
    """Create a prices Frame for testing."""
    return Frame(sample_data_func, cache_dir=cache_dir)


class TestCsRank:
    """Test CsRank operation."""

    def test_cs_rank_basic(self, prices_frame):
        """Test basic cross-sectional rank."""
        ranked = CsRank(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = ranked.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Ranks should be between 0 and 1
        assert (result["value"] >= 0).all()
        assert (result["value"] <= 1).all()

    def test_cs_rank_ordering(self, prices_frame):
        """Test that ranks preserve ordering within each date."""
        ranked = CsRank(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1)

        result = ranked.get_range(start, end)

        # Higher values should have higher ranks
        # Original: id 0=10, id 1=20, id 2=30, id 3=40, id 4=50
        # So rank should be: id 0 < id 1 < id 2 < id 3 < id 4
        ranks = result["value"].values
        assert ranks[0] < ranks[1] < ranks[2] < ranks[3] < ranks[4]

    def test_cs_rank_fluent_api(self, prices_frame):
        """Test Frame.cs_rank() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cs_rank().get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert (result["value"] >= 0).all()
        assert (result["value"] <= 1).all()


class TestCsZscore:
    """Test CsZscore operation."""

    def test_cs_zscore_basic(self, prices_frame):
        """Test basic cross-sectional z-score."""
        zscored = CsZscore(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = zscored.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_cs_zscore_zero_mean(self, prices_frame):
        """Test that cross-sectional z-score has zero mean per date."""
        zscored = CsZscore(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = zscored.get_range(start, end)

        # Mean should be approximately 0 for each date
        means = result.groupby(level="as_of_date")["value"].mean()
        assert (means.abs() < 1e-10).all()

    def test_cs_zscore_unit_std(self, prices_frame):
        """Test that cross-sectional z-score has unit std per date."""
        zscored = CsZscore(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = zscored.get_range(start, end)

        # Std should be approximately 1 for each date
        stds = result.groupby(level="as_of_date")["value"].std()
        assert ((stds - 1.0).abs() < 1e-10).all()

    def test_cs_zscore_fluent_api(self, prices_frame):
        """Test Frame.cs_zscore() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cs_zscore().get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestCsDemean:
    """Test CsDemean operation."""

    def test_cs_demean_basic(self, prices_frame):
        """Test basic cross-sectional demean."""
        demeaned = CsDemean(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = demeaned.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_cs_demean_zero_mean(self, prices_frame):
        """Test that cross-sectional demean has zero mean per date."""
        demeaned = CsDemean(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = demeaned.get_range(start, end)

        # Mean should be approximately 0 for each date
        means = result.groupby(level="as_of_date")["value"].mean()
        assert (means.abs() < 1e-10).all()

    def test_cs_demean_fluent_api(self, prices_frame):
        """Test Frame.cs_demean() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cs_demean().get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Mean should be approximately 0 for each date
        means = result.groupby(level="as_of_date")["value"].mean()
        assert (means.abs() < 1e-10).all()


class TestCsWinsorize:
    """Test CsWinsorize operation."""

    def test_cs_winsorize_basic(self, prices_frame):
        """Test basic cross-sectional winsorization."""
        winsorized = CsWinsorize(prices_frame, lower=0.1, upper=0.9)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = winsorized.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Shape should be preserved
        original = prices_frame.get_range(start, end)
        assert result.shape == original.shape

    def test_cs_winsorize_clips_extremes(self, cache_dir):
        """Test that winsorization clips extreme values."""
        # Create data with outliers
        def fetch_with_outliers(start_dt: datetime, end_dt: datetime):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 10,
                "id": list(range(10)),
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],  # 100 is outlier
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_with_outliers, cache_dir=cache_dir)
        winsorized = CsWinsorize(frame, lower=0.1, upper=0.9)

        result = winsorized.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        # The extreme value 100 should be clipped
        assert result["value"].max() < 100.0

    def test_cs_winsorize_invalid_percentiles(self, prices_frame):
        """Test that invalid percentiles raise error."""
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            CsWinsorize(prices_frame, lower=0.9, upper=0.1)  # lower > upper

        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            CsWinsorize(prices_frame, lower=-0.1, upper=0.9)  # lower < 0

    def test_cs_winsorize_fluent_api(self, prices_frame):
        """Test Frame.cs_winsorize() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cs_winsorize(lower=0.1, upper=0.9).get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestCrossSecionalChaining:
    """Test chaining cross-sectional operations."""

    def test_cs_rank_then_zscore(self, prices_frame):
        """Test chaining cs_rank and cs_zscore."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cs_rank().cs_zscore().get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_cs_demean_then_winsorize(self, prices_frame):
        """Test chaining cs_demean and cs_winsorize."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.cs_demean().cs_winsorize().get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_rolling_then_cs_rank(self, prices_frame):
        """Test mixing time-series and cross-sectional operations."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # First apply rolling (time-series), then rank (cross-sectional)
        result = prices_frame.rolling(window=3).cs_rank().get_range(start, end)

        assert isinstance(result, pd.DataFrame)
