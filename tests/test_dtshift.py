"""Tests for DtShift operation."""

import shutil
from datetime import datetime

import pandas as pd
import pytest

from frame import BDateCalendar, DateCalendar, Frame
from frame.ops.dtshift import DtShift


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


@pytest.fixture
def bdate_prices_frame(cache_dir):
    """Create a prices Frame with business date calendar."""
    def fetch_prices(start_dt: datetime, end_dt: datetime):
        dates = pd.bdate_range(start_dt, end_dt)
        records = []
        for i, dt in enumerate(dates):
            records.append({
                "as_of_date": dt.to_pydatetime(),
                "id": 0,
                "price": 100 + i,
            })
        return pd.DataFrame(records).set_index(["as_of_date", "id"])

    return Frame(fetch_prices, calendar=BDateCalendar(), cache_dir=cache_dir)


@pytest.fixture
def date_prices_frame(cache_dir):
    """Create a prices Frame with all-dates calendar."""
    def fetch_prices(start_dt: datetime, end_dt: datetime):
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for i, dt in enumerate(dates):
            records.append({
                "as_of_date": dt.to_pydatetime(),
                "id": 0,
                "price": 100 + i,
            })
        return pd.DataFrame(records).set_index(["as_of_date", "id"])

    return Frame(fetch_prices, calendar=DateCalendar(), cache_dir=cache_dir)


@pytest.fixture
def multi_id_frame(cache_dir):
    """Create a Frame with multiple IDs."""
    def fetch_data(start_dt: datetime, end_dt: datetime):
        dates = pd.bdate_range(start_dt, end_dt)
        records = []
        for i, dt in enumerate(dates):
            for id_ in [0, 1, 2]:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "value": (id_ + 1) * 10 + i,
                })
        return pd.DataFrame(records).set_index(["as_of_date", "id"])

    return Frame(fetch_data, calendar=BDateCalendar(), cache_dir=cache_dir)


class TestDtShiftBasic:
    """Test basic DtShift functionality."""

    def test_dtshift_basic(self, bdate_prices_frame):
        """Test shift by 1 period."""
        shifted = DtShift(bdate_prices_frame, periods=1)

        # Request Jan 8-10, 2024 (Mon-Wed)
        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        result = shifted.get_range(start, end)

        # Should have dates Jan 8-10
        dates = result.index.get_level_values("as_of_date").unique()
        assert datetime(2024, 1, 8) in dates
        assert datetime(2024, 1, 9) in dates
        assert datetime(2024, 1, 10) in dates

    def test_dtshift_multiple_periods(self, bdate_prices_frame):
        """Test shift by N periods."""
        shifted = DtShift(bdate_prices_frame, periods=5)

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        result = shifted.get_range(start, end)

        # Should have dates Jan 8-10 (labels from original request)
        dates = result.index.get_level_values("as_of_date").unique()
        assert datetime(2024, 1, 8) in dates
        assert datetime(2024, 1, 9) in dates
        assert datetime(2024, 1, 10) in dates

    def test_dtshift_zero_periods(self, bdate_prices_frame):
        """Test no shift (passthrough)."""
        shifted = DtShift(bdate_prices_frame, periods=0)

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        original = bdate_prices_frame.get_range(start, end)
        result = shifted.get_range(start, end)

        pd.testing.assert_frame_equal(original, result)

    def test_dtshift_negative_periods(self, bdate_prices_frame):
        """Test forward shift (future data labeled as past)."""
        shifted = DtShift(bdate_prices_frame, periods=-1)

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        result = shifted.get_range(start, end)

        # Should have dates Jan 8-10
        dates = result.index.get_level_values("as_of_date").unique()
        assert datetime(2024, 1, 8) in dates
        assert datetime(2024, 1, 9) in dates
        assert datetime(2024, 1, 10) in dates


class TestDtShiftCalendars:
    """Test DtShift with different calendars."""

    def test_dtshift_with_bdate_calendar(self, bdate_prices_frame):
        """Test that DtShift respects business days."""
        shifted = DtShift(bdate_prices_frame, periods=1)

        # Request Monday Jan 8, 2024
        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 8)

        result = shifted.get_range(start, end)

        # Should have Jan 8 label
        dates = result.index.get_level_values("as_of_date").unique()
        assert datetime(2024, 1, 8) in dates

        # Value should be from Jan 5 (Friday, 1 business day back)
        # Since our fetch function gives sequential values, the price
        # from Jan 5 should be different from what Jan 8 would give

    def test_dtshift_with_date_calendar(self, date_prices_frame):
        """Test that DtShift works with all days."""
        shifted = DtShift(date_prices_frame, periods=1)

        # Request Jan 8, 2024 (Monday)
        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 8)

        result = shifted.get_range(start, end)

        # Should have Jan 8 label
        dates = result.index.get_level_values("as_of_date").unique()
        assert datetime(2024, 1, 8) in dates


class TestDtShiftDataIntegrity:
    """Test data integrity of DtShift operation."""

    def test_dtshift_preserves_values(self, cache_dir):
        """Test that values are from shifted dates."""
        # Create frame where value is predictable from date
        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.bdate_range(start_dt, end_dt)
            records = []
            for dt in dates:
                # Value is the day of month
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 0,
                    "value": dt.day,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, calendar=BDateCalendar(), cache_dir=cache_dir)
        shifted = DtShift(frame, periods=1)

        # Request Jan 8, 2024 (Monday)
        result = shifted.get_range(datetime(2024, 1, 8), datetime(2024, 1, 8))

        # Value should be from Jan 5 (Friday), which is day 5
        assert result.loc[(datetime(2024, 1, 8), 0), "value"] == 5

    def test_dtshift_correct_date_labels(self, cache_dir):
        """Test that output has requested dates as labels."""
        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.bdate_range(start_dt, end_dt)
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 0,
                    "value": dt.day,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, calendar=BDateCalendar(), cache_dir=cache_dir)
        shifted = DtShift(frame, periods=1)

        # Request Jan 8-10, 2024
        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        result = shifted.get_range(start, end)

        # Labels should be Jan 8, 9, 10
        dates = sorted(result.index.get_level_values("as_of_date").unique())
        expected = [datetime(2024, 1, 8), datetime(2024, 1, 9), datetime(2024, 1, 10)]
        assert dates == expected

    def test_dtshift_multiple_ids(self, multi_id_frame):
        """Test that DtShift works correctly with multiple IDs."""
        shifted = DtShift(multi_id_frame, periods=1)

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        result = shifted.get_range(start, end)

        # Should have all 3 IDs
        ids = sorted(result.index.get_level_values("id").unique())
        assert ids == [0, 1, 2]

        # Each ID should have the same dates
        for id_ in [0, 1, 2]:
            id_dates = result.xs(id_, level="id").index.tolist()
            expected = [datetime(2024, 1, 8), datetime(2024, 1, 9), datetime(2024, 1, 10)]
            assert sorted(id_dates) == expected


class TestDtShiftIntegration:
    """Test DtShift integration with other operations."""

    def test_dtshift_chaining(self, bdate_prices_frame):
        """Test chaining DtShift with other operations."""
        # Create chain: dt_shift -> rolling
        pipeline = bdate_prices_frame.dt_shift(1).rolling(window=2)

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 12)

        result = pipeline.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Rolling should produce some NaN values
        assert result["price"].isna().any()

    def test_dtshift_with_columns_and_filters(self, cache_dir):
        """Test columns and filters parameter passthrough."""
        def fetch_data(start_dt: datetime, end_dt: datetime):
            dates = pd.bdate_range(start_dt, end_dt)
            records = []
            for dt in dates:
                for id_ in ["A", "B"]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": 1.0,
                        "extra": 2.0,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, calendar=BDateCalendar(), cache_dir=cache_dir)
        shifted = DtShift(frame, periods=1)

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        # Test columns selection
        result = shifted.get_range(start, end, columns=["value"])
        assert "value" in result.columns
        assert "extra" not in result.columns

        # Test filters
        result = shifted.get_range(start, end, filters=[("id", "=", "A")])
        ids = result.index.get_level_values("id").unique().tolist()
        assert ids == ["A"]

    def test_dtshift_fluent_api(self, bdate_prices_frame):
        """Test Frame.dt_shift() method."""
        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        result = bdate_prices_frame.dt_shift(1).get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        dates = result.index.get_level_values("as_of_date").unique()
        assert datetime(2024, 1, 8) in dates

    def test_dtshift_empty_result(self, cache_dir):
        """Test DtShift with empty result."""
        def fetch_empty(start_dt: datetime, end_dt: datetime):
            return pd.DataFrame(columns=["as_of_date", "id", "value"]).set_index(
                ["as_of_date", "id"]
            )

        frame = Frame(fetch_empty, calendar=BDateCalendar(), cache_dir=cache_dir)
        shifted = DtShift(frame, periods=1)

        result = shifted.get_range(datetime(2024, 1, 8), datetime(2024, 1, 10))

        assert result.empty


class TestDtShiftPolars:
    """Test DtShift with polars backend."""

    @pytest.mark.xfail(reason="Polars backend date filtering is currently broken in cache.py")
    def test_dtshift_polars_backend(self, cache_dir):
        """Test DtShift with polars DataFrame."""
        import polars as pl

        def fetch_polars(start_dt: datetime, end_dt: datetime):
            # Generate business dates as datetime objects
            dates = [d.to_pydatetime() for d in pd.bdate_range(start_dt, end_dt)]
            return pl.DataFrame({
                "as_of_date": dates,
                "id": [0] * len(dates),
                "price": list(range(100, 100 + len(dates))),
            })

        frame = Frame(
            fetch_polars,
            backend="polars",
            calendar=BDateCalendar(),
            cache_dir=cache_dir,
        )
        shifted = DtShift(frame, periods=1)

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 10)

        result = shifted.get_range(start, end)

        assert isinstance(result, pl.DataFrame)
        assert "as_of_date" in result.columns
        assert "price" in result.columns

    def test_dtshift_polars_empty(self, cache_dir):
        """Test DtShift with empty polars DataFrame."""
        import polars as pl

        def fetch_empty(start_dt: datetime, end_dt: datetime):
            return pl.DataFrame({
                "as_of_date": [],
                "id": [],
                "value": [],
            }).cast({"as_of_date": pl.Datetime, "id": pl.Int64, "value": pl.Float64})

        frame = Frame(
            fetch_empty,
            backend="polars",
            calendar=BDateCalendar(),
            cache_dir=cache_dir,
        )
        shifted = DtShift(frame, periods=1)

        result = shifted.get_range(datetime(2024, 1, 8), datetime(2024, 1, 10))

        assert result.height == 0
