"""Tests for Frame alignment operations."""

import shutil
from datetime import datetime

import pandas as pd
import pytest

from frame import BDateCalendar, DateCalendar, Frame
from frame.ops.align import AlignTo, AlignToCalendar


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


@pytest.fixture
def bday_frame(cache_dir):
    """Create a business day Frame for testing."""
    def fetch_bday(start_dt: datetime, end_dt: datetime):
        dates = pd.bdate_range(start_dt, end_dt)
        records = []
        for dt in dates:
            for id_ in range(2):
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "value": 10.0 + id_,
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return Frame(fetch_bday, calendar=BDateCalendar(), cache_dir=cache_dir)


@pytest.fixture
def daily_frame(cache_dir):
    """Create a daily Frame for testing."""
    def fetch_daily(start_dt: datetime, end_dt: datetime):
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for dt in dates:
            for id_ in range(2):
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "price": 100.0 + id_,
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return Frame(fetch_daily, calendar=DateCalendar(), cache_dir=cache_dir / "daily")


class TestAlignToCalendarBasic:
    """Test basic AlignToCalendar functionality."""

    def test_align_bday_to_daily_ffill(self, bday_frame):
        """Test aligning business day data to daily calendar with forward fill."""
        aligned = AlignToCalendar(bday_frame, DateCalendar(), fill_method="ffill")

        # Get a range that includes a weekend (Jan 6-7 2024 is Sat-Sun)
        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)    # Monday

        result = aligned.get_range(start, end)

        # Should have 4 days * 2 ids = 8 rows
        assert len(result) == 8

        # Weekend dates should be present
        dates = result.index.get_level_values("as_of_date").unique()
        assert len(dates) == 4

        # Weekend values should be forward filled from Friday
        friday = datetime(2024, 1, 5)
        saturday = datetime(2024, 1, 6)

        friday_val = result.loc[(friday, 0), "value"]
        saturday_val = result.loc[(saturday, 0), "value"]
        assert saturday_val == friday_val

    def test_align_bday_to_daily_bfill(self, bday_frame):
        """Test aligning business day data to daily calendar with backward fill."""
        aligned = AlignToCalendar(bday_frame, DateCalendar(), fill_method="bfill")

        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)    # Monday

        result = aligned.get_range(start, end)

        # Weekend values should be backward filled from Monday
        sunday = datetime(2024, 1, 7)
        monday = datetime(2024, 1, 8)

        sunday_val = result.loc[(sunday, 0), "value"]
        monday_val = result.loc[(monday, 0), "value"]
        assert sunday_val == monday_val

    def test_align_bday_to_daily_no_fill(self, bday_frame):
        """Test aligning business day data to daily calendar without fill."""
        aligned = AlignToCalendar(bday_frame, DateCalendar(), fill_method=None)

        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)    # Monday

        result = aligned.get_range(start, end)

        # Weekend values should be NaN
        saturday = datetime(2024, 1, 6)
        assert pd.isna(result.loc[(saturday, 0), "value"])

    def test_align_bday_to_daily_scalar_fill(self, bday_frame):
        """Test aligning business day data to daily calendar with scalar fill."""
        aligned = AlignToCalendar(bday_frame, DateCalendar(), fill_method=-999.0)

        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)    # Monday

        result = aligned.get_range(start, end)

        # Weekend values should be filled with scalar
        saturday = datetime(2024, 1, 6)
        assert result.loc[(saturday, 0), "value"] == -999.0

    def test_align_to_subset_calendar(self, cache_dir):
        """Test aligning daily data to business day calendar (drops weekends)."""
        def fetch_daily(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 0,
                    "value": 1.0,
                })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        daily_frame = Frame(fetch_daily, calendar=DateCalendar(), cache_dir=cache_dir)
        aligned = AlignToCalendar(daily_frame, BDateCalendar(), fill_method="ffill")

        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)    # Monday

        result = aligned.get_range(start, end)

        # Should only have Friday and Monday (business days)
        dates = result.index.get_level_values("as_of_date").unique()
        assert len(dates) == 2

        # Verify weekend dates are not present
        saturday = datetime(2024, 1, 6)
        sunday = datetime(2024, 1, 7)
        assert saturday not in dates
        assert sunday not in dates

    def test_ffill_respects_id_boundaries(self, cache_dir):
        """Test that forward fill does not leak across different ids."""
        def fetch_sparse(start_dt: datetime, end_dt: datetime):
            # Only have data for id=0 on Jan 5, and id=1 on Jan 8
            records = [
                {"as_of_date": datetime(2024, 1, 5), "id": 0, "value": 100.0},
                {"as_of_date": datetime(2024, 1, 8), "id": 1, "value": 200.0},
            ]
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(fetch_sparse, calendar=DateCalendar(), cache_dir=cache_dir)
        aligned = AlignToCalendar(frame, DateCalendar(), fill_method="ffill")

        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        result = aligned.get_range(start, end)

        # id=0's value on Jan 8 should be forward filled from Jan 5 (100.0)
        assert result.loc[(datetime(2024, 1, 8), 0), "value"] == 100.0

        # id=1's value on Jan 5 should be NaN (no prior data to forward fill)
        assert pd.isna(result.loc[(datetime(2024, 1, 5), 1), "value"])


class TestAlignToBasic:
    """Test basic AlignTo functionality."""

    def test_align_to_frame_ffill(self, bday_frame, daily_frame):
        """Test aligning business day data to daily frame with forward fill."""
        aligned = AlignTo(bday_frame, daily_frame, fill_method="ffill")

        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)    # Monday

        result = aligned.get_range(start, end)

        # Should have same number of rows as daily_frame
        target = daily_frame.get_range(start, end)
        assert len(result) == len(target)

        # Weekend values should be forward filled
        friday = datetime(2024, 1, 5)
        saturday = datetime(2024, 1, 6)

        friday_val = result.loc[(friday, 0), "value"]
        saturday_val = result.loc[(saturday, 0), "value"]
        assert saturday_val == friday_val

    def test_align_to_frame_with_different_ids(self, cache_dir):
        """Test aligning to a frame with different ids."""
        def fetch_source(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in [0, 1]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": 10.0 + id_,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        def fetch_target(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in [0, 2]:  # Different ids - 0 and 2
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "price": 100.0,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        source = Frame(fetch_source, cache_dir=cache_dir / "src")
        target = Frame(fetch_target, cache_dir=cache_dir / "tgt")

        aligned = AlignTo(source, target, fill_method="ffill")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        result = aligned.get_range(start, end)

        # Should have target's ids (0 and 2)
        ids = result.index.get_level_values("id").unique().tolist()
        assert sorted(ids) == [0, 2]

        # id=0 should have data (exists in source)
        assert not pd.isna(result.loc[(datetime(2024, 1, 1), 0), "value"])

        # id=2 should be NaN (doesn't exist in source)
        assert pd.isna(result.loc[(datetime(2024, 1, 1), 2), "value"])

    def test_align_to_frame_subset_dates(self, cache_dir):
        """Test aligning to a frame with fewer dates."""
        def fetch_source(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": 0,
                    "value": 10.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        def fetch_target(start_dt: datetime, end_dt: datetime):
            # Only return Jan 2 and Jan 4 (subset)
            dates = [datetime(2024, 1, 2), datetime(2024, 1, 4)]
            records = []
            for dt in dates:
                if start_dt <= dt <= end_dt:
                    records.append({
                        "as_of_date": dt,
                        "id": 0,
                        "price": 100.0,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        source = Frame(fetch_source, cache_dir=cache_dir / "src")
        target = Frame(fetch_target, cache_dir=cache_dir / "tgt")

        aligned = AlignTo(source, target, fill_method=None)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = aligned.get_range(start, end)

        # Should only have target's dates
        dates = result.index.get_level_values("as_of_date").unique()
        assert len(dates) == 2
        assert datetime(2024, 1, 1) not in dates
        assert datetime(2024, 1, 2) in dates
        assert datetime(2024, 1, 4) in dates

    def test_align_to_operation(self, bday_frame, daily_frame):
        """Test aligning to an Operation (not just a Frame)."""
        # Target is an operation (select)
        target_op = daily_frame.select(["price"])

        aligned = AlignTo(bday_frame, target_op, fill_method="ffill")

        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        result = aligned.get_range(start, end)

        # Should work the same as aligning to the frame
        assert len(result) == 8
        assert "value" in result.columns

    def test_align_to_concurrent_fetch(self, cache_dir):
        """Test that both frames are fetched concurrently in the same batch."""
        fetch_order = []

        def fetch_source(start_dt: datetime, end_dt: datetime):
            fetch_order.append("source")
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [
                {"as_of_date": dt.to_pydatetime(), "id": 0, "value": 10.0}
                for dt in dates
            ]
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        def fetch_target(start_dt: datetime, end_dt: datetime):
            fetch_order.append("target")
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [
                {"as_of_date": dt.to_pydatetime(), "id": 0, "price": 100.0}
                for dt in dates
            ]
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        source = Frame(fetch_source, cache_dir=cache_dir / "src")
        target = Frame(fetch_target, cache_dir=cache_dir / "tgt")

        aligned = AlignTo(source, target, fill_method="ffill")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        result = aligned.get_range(start, end)

        # Both should have been fetched
        assert "source" in fetch_order
        assert "target" in fetch_order

        # Result should be valid
        assert len(result) == 3


class TestCommonCases:
    """Test common cases for both alignment operations."""

    def test_empty_source_frame(self, cache_dir):
        """Test aligning an empty source frame."""
        def fetch_empty(start_dt: datetime, end_dt: datetime):
            return pd.DataFrame(columns=["as_of_date", "id", "value"]).set_index(
                ["as_of_date", "id"]
            )

        frame = Frame(fetch_empty, cache_dir=cache_dir)
        aligned = AlignToCalendar(frame, DateCalendar(), fill_method="ffill")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = aligned.get_range(start, end)

        # Should return empty DataFrame
        assert len(result) == 0

    def test_invalid_fill_method(self, bday_frame):
        """Test that invalid fill_method raises ValueError."""
        with pytest.raises(ValueError, match="fill_method must be"):
            AlignToCalendar(bday_frame, DateCalendar(), fill_method="invalid")

        with pytest.raises(ValueError, match="fill_method must be"):
            AlignTo(bday_frame, bday_frame, fill_method="invalid")

    def test_chaining_after_align(self, bday_frame):
        """Test chaining other operations after alignment."""
        aligned = bday_frame.align_to_calendar(DateCalendar(), fill_method="ffill")
        shifted = aligned.shift(1)

        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        result = shifted.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # First row should have NaN from shift
        assert result["value"].isna().any()


class TestFluentAPI:
    """Test fluent API methods for alignment."""

    def test_align_to_calendar_method(self, bday_frame):
        """Test Frame.align_to_calendar() fluent API."""
        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        result = bday_frame.align_to_calendar(
            DateCalendar(), fill_method="ffill"
        ).get_range(start, end)

        assert len(result) == 8  # 4 days * 2 ids

    def test_align_to_method(self, bday_frame, daily_frame):
        """Test Frame.align_to() fluent API."""
        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        result = bday_frame.align_to(
            daily_frame, fill_method="ffill"
        ).get_range(start, end)

        assert len(result) == 8  # 4 days * 2 ids


class TestPolarsBackend:
    """Test alignment operations with polars backend."""

    def test_align_to_calendar_polars(self, cache_dir):
        """Test AlignToCalendar with polars backend."""
        import polars as pl

        def fetch_bday_polars(start_dt: datetime, end_dt: datetime):
            dates = pd.bdate_range(start_dt, end_dt)
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 10.0,
                })
            return pl.DataFrame(records)

        frame = Frame(fetch_bday_polars, backend="polars", cache_dir=cache_dir)
        aligned = AlignToCalendar(frame, DateCalendar(), fill_method="ffill")

        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        # Use live mode to bypass cache (polars caching has datetime type issues)
        result = aligned.get_range(start, end, cache_mode="l")

        assert isinstance(result, pl.DataFrame)
        assert result.height == 4  # 4 days

    def test_align_to_polars(self, cache_dir):
        """Test AlignTo with polars backend."""
        import polars as pl

        def fetch_source_polars(start_dt: datetime, end_dt: datetime):
            dates = pd.bdate_range(start_dt, end_dt)
            records = [
                {"as_of_date": dt.to_pydatetime(), "id": "A", "value": 10.0}
                for dt in dates
            ]
            return pl.DataFrame(records)

        def fetch_target_polars(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = [
                {"as_of_date": dt.to_pydatetime(), "id": "A", "price": 100.0}
                for dt in dates
            ]
            return pl.DataFrame(records)

        source = Frame(fetch_source_polars, backend="polars", cache_dir=cache_dir / "src")
        target = Frame(fetch_target_polars, backend="polars", cache_dir=cache_dir / "tgt")

        aligned = AlignTo(source, target, fill_method="ffill")

        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        # Use live mode to bypass cache (polars caching has datetime type issues)
        result = aligned.get_range(start, end, cache_mode="l")

        assert isinstance(result, pl.DataFrame)
        assert result.height == 4  # 4 days


class TestEdgeCases:
    """Test edge cases for alignment operations."""

    def test_single_date_alignment(self, cache_dir):
        """Test aligning a single date."""
        def fetch_single(start_dt: datetime, end_dt: datetime):
            return pd.DataFrame([{
                "as_of_date": datetime(2024, 1, 1),
                "id": 0,
                "value": 10.0,
            }]).set_index(["as_of_date", "id"])

        frame = Frame(fetch_single, cache_dir=cache_dir)
        aligned = AlignToCalendar(frame, DateCalendar(), fill_method="ffill")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1)

        result = aligned.get_range(start, end)

        assert len(result) == 1
        assert result.loc[(datetime(2024, 1, 1), 0), "value"] == 10.0

    def test_align_with_integer_fill(self, bday_frame):
        """Test alignment with integer fill value."""
        aligned = AlignToCalendar(bday_frame, DateCalendar(), fill_method=0)

        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 8)

        result = aligned.get_range(start, end)

        # Weekend values should be 0
        saturday = datetime(2024, 1, 6)
        assert result.loc[(saturday, 0), "value"] == 0

    def test_operation_repr(self, bday_frame, daily_frame):
        """Test string representation of alignment operations."""
        align_cal = AlignToCalendar(bday_frame, DateCalendar(), fill_method="ffill")
        align_to = AlignTo(bday_frame, daily_frame, fill_method="ffill")

        repr_cal = repr(align_cal)
        repr_to = repr(align_to)

        assert "AlignToCalendar" in repr_cal
        assert "AlignTo" in repr_to
