"""Tests for calendar functionality."""

from datetime import date, datetime

import pandas as pd
import pytest

from frame import BDateCalendar, Calendar, DateCalendar, Frame


class TestDateCalendar:
    """Test DateCalendar functionality."""

    def test_dt_range_includes_all_dates(self):
        """DateCalendar includes all dates in range."""
        cal = DateCalendar()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 7)

        dates = list(cal.dt_range(start, end))

        assert len(dates) == 7
        assert dates[0] == date(2024, 1, 1)
        assert dates[-1] == date(2024, 1, 7)

    def test_dt_range_includes_weekends(self):
        """DateCalendar includes Saturday and Sunday."""
        cal = DateCalendar()
        # Jan 6, 2024 is Saturday, Jan 7 is Sunday
        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)  # Monday

        dates = list(cal.dt_range(start, end))

        assert len(dates) == 4
        assert date(2024, 1, 6) in dates  # Saturday
        assert date(2024, 1, 7) in dates  # Sunday

    def test_dt_offset_positive(self):
        """dt_offset moves forward by N days."""
        cal = DateCalendar()
        dt = datetime(2024, 1, 1)

        result = cal.dt_offset(dt, 5)

        assert result == datetime(2024, 1, 6)

    def test_dt_offset_negative(self):
        """dt_offset moves backward by N days."""
        cal = DateCalendar()
        dt = datetime(2024, 1, 10)

        result = cal.dt_offset(dt, -3)

        assert result == datetime(2024, 1, 7)

    def test_dt_offset_zero(self):
        """dt_offset with 0 returns same datetime."""
        cal = DateCalendar()
        dt = datetime(2024, 1, 15)

        result = cal.dt_offset(dt, 0)

        assert result == dt


class TestBDateCalendar:
    """Test BDateCalendar (business date) functionality."""

    def test_dt_range_excludes_weekends(self):
        """BDateCalendar excludes Saturday and Sunday."""
        cal = BDateCalendar()
        # Jan 1-7, 2024: Mon, Tue, Wed, Thu, Fri, Sat, Sun
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 7)

        dates = list(cal.dt_range(start, end))

        # Should have 5 business days (Mon-Fri)
        assert len(dates) == 5
        assert date(2024, 1, 6) not in dates  # Saturday
        assert date(2024, 1, 7) not in dates  # Sunday

    def test_dt_range_only_business_days(self):
        """BDateCalendar returns only weekdays."""
        cal = BDateCalendar()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        dates = list(cal.dt_range(start, end))

        # All dates should be weekdays (Mon=0 to Fri=4)
        for d in dates:
            assert d.weekday() < 5, f"{d} is a weekend day"

    def test_dt_offset_positive_skips_weekends(self):
        """dt_offset skips weekends when moving forward."""
        cal = BDateCalendar()
        # Friday Jan 5, 2024
        dt = datetime(2024, 1, 5)

        # +1 business day should be Monday Jan 8
        result = cal.dt_offset(dt, 1)
        assert result == datetime(2024, 1, 8)

    def test_dt_offset_negative_skips_weekends(self):
        """dt_offset skips weekends when moving backward."""
        cal = BDateCalendar()
        # Monday Jan 8, 2024
        dt = datetime(2024, 1, 8)

        # -1 business day should be Friday Jan 5
        result = cal.dt_offset(dt, -1)
        assert result == datetime(2024, 1, 5)

    def test_dt_offset_multiple_weeks(self):
        """dt_offset handles multiple weeks correctly."""
        cal = BDateCalendar()
        dt = datetime(2024, 1, 1)  # Monday

        # +10 business days = 2 weeks of business days = Jan 15 (Monday)
        result = cal.dt_offset(dt, 10)
        assert result == datetime(2024, 1, 15)

    def test_dt_offset_zero(self):
        """dt_offset with 0 returns same datetime."""
        cal = BDateCalendar()
        dt = datetime(2024, 1, 15)

        result = cal.dt_offset(dt, 0)

        assert result == dt


class TestCalendarWithFrame:
    """Test calendar integration with Frame."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Provide a temporary cache directory."""
        return tmp_path / ".frame_cache"

    @pytest.fixture
    def sample_data_func(self):
        """Sample data function that returns data for all dates."""
        def fetch_data(start_dt: datetime, end_dt: datetime):
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
        return fetch_data

    def test_default_calendar_is_bdate(self, sample_data_func, cache_dir):
        """Default calendar is BDateCalendar."""
        frame = Frame(sample_data_func, cache_dir=cache_dir)

        assert isinstance(frame._calendar, BDateCalendar)

    def test_explicit_bdate_calendar(self, sample_data_func, cache_dir):
        """Can explicitly set BDateCalendar."""
        frame = Frame(
            sample_data_func,
            cache_dir=cache_dir,
            calendar=BDateCalendar(),
        )

        assert isinstance(frame._calendar, BDateCalendar)

    def test_date_calendar(self, sample_data_func, cache_dir):
        """Can use DateCalendar for all dates."""
        frame = Frame(
            sample_data_func,
            cache_dir=cache_dir,
            calendar=DateCalendar(),
        )

        assert isinstance(frame._calendar, DateCalendar)

    def test_bdate_calendar_cache_only_business_days(self, cache_dir):
        """BDateCalendar only considers business days as missing."""
        call_counter = {"dates_requested": []}

        def tracking_func(start_dt, end_dt):
            call_counter["dates_requested"].append((start_dt.date(), end_dt.date()))
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
            tracking_func,
            cache_dir=cache_dir,
            calendar=BDateCalendar(),
            chunk_by="month",
        )

        # Request a week of data (includes weekend)
        # Jan 1-7, 2024: Mon, Tue, Wed, Thu, Fri, Sat, Sun
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 7))

        # Verify data was fetched
        assert len(call_counter["dates_requested"]) >= 1

    def test_date_calendar_considers_all_days(self, cache_dir):
        """DateCalendar considers all days including weekends."""
        call_counter = {"count": 0}

        def tracking_func(start_dt, end_dt):
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
            tracking_func,
            cache_dir=cache_dir,
            calendar=DateCalendar(),
            chunk_by="month",
        )

        # Request data
        frame.get_range(datetime(2024, 1, 1), datetime(2024, 1, 7))

        # Verify data was fetched
        assert call_counter["count"] >= 1


class TestCalendarABC:
    """Test Calendar abstract base class."""

    def test_calendar_is_abstract(self):
        """Calendar cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Calendar()  # type: ignore

    def test_custom_calendar(self):
        """Can create custom calendar by subclassing."""
        class WeekdaysOnlyCalendar(Calendar):
            """Custom calendar that only includes Monday-Thursday."""

            def dt_range(self, start_dt, end_dt):
                from datetime import timedelta
                current = start_dt.date()
                end_date = end_dt.date()
                while current <= end_date:
                    if current.weekday() < 4:  # Mon-Thu = 0-3
                        yield current
                    current += timedelta(days=1)

            def dt_offset(self, dt, periods):
                from datetime import timedelta
                if periods == 0:
                    return dt
                direction = 1 if periods > 0 else -1
                remaining = abs(periods)
                current = dt
                while remaining > 0:
                    current += timedelta(days=direction)
                    if current.weekday() < 4:
                        remaining -= 1
                return current

        cal = WeekdaysOnlyCalendar()
        dates = list(cal.dt_range(datetime(2024, 1, 1), datetime(2024, 1, 7)))

        # Should only have Mon-Thu (4 days)
        assert len(dates) == 4
        assert date(2024, 1, 5) not in dates  # Friday
        assert date(2024, 1, 6) not in dates  # Saturday
        assert date(2024, 1, 7) not in dates  # Sunday
