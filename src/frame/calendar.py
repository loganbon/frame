"""Calendar implementations for Frame date handling."""

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Iterator


class Calendar(ABC):
    """Abstract base class for calendars."""

    @abstractmethod
    def dt_range(self, start_dt: datetime, end_dt: datetime) -> Iterator[datetime]:
        """Generate valid dates in range [start_dt, end_dt]."""
        pass

    @abstractmethod
    def dt_offset(self, dt: datetime, periods: int) -> datetime:
        """Shift datetime by N calendar periods."""
        pass


class DateCalendar(Calendar):
    """Calendar that includes all dates."""

    def dt_range(self, start_dt: datetime, end_dt: datetime) -> Iterator[datetime]:
        current = start_dt
        while current <= end_dt:
            yield current
            current += timedelta(days=1)

    def dt_offset(self, dt: datetime, periods: int) -> datetime:
        return dt + timedelta(days=periods)


class BDateCalendar(Calendar):
    """Business date calendar - excludes weekends (Sat/Sun)."""

    def dt_range(self, start_dt: datetime, end_dt: datetime) -> Iterator[datetime]:
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:  # Mon-Fri = 0-4
                yield current
            current += timedelta(days=1)

    def dt_offset(self, dt: datetime, periods: int) -> datetime:
        if periods == 0:
            return dt
        direction = 1 if periods > 0 else -1
        remaining = abs(periods)
        current = dt
        while remaining > 0:
            current += timedelta(days=direction)
            if current.weekday() < 5:
                remaining -= 1
        return current
