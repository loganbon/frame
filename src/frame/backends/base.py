"""Abstract backend protocol for DataFrame operations."""

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

DF = TypeVar("DF", covariant=True)


class Backend(Protocol[DF]):
    """Protocol defining required DataFrame operations for a backend."""

    def concat(self, frames: list) -> DF:
        """Concatenate multiple DataFrames into one."""
        ...

    def filter_date_range(self, df: DF, start: datetime, end: datetime) -> DF:
        """Filter DataFrame to rows within the date range (inclusive)."""
        ...

    def drop_index_level(self, df: DF, level: str) -> DF:
        """Remove a level from a MultiIndex, returning a new DataFrame."""
        ...

    def to_parquet(self, df: DF, path: Path) -> None:
        """Write DataFrame to parquet file."""
        ...

    def read_parquet(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> DF:
        """Read DataFrame from parquet file.

        Args:
            path: Path to the parquet file.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
                Operators: =, ==, !=, <, <=, >, >=, in, not in
        """
        ...

    def get_date_range(self, df: DF) -> tuple[datetime, datetime]:
        """Get the min and max dates from the DataFrame's as_of_date index level."""
        ...

    def empty(self) -> DF:
        """Return an empty DataFrame."""
        ...

    def is_empty(self, df: DF) -> bool:
        """Check if DataFrame is empty."""
        ...

    def sort_by_date(self, df: DF) -> DF:
        """Sort DataFrame by as_of_date index level or column."""
        ...
