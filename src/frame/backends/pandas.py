"""Pandas backend implementation."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class PandasBackend:
    """Backend implementation for pandas DataFrames."""

    def concat(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate multiple DataFrames into one."""
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)

    def filter_date_range(
        self, df: pd.DataFrame, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Filter DataFrame to rows within the date range (inclusive).

        Expects 'as_of_date' to be a level in the MultiIndex.
        """
        if df.empty:
            return df

        if isinstance(df.index, pd.MultiIndex) and "as_of_date" in df.index.names:
            dates = df.index.get_level_values("as_of_date")
            mask = (dates >= start) & (dates <= end)
            return df[mask]

        return df

    def drop_index_level(self, df: pd.DataFrame, level: str) -> pd.DataFrame:
        """Remove a level from a MultiIndex, returning a new DataFrame."""
        if isinstance(df.index, pd.MultiIndex) and level in df.index.names:
            return df.droplevel(level)
        return df

    def to_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """Write DataFrame to parquet file.

        Resets index before saving so as_of_date and id are stored as columns.
        This allows efficient column selection when reading.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reset index so as_of_date/id are columns, not index
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif df.index.name is not None:
            df = df.reset_index()
        df.to_parquet(path, index=False)

    def read_parquet(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Read DataFrame from parquet file.

        Args:
            path: Path to the parquet file.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
                Uses pyarrow filter format. Operators: =, !=, <, <=, >, >=, in, not in

        Restores MultiIndex from as_of_date and id columns if they exist.
        """
        df = pd.read_parquet(path, columns=columns, filters=filters)
        # Restore MultiIndex if as_of_date and id columns exist
        if "as_of_date" in df.columns and "id" in df.columns:
            df = df.set_index(["as_of_date", "id"])
        return df

    def get_date_range(self, df: pd.DataFrame) -> tuple[datetime, datetime]:
        """Get the min and max dates from the DataFrame's as_of_date index level."""
        if df.empty:
            raise ValueError("Cannot get date range from empty DataFrame")

        if isinstance(df.index, pd.MultiIndex) and "as_of_date" in df.index.names:
            dates = df.index.get_level_values("as_of_date")
        else:
            raise ValueError("DataFrame must have 'as_of_date' in MultiIndex")

        min_date = dates.min()
        max_date = dates.max()

        if isinstance(min_date, pd.Timestamp):
            min_date = min_date.to_pydatetime()
        if isinstance(max_date, pd.Timestamp):
            max_date = max_date.to_pydatetime()

        return min_date, max_date

    def empty(self) -> pd.DataFrame:
        """Return an empty DataFrame."""
        return pd.DataFrame()

    def is_empty(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is empty."""
        return df.empty

    def sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by as_of_date index level."""
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex) and "as_of_date" in df.index.names:
            return df.sort_index(level="as_of_date")
        return df
