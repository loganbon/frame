"""Polars backend implementation."""

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


def _build_polars_filter(col: str, op: str, val: Any) -> pl.Expr:
    """Convert (column, operator, value) tuple to polars expression.

    Args:
        col: Column name.
        op: Operator (=, ==, !=, <, <=, >, >=, in, not in).
        val: Value to compare against.

    Returns:
        Polars expression for the filter condition.

    Raises:
        ValueError: If the operator is not recognized.
    """
    c = pl.col(col)
    if op in ("=", "=="):
        return c == val
    elif op == "!=":
        return c != val
    elif op == "<":
        return c < val
    elif op == "<=":
        return c <= val
    elif op == ">":
        return c > val
    elif op == ">=":
        return c >= val
    elif op == "in":
        return c.is_in(val)
    elif op == "not in":
        return ~c.is_in(val)
    else:
        raise ValueError(f"Unknown filter operator: {op}")


class PolarsBackend:
    """Backend implementation for polars DataFrames."""

    def concat(self, frames: list[pl.DataFrame]) -> pl.DataFrame:
        """Concatenate multiple DataFrames into one."""
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)

    def filter_date_range(
        self, df: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Filter DataFrame to rows within the date range (inclusive).

        Expects 'as_of_date' column in the DataFrame.
        """
        if df.is_empty():
            return df

        if "as_of_date" in df.columns:
            return df.filter(
                (pl.col("as_of_date") >= start) & (pl.col("as_of_date") <= end)
            )

        return df

    def drop_index_level(self, df: pl.DataFrame, level: str) -> pl.DataFrame:
        """Remove a column (Polars doesn't have index levels).

        In Polars, we treat the 'as_of_date' column like an index level.
        """
        if level in df.columns:
            return df.drop(level)
        return df

    def to_parquet(self, df: pl.DataFrame, path: Path) -> None:
        """Write DataFrame to parquet file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    def read_parquet(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> pl.DataFrame:
        """Read DataFrame from parquet file.

        Uses scan_parquet for lazy evaluation with predicate pushdown.

        Args:
            path: Path to the parquet file.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
                Operators: =, ==, !=, <, <=, >, >=, in, not in
        """
        lf = pl.scan_parquet(path)
        if columns:
            lf = lf.select(columns)
        if filters:
            for col, op, val in filters:
                lf = lf.filter(_build_polars_filter(col, op, val))
        return lf.collect()

    def get_date_range(self, df: pl.DataFrame) -> tuple[datetime, datetime]:
        """Get the min and max dates from the DataFrame's as_of_date column."""
        if df.is_empty():
            raise ValueError("Cannot get date range from empty DataFrame")

        if "as_of_date" not in df.columns:
            raise ValueError("DataFrame must have 'as_of_date' column")

        min_date = df["as_of_date"].min()
        max_date = df["as_of_date"].max()

        return min_date, max_date

    def empty(self) -> pl.DataFrame:
        """Return an empty DataFrame."""
        return pl.DataFrame()

    def is_empty(self, df: pl.DataFrame) -> bool:
        """Check if DataFrame is empty."""
        return df.is_empty()

    def sort_by_date(self, df: pl.DataFrame) -> pl.DataFrame:
        """Sort DataFrame by as_of_date column."""
        if df.is_empty():
            return df
        if "as_of_date" in df.columns:
            return df.sort("as_of_date")
        return df
