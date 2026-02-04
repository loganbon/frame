"""Common utility functions for Frame."""

from typing import Any

import pandas as pd


def _is_polars(df: Any) -> bool:
    """Check if df is a polars DataFrame or LazyFrame."""
    try:
        import polars as pl
        return isinstance(df, (pl.DataFrame, pl.LazyFrame))
    except ImportError:
        return False


def _is_pandas(df: Any) -> bool:
    """Check if DataFrame is a pandas DataFrame."""
    return isinstance(df, pd.DataFrame)
