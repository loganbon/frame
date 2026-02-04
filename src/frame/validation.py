"""DataFrame validation utilities."""

from typing import Any

import pandas as pd

from frame.utils import _is_pandas, _is_polars


class ValidationError(Exception):
    """Raised when DataFrame validation fails."""
    pass


def validate_dataframe(df: Any) -> None:
    """Validate DataFrame structure before caching.

    Checks:
    1. DataFrame has as_of_date and id (in index or columns)
    2. No duplicate rows when considering (as_of_date, id)

    Raises:
        ValidationError: If validation fails
    """
    if _is_polars(df):
        _validate_polars(df)
    elif _is_pandas(df):
        _validate_pandas(df)
    else:
        raise ValidationError(f"Unknown DataFrame type: {type(df)}")


def _validate_pandas(df: pd.DataFrame) -> None:
    """Validate pandas DataFrame."""
    # Get as_of_date and id from index or columns
    index_names = list(df.index.names) if df.index.names[0] is not None else []
    all_keys = set(index_names) | set(df.columns)

    # Check required keys exist
    if "as_of_date" not in all_keys:
        raise ValidationError("DataFrame missing 'as_of_date' in index or columns")
    if "id" not in all_keys:
        raise ValidationError("DataFrame missing 'id' in index or columns")

    # Check for duplicates
    if "as_of_date" in index_names and "id" in index_names:
        # Both in index - check index duplicates
        if df.index.duplicated().any():
            raise ValidationError("DataFrame has duplicate (as_of_date, id) rows")
    else:
        # Need to build key from mix of index and columns
        df_check = df.reset_index() if index_names else df
        if df_check.duplicated(subset=["as_of_date", "id"]).any():
            raise ValidationError("DataFrame has duplicate (as_of_date, id) rows")


def _validate_polars(df: Any) -> None:
    """Validate polars DataFrame."""
    import polars as pl

    # Check required columns exist
    if "as_of_date" not in df.columns:
        raise ValidationError("DataFrame missing 'as_of_date' column")
    if "id" not in df.columns:
        raise ValidationError("DataFrame missing 'id' column")

    # Check for duplicates
    n_rows = df.height
    n_unique = df.select(["as_of_date", "id"]).unique().height
    if n_unique < n_rows:
        raise ValidationError("DataFrame has duplicate (as_of_date, id) rows")
