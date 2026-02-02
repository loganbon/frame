"""Backend conversion operations."""

from typing import Any

import pandas as pd

from frame.ops.base import Operation

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class ToPandas(Operation):
    """Convert input DataFrame to pandas.

    Works with any input that has a to_pandas() method (like Polars)
    or is already a pandas DataFrame.

    Example:
        polars_frame = Frame(fetch_func, backend="polars")
        pandas_view = ToPandas(polars_frame)
        df = pandas_view.get_range(start, end)  # Returns pandas.DataFrame
    """

    def _apply(self, inputs: list[Any], **params: Any) -> pd.DataFrame:
        df = inputs[0]
        if hasattr(df, "to_pandas"):
            result = df.to_pandas()
            # Set the standard [as_of_date, id] MultiIndex if columns exist
            if "as_of_date" in result.columns and "id" in result.columns:
                result = result.set_index(["as_of_date", "id"])
            return result
        if isinstance(df, pd.DataFrame):
            return df
        raise TypeError(f"Cannot convert {type(df)} to pandas")


class ToPolars(Operation):
    """Convert input DataFrame to polars.

    Works with pandas DataFrames or anything already a Polars DataFrame.

    Example:
        pandas_frame = Frame(fetch_func, backend="pandas")
        polars_view = ToPolars(pandas_frame)
        df = polars_view.get_range(start, end)  # Returns polars.DataFrame
    """

    def _apply(self, inputs: list[Any], **params: Any) -> Any:
        if not HAS_POLARS:
            raise ImportError("polars is required for ToPolars operation")

        df = inputs[0]
        if isinstance(df, pl.DataFrame):
            return df
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        if hasattr(df, "to_polars"):
            return df.to_polars()
        raise TypeError(f"Cannot convert {type(df)} to polars")
