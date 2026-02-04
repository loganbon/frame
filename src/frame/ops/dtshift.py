"""Date shift operation for temporal date range shifting."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from frame.cache import CacheMode
from frame.ops.base import Operation, _apply_filters, _is_polars

if TYPE_CHECKING:
    from frame.core import Frame


class DtShift(Operation):
    """Shift date range when fetching data.

    Fetches data from N periods back and re-labels dates to match
    the originally requested range.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        periods: int = 1,
    ) -> None:
        """Initialize DtShift operation.

        Args:
            frame: Input Frame or Operation.
            periods: Number of periods to shift backward. Positive shifts
                backward (returns past data), negative shifts forward.
        """
        super().__init__(frame, periods=periods)

    def _execute(
        self,
        start_dt: datetime,
        end_dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Execute with shifted date range."""
        periods = self._params["periods"]

        # Get calendar from input frame
        calendar = self._inputs[0]._calendar

        # Shift dates backward by N periods
        shifted_start = calendar.dt_offset(start_dt, -periods)
        shifted_end = calendar.dt_offset(end_dt, -periods)

        # Fetch data for shifted dates
        lazy_input = self._inputs[0].get_range(
            shifted_start, shifted_end, cache_mode=cache_mode
        )

        # Resolve if lazy
        if hasattr(lazy_input, "_resolve"):
            df = lazy_input._resolve()
        else:
            df = lazy_input

        # Re-label dates forward by N periods
        result = self._relabel_dates(df, periods, calendar)

        # Apply columns selection to output
        if columns is not None:
            result = result[columns]

        # Apply filters to output
        if filters is not None:
            result = _apply_filters(result, filters)

        return result

    def _apply(self, inputs: list[Any], **params: Any) -> Any:
        """Apply date relabeling to resolved input data.

        This is called when DtShift is used as input to another operation
        (e.g., frame.dt_shift(1).rolling(3)). In this case, the data has
        already been fetched for the original date range by the parent
        operation, so we just relabel the dates.

        Note: When _apply is called, we're in a chain context where the
        date shifting of the fetch cannot happen (dates are fixed by parent).
        We only relabel the existing data's dates.
        """
        periods = params["periods"]
        calendar = self._inputs[0]._calendar
        df = inputs[0]

        return self._relabel_dates(df, periods, calendar)

    def _relabel_dates(self, df: Any, periods: int, calendar: Any) -> Any:
        """Shift dates in the DataFrame by N periods."""
        if _is_polars(df):
            return self._relabel_dates_polars(df, periods, calendar)
        else:
            return self._relabel_dates_pandas(df, periods, calendar)

    def _relabel_dates_pandas(
        self, df: pd.DataFrame, periods: int, calendar: Any
    ) -> pd.DataFrame:
        """Relabel dates for pandas DataFrame."""
        if df.empty:
            return df

        # Reset index to modify as_of_date
        result = df.reset_index()

        # Shift each date forward by N periods
        result["as_of_date"] = result["as_of_date"].apply(
            lambda x: calendar.dt_offset(x, periods)
        )

        # Restore MultiIndex
        return result.set_index(["as_of_date", "id"])

    def _relabel_dates_polars(
        self, df: Any, periods: int, calendar: Any
    ) -> Any:
        """Relabel dates for polars DataFrame."""
        import polars as pl

        if df.height == 0:
            return df

        # Get unique dates and create mapping
        old_dates = df.select("as_of_date").unique().to_series().to_list()
        new_dates = [calendar.dt_offset(d, periods) for d in old_dates]

        # Create a mapping dataframe for the join
        date_mapping = pl.DataFrame({
            "as_of_date_old": old_dates,
            "as_of_date_new": new_dates,
        })

        # Join to get new dates, then drop old and rename
        result = (
            df.join(date_mapping, left_on="as_of_date", right_on="as_of_date_old")
            .drop("as_of_date", "as_of_date_old")
            .rename({"as_of_date_new": "as_of_date"})
        )

        return result
