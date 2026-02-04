"""Alignment operations for Frame date/id synchronization."""

from typing import TYPE_CHECKING, Any

import pandas as pd

from frame.ops.base import Operation, _is_polars

if TYPE_CHECKING:
    from frame.calendar import Calendar
    from frame.core import Frame


class AlignToCalendar(Operation):
    """Align Frame data to a target calendar.

    Reindexes the input Frame's data to match dates from the target calendar,
    using the input data's date range (min/max dates). Missing values can be
    filled using forward fill, backward fill, a scalar value, or left as NaN.

    Example:
        # Convert business-day-only data to include weekends
        daily_data = bday_frame.align_to_calendar(DateCalendar(), fill_method="ffill")
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        calendar: "Calendar",
        fill_method: str | float | int | None = "ffill",
    ) -> None:
        """Initialize AlignToCalendar operation.

        Args:
            frame: Input Frame or Operation to align.
            calendar: Target calendar to align to.
            fill_method: How to fill missing values:
                - "ffill": Forward fill (use last available value)
                - "bfill": Backward fill (use next available value)
                - None: Leave as NaN/null
                - float/int: Fill with constant value
        """
        self._validate_fill_method(fill_method)
        super().__init__(frame, calendar=calendar, fill_method=fill_method)

    def _validate_fill_method(self, fill_method: str | float | int | None) -> None:
        """Validate the fill_method parameter."""
        if fill_method is None:
            return
        if isinstance(fill_method, (int, float)):
            return
        if isinstance(fill_method, str) and fill_method in ("ffill", "bfill"):
            return
        raise ValueError(
            f"fill_method must be 'ffill', 'bfill', None, or a scalar value, "
            f"got {fill_method!r}"
        )

    def _apply(
        self,
        inputs: list[Any],
        calendar: "Calendar",
        fill_method: str | float | int | None,
    ) -> Any:
        df = inputs[0]

        if _is_polars(df):
            return self._apply_polars(df, calendar, fill_method)
        else:
            return self._apply_pandas(df, calendar, fill_method)

    def _apply_pandas(
        self,
        df: pd.DataFrame,
        calendar: "Calendar",
        fill_method: str | float | int | None,
    ) -> pd.DataFrame:
        """Apply alignment using pandas."""
        if df.empty:
            return df

        # Get date range from input data
        dates = df.index.get_level_values("as_of_date")
        min_date, max_date = dates.min(), dates.max()

        # Generate target calendar dates
        target_dates = list(calendar.dt_range(min_date, max_date))
        ids = df.index.get_level_values("id").unique()

        # Build new MultiIndex
        new_index = pd.MultiIndex.from_product(
            [pd.to_datetime(target_dates), ids],
            names=["as_of_date", "id"]
        )

        # Reindex to target calendar
        result = df.reindex(new_index)

        # Apply fill method
        result = self._fill_pandas(result, fill_method)

        return result

    def _apply_polars(
        self,
        df: Any,
        calendar: "Calendar",
        fill_method: str | float | int | None,
    ) -> Any:
        """Apply alignment using polars."""
        import polars as pl

        if df.height == 0:
            return df

        # Get date range from input data
        dates = df.select("as_of_date").to_series()
        min_date = dates.min()
        max_date = dates.max()

        # Generate target calendar dates
        target_dates = list(calendar.dt_range(min_date, max_date))
        ids = df.select("id").unique().to_series().to_list()

        # Build target DataFrame with all date/id combinations
        target_records = [
            {"as_of_date": d, "id": i}
            for d in target_dates
            for i in ids
        ]
        target_df = pl.DataFrame(target_records)

        # Join source data to target index
        result = target_df.join(df, on=["as_of_date", "id"], how="left")

        # Apply fill method
        result = self._fill_polars(result, fill_method)

        return result

    def _fill_pandas(
        self,
        df: pd.DataFrame,
        fill_method: str | float | int | None,
    ) -> pd.DataFrame:
        """Apply fill method for pandas DataFrame."""
        if fill_method == "ffill":
            return df.groupby(level="id", group_keys=False).ffill()
        elif fill_method == "bfill":
            return df.groupby(level="id", group_keys=False).bfill()
        elif fill_method is not None:
            return df.fillna(fill_method)
        return df

    def _fill_polars(
        self,
        df: Any,
        fill_method: str | float | int | None,
    ) -> Any:
        """Apply fill method for polars DataFrame."""
        import polars as pl

        if fill_method == "ffill":
            # Forward fill per id
            return df.with_columns(
                pl.all().exclude(["as_of_date", "id"]).forward_fill().over("id")
            )
        elif fill_method == "bfill":
            # Backward fill per id
            return df.with_columns(
                pl.all().exclude(["as_of_date", "id"]).backward_fill().over("id")
            )
        elif fill_method is not None:
            return df.fill_null(fill_method)
        return df


class AlignTo(Operation):
    """Align Frame data to match another Frame's dates/ids.

    Reindexes the source Frame's data to match the target Frame's index
    (date/id combinations). Missing values can be filled using forward fill,
    backward fill, a scalar value, or left as NaN.

    This operation takes two inputs: the source frame and the target frame.
    Both are fetched during batch execution, and the target's actual data
    determines the output shape.

    Example:
        # Align signals data to match prices data's dates/ids
        aligned_signals = signals.align_to(prices, fill_method="ffill")
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        target: "Frame | Operation",
        fill_method: str | float | int | None = "ffill",
    ) -> None:
        """Initialize AlignTo operation.

        Args:
            frame: Source Frame or Operation to align.
            target: Target Frame or Operation whose dates/ids to align to.
            fill_method: How to fill missing values:
                - "ffill": Forward fill (use last available value)
                - "bfill": Backward fill (use next available value)
                - None: Leave as NaN/null
                - float/int: Fill with constant value
        """
        self._validate_fill_method(fill_method)
        # Pass both frames as inputs - they will be fetched concurrently
        super().__init__(frame, target, fill_method=fill_method)

    def _validate_fill_method(self, fill_method: str | float | int | None) -> None:
        """Validate the fill_method parameter."""
        if fill_method is None:
            return
        if isinstance(fill_method, (int, float)):
            return
        if isinstance(fill_method, str) and fill_method in ("ffill", "bfill"):
            return
        raise ValueError(
            f"fill_method must be 'ffill', 'bfill', None, or a scalar value, "
            f"got {fill_method!r}"
        )

    def _apply(
        self,
        inputs: list[Any],
        fill_method: str | float | int | None,
    ) -> Any:
        source_df = inputs[0]
        target_df = inputs[1]

        if _is_polars(source_df):
            return self._apply_polars(source_df, target_df, fill_method)
        else:
            return self._apply_pandas(source_df, target_df, fill_method)

    def _apply_pandas(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        fill_method: str | float | int | None,
    ) -> pd.DataFrame:
        """Apply alignment using pandas."""
        # Get target's index (dates/ids)
        target_index = target_df.index

        # Reindex source to match target's index
        result = source_df.reindex(target_index)

        # Apply fill method
        if fill_method == "ffill":
            result = result.groupby(level="id", group_keys=False).ffill()
        elif fill_method == "bfill":
            result = result.groupby(level="id", group_keys=False).bfill()
        elif fill_method is not None:
            result = result.fillna(fill_method)

        return result

    def _apply_polars(
        self,
        source_df: Any,
        target_df: Any,
        fill_method: str | float | int | None,
    ) -> Any:
        """Apply alignment using polars."""
        import polars as pl

        # Get target's date/id combinations
        target_index = target_df.select(["as_of_date", "id"]).unique()

        # Join source data to target index
        result = target_index.join(source_df, on=["as_of_date", "id"], how="left")

        # Apply fill method
        if fill_method == "ffill":
            result = result.with_columns(
                pl.all().exclude(["as_of_date", "id"]).forward_fill().over("id")
            )
        elif fill_method == "bfill":
            result = result.with_columns(
                pl.all().exclude(["as_of_date", "id"]).backward_fill().over("id")
            )
        elif fill_method is not None:
            result = result.fill_null(fill_method)

        return result
