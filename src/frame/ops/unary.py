"""Unary operations - single-input Frame transformations."""

from typing import TYPE_CHECKING, Any

import pandas as pd

from frame.ops.base import Operation

if TYPE_CHECKING:
    from frame.core import Frame


def _is_polars(df: Any) -> bool:
    """Check if DataFrame is a polars DataFrame."""
    return hasattr(df, "lazy")


def _is_pandas(df: Any) -> bool:
    """Check if DataFrame is a pandas DataFrame."""
    return isinstance(df, pd.DataFrame)


class Rolling(Operation):
    """Rolling window operation.

    Applies a rolling window function (mean, sum, std, etc.) over the data.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        window: int,
        func: str = "mean",
        min_periods: int | None = None,
    ) -> None:
        """Initialize Rolling operation.

        Args:
            frame: Input Frame or Operation.
            window: Window size in rows.
            func: Rolling function to apply ("mean", "sum", "std", "min", "max", etc.).
            min_periods: Minimum observations required. Defaults to window size.
        """
        super().__init__(frame, window=window, func=func, min_periods=min_periods)

    def _apply(
        self,
        inputs: list[Any],
        window: int,
        func: str,
        min_periods: int | None,
    ) -> Any:
        df = inputs[0]
        min_p = min_periods or window

        if _is_polars(df):
            import polars as pl

            return df.with_columns(
                pl.all().rolling_mean(window) if func == "mean"
                else pl.all().rolling_sum(window) if func == "sum"
                else pl.all().rolling_std(window) if func == "std"
                else pl.all().rolling_min(window) if func == "min"
                else pl.all().rolling_max(window) if func == "max"
                else pl.all()
            )
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            rolling = df.groupby(level="id", group_keys=False).rolling(
                window, min_periods=min_p
            )
            result = getattr(rolling, func)()
            return result.droplevel(0)


class Shift(Operation):
    """Shift/lag operation.

    Shifts data by a specified number of periods.
    """

    def __init__(self, frame: "Frame | Operation", periods: int = 1) -> None:
        """Initialize Shift operation.

        Args:
            frame: Input Frame or Operation.
            periods: Number of periods to shift. Positive shifts forward (lag),
                     negative shifts backward (lead).
        """
        super().__init__(frame, periods=periods)

    def _apply(self, inputs: list[Any], periods: int) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all().shift(periods))
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).shift(periods)


class Diff(Operation):
    """Difference operation.

    Computes the difference between current and previous values.
    """

    def __init__(self, frame: "Frame | Operation", periods: int = 1) -> None:
        """Initialize Diff operation.

        Args:
            frame: Input Frame or Operation.
            periods: Number of periods for difference calculation.
        """
        super().__init__(frame, periods=periods)

    def _apply(self, inputs: list[Any], periods: int) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all().diff(periods))
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).diff(periods)


class Abs(Operation):
    """Absolute value operation."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize Abs operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        return inputs[0].abs()


class Pct(Operation):
    """Percentage change operation.

    Computes the percentage change between current and previous values.
    """

    def __init__(self, frame: "Frame | Operation", periods: int = 1) -> None:
        """Initialize Pct operation.

        Args:
            frame: Input Frame or Operation.
            periods: Number of periods for percentage calculation.
        """
        super().__init__(frame, periods=periods)

    def _apply(self, inputs: list[Any], periods: int) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all().pct_change(periods))
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).pct_change(periods)


class Select(Operation):
    """Column selection operation."""

    def __init__(self, frame: "Frame | Operation", columns: list[str]) -> None:
        """Initialize Select operation.

        Args:
            frame: Input Frame or Operation.
            columns: List of column names to select.
        """
        super().__init__(frame, columns=columns)

    def _apply(self, inputs: list[Any], columns: list[str]) -> Any:
        df = inputs[0]
        # Works for both pandas and polars
        return df[columns]


class Zscore(Operation):
    """Rolling z-score (standardization) operation.

    Computes the z-score as (x - rolling_mean) / rolling_std for each value.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        window: int,
        min_periods: int | None = None,
    ) -> None:
        """Initialize Zscore operation.

        Args:
            frame: Input Frame or Operation.
            window: Window size for rolling mean and std calculation.
            min_periods: Minimum observations required. Defaults to window size.
        """
        super().__init__(frame, window=window, min_periods=min_periods)

    def _apply(
        self,
        inputs: list[Any],
        window: int,
        min_periods: int | None,
    ) -> Any:
        df = inputs[0]
        min_p = min_periods or window

        if _is_polars(df):
            import polars as pl

            return df.with_columns(
                (pl.all() - pl.all().rolling_mean(window))
                / pl.all().rolling_std(window)
            )
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            grouped = df.groupby(level="id", group_keys=False)
            rolling = grouped.rolling(window, min_periods=min_p)
            # groupby().rolling() adds an extra 'id' level at position 0, drop it
            mean = rolling.mean().droplevel(0)
            std = rolling.std().droplevel(0)
            return (df - mean) / std


class Clip(Operation):
    """Clip values to a specified range.

    Values outside the range [lower, upper] are set to the boundary values.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        """Initialize Clip operation.

        Args:
            frame: Input Frame or Operation.
            lower: Lower bound. Values below this are set to lower. None means no lower bound.
            upper: Upper bound. Values above this are set to upper. None means no upper bound.
        """
        super().__init__(frame, lower=lower, upper=upper)

    def _apply(
        self, inputs: list[Any], lower: float | None, upper: float | None
    ) -> Any:
        df = inputs[0]
        return df.clip(lower=lower, upper=upper)


class Winsorize(Operation):
    """Winsorize values to specified percentiles.

    Limits extreme values by capping them at the given percentiles.
    For example, winsorize(0.01, 0.99) caps values below the 1st percentile
    and above the 99th percentile.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> None:
        """Initialize Winsorize operation.

        Args:
            frame: Input Frame or Operation.
            lower: Lower percentile (0-1). Values below this percentile are capped.
            upper: Upper percentile (0-1). Values above this percentile are capped.
        """
        if not 0 <= lower < upper <= 1:
            raise ValueError(
                f"Percentiles must satisfy 0 <= lower < upper <= 1, got lower={lower}, upper={upper}"
            )
        super().__init__(frame, lower=lower, upper=upper)

    def _apply(self, inputs: list[Any], lower: float, upper: float) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            result = df.clone()
            for col in df.columns:
                if df[col].dtype in (
                    pl.Float32,
                    pl.Float64,
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                ):
                    lower_val = df[col].quantile(lower)
                    upper_val = df[col].quantile(upper)
                    result = result.with_columns(pl.col(col).clip(lower_val, upper_val))
            return result
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            result = df.copy()
            for col in df.select_dtypes(include=["number"]).columns:
                grouped = df.groupby(level="id")[col]
                lower_q = grouped.transform(lambda x: x.quantile(lower))
                upper_q = grouped.transform(lambda x: x.quantile(upper))
                result[col] = df[col].clip(lower=lower_q, upper=upper_q)
            return result


class Fillna(Operation):
    """Fill missing values.

    Can fill with a scalar value or use a method like forward fill or backward fill.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        value: float | None = None,
        method: str | None = None,
    ) -> None:
        """Initialize Fillna operation.

        Args:
            frame: Input Frame or Operation.
            value: Scalar value to fill missing values with.
            method: Fill method - "ffill" (forward fill) or "bfill" (backward fill).
                Only one of value or method should be specified.
        """
        if value is not None and method is not None:
            raise ValueError("Cannot specify both value and method")
        if value is None and method is None:
            raise ValueError("Must specify either value or method")
        if method is not None and method not in ("ffill", "bfill"):
            raise ValueError(f"method must be 'ffill' or 'bfill', got {method}")
        super().__init__(frame, value=value, method=method)

    def _apply(
        self, inputs: list[Any], value: float | None, method: str | None
    ) -> Any:
        df = inputs[0]

        if value is not None:
            return df.fillna(value)

        if _is_polars(df):
            if method == "ffill":
                return df.fill_null(strategy="forward")
            elif method == "bfill":
                return df.fill_null(strategy="backward")
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            if method == "ffill":
                return df.groupby(level="id", group_keys=False).ffill()
            elif method == "bfill":
                return df.groupby(level="id", group_keys=False).bfill()

        return df


class Filter(Operation):
    """Row filtering operation."""

    def __init__(self, frame: "Frame | Operation", filters: list[tuple]) -> None:
        """Initialize Filter operation.

        Args:
            frame: Input Frame or Operation.
            filters: List of (column, operator, value) tuples.
                Operators: =, ==, !=, <, <=, >, >=, in, not in
        """
        super().__init__(frame, filters=filters)

    def _apply(self, inputs: list[Any], filters: list[tuple]) -> Any:
        df = inputs[0]
        for col, op, val in filters:
            df = self._apply_filter(df, col, op, val)
        return df

    def _apply_filter(self, df: Any, col: str, op: str, val: Any) -> Any:
        """Apply a single filter condition.

        Detects the backend by checking for polars-specific attributes.
        """
        if _is_polars(df):
            from frame.backends.polars import _build_polars_filter

            expr = _build_polars_filter(col, op, val)
            return df.filter(expr)
        else:
            # Pandas
            if op in ("=", "=="):
                return df[df[col] == val]
            elif op == "!=":
                return df[df[col] != val]
            elif op == "<":
                return df[df[col] < val]
            elif op == "<=":
                return df[df[col] <= val]
            elif op == ">":
                return df[df[col] > val]
            elif op == ">=":
                return df[df[col] >= val]
            elif op == "in":
                return df[df[col].isin(val)]
            elif op == "not in":
                return df[~df[col].isin(val)]
            else:
                raise ValueError(f"Unknown filter operator: {op}")
