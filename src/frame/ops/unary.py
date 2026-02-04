"""Unary operations - single-input Frame transformations."""

from typing import TYPE_CHECKING, Any

from frame.ops.base import Operation, _is_polars, _apply_filters

if TYPE_CHECKING:
    from frame.core import Frame


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
        return _apply_filters(df, filters)


class Where(Operation):
    """Replace values where condition is False.

    Similar to pandas DataFrame.where() - keeps values where condition is True,
    replaces with `other` where condition is False.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        cond: "Frame | Operation",
        other: Any = None,
    ) -> None:
        """Initialize Where operation.

        Args:
            frame: Input Frame or Operation.
            cond: Boolean condition Frame/Operation. Same shape as frame.
            other: Value to use where condition is False. Default is NaN.
        """
        super().__init__(frame, cond, other=other)

    def _apply(self, inputs: list[Any], other: Any) -> Any:
        df, cond = inputs[0], inputs[1]

        if _is_polars(df):
            import polars as pl

            # For polars, use when/then/otherwise
            if other is None:
                other = float("nan")
            # Apply condition column-wise
            result = df.clone()
            for col in df.columns:
                if col in cond.columns:
                    result = result.with_columns(
                        pl.when(pl.col(col).is_in(cond[col]))
                        .then(pl.col(col))
                        .otherwise(pl.lit(other))
                        .alias(col)
                    )
            return result
        else:
            # Pandas where
            return df.where(cond, other=other)


class Mask(Operation):
    """Replace values where condition is True.

    Similar to pandas DataFrame.mask() - replaces values where condition is True,
    keeps original values where condition is False.
    """

    def __init__(
        self,
        frame: "Frame | Operation",
        cond: "Frame | Operation",
        other: Any = None,
    ) -> None:
        """Initialize Mask operation.

        Args:
            frame: Input Frame or Operation.
            cond: Boolean condition Frame/Operation. Same shape as frame.
            other: Value to use where condition is True. Default is NaN.
        """
        super().__init__(frame, cond, other=other)

    def _apply(self, inputs: list[Any], other: Any) -> Any:
        df, cond = inputs[0], inputs[1]

        if _is_polars(df):
            import polars as pl

            # For polars, use when/then/otherwise (inverted from Where)
            if other is None:
                other = float("nan")
            result = df.clone()
            for col in df.columns:
                if col in cond.columns:
                    result = result.with_columns(
                        pl.when(pl.col(col).is_in(cond[col]))
                        .then(pl.lit(other))
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
            return result
        else:
            # Pandas mask
            return df.mask(cond, other=other)


class Dropna(Operation):
    """Remove rows with missing values."""

    def __init__(
        self,
        frame: "Frame | Operation",
        how: str = "any",
    ) -> None:
        """Initialize Dropna operation.

        Args:
            frame: Input Frame or Operation.
            how: Determine when to drop row:
                - "any": Drop row if any value is NA
                - "all": Drop row if all values are NA
        """
        if how not in ("any", "all"):
            raise ValueError(f"how must be 'any' or 'all', got {how}")
        super().__init__(frame, how=how)

    def _apply(self, inputs: list[Any], how: str) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            if how == "any":
                # Drop rows where any column has null
                mask = pl.lit(True)
                for col in df.columns:
                    mask = mask & pl.col(col).is_not_null()
                return df.filter(mask)
            else:
                # Drop rows where all columns are null
                mask = pl.lit(False)
                for col in df.columns:
                    mask = mask | pl.col(col).is_not_null()
                return df.filter(mask)
        else:
            # Pandas dropna
            return df.dropna(how=how)


class Rename(Operation):
    """Rename columns."""

    def __init__(
        self,
        frame: "Frame | Operation",
        mapping: dict[str, str],
    ) -> None:
        """Initialize Rename operation.

        Args:
            frame: Input Frame or Operation.
            mapping: Dictionary mapping old column names to new names.
        """
        super().__init__(frame, mapping=mapping)

    def _apply(self, inputs: list[Any], mapping: dict[str, str]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            return df.rename(mapping)
        else:
            # Pandas rename
            return df.rename(columns=mapping)


class Apply(Operation):
    """Apply a custom function to the DataFrame."""

    def __init__(
        self,
        frame: "Frame | Operation",
        func: Any,
    ) -> None:
        """Initialize Apply operation.

        Args:
            frame: Input Frame or Operation.
            func: Function to apply to the DataFrame. Should take a DataFrame
                and return a DataFrame.
        """
        super().__init__(frame, func=func)

    def _apply(self, inputs: list[Any], func: Any) -> Any:
        df = inputs[0]
        return func(df)
