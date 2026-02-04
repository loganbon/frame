"""Time series operations - transformations along the time dimension."""

from typing import TYPE_CHECKING, Any

from frame.ops.base import Operation, _is_polars

if TYPE_CHECKING:
    from frame.core import Frame


class Ewm(Operation):
    """Exponentially weighted moving operation (mean, std, var)."""

    def __init__(
        self,
        frame: "Frame | Operation",
        span: float,
        func: str = "mean",
    ) -> None:
        """Initialize Ewm operation.

        Args:
            frame: Input Frame or Operation.
            span: Decay factor as span (larger = slower decay).
            func: Function to apply - "mean", "std", or "var".
        """
        if func not in ("mean", "std", "var"):
            raise ValueError(f"func must be 'mean', 'std', or 'var', got {func}")
        super().__init__(frame, span=span, func=func)

    def _apply(self, inputs: list[Any], span: float, func: str) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            alpha = 2 / (span + 1)
            if func == "mean":
                return df.with_columns(pl.all().ewm_mean(alpha=alpha))
            elif func == "std":
                return df.with_columns(pl.all().ewm_std(alpha=alpha))
            elif func == "var":
                return df.with_columns(pl.all().ewm_var(alpha=alpha))
            else:
                raise ValueError(f"Unknown ewm function: {func}")
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            ewm = df.groupby(level="id", group_keys=False).apply(
                lambda g: getattr(g.ewm(span=span), func)()
            )
            return ewm


class Expanding(Operation):
    """Expanding window operation from start."""

    def __init__(
        self,
        frame: "Frame | Operation",
        func: str = "mean",
        min_periods: int = 1,
    ) -> None:
        """Initialize Expanding operation.

        Args:
            frame: Input Frame or Operation.
            func: Function to apply - "mean", "sum", "std", "min", "max", etc.
            min_periods: Minimum observations required.
        """
        super().__init__(frame, func=func, min_periods=min_periods)

    def _apply(self, inputs: list[Any], func: str, min_periods: int) -> Any:
        df = inputs[0]

        if _is_polars(df):
            # Polars doesn't have a direct expanding, use rolling with window=null
            # For now, collect and convert to pandas for expanding
            import polars as pl

            # Use cum_ operations where available
            if func == "sum":
                return df.with_columns(pl.all().cum_sum())
            elif func == "max":
                return df.with_columns(pl.all().cum_max())
            elif func == "min":
                return df.with_columns(pl.all().cum_min())
            elif func == "mean":
                # cumulative mean = cumsum / cumcount
                return df.with_columns(
                    pl.all().cum_sum() / pl.all().cum_count()
                )
            else:
                # For other functions, fall back to pandas
                pdf = df.to_pandas()
                result = pdf.groupby(level="id", group_keys=False).apply(
                    lambda g: getattr(g.expanding(min_periods=min_periods), func)()
                )
                return pl.from_pandas(result.reset_index())
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).apply(
                lambda g: getattr(g.expanding(min_periods=min_periods), func)()
            )


class Cumsum(Operation):
    """Cumulative sum operation."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize Cumsum operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all().cum_sum())
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).cumsum()


class Cumprod(Operation):
    """Cumulative product operation."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize Cumprod operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all().cum_prod())
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).cumprod()


class Cummax(Operation):
    """Cumulative maximum operation."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize Cummax operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all().cum_max())
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).cummax()


class Cummin(Operation):
    """Cumulative minimum operation."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize Cummin operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all().cum_min())
        else:
            # Pandas with MultiIndex [as_of_date, id] - group by id level
            return df.groupby(level="id", group_keys=False).cummin()


class Resample(Operation):
    """Resample to a different frequency (D->W, D->M, etc.)."""

    def __init__(
        self,
        frame: "Frame | Operation",
        freq: str,
        func: str = "last",
    ) -> None:
        """Initialize Resample operation.

        Args:
            frame: Input Frame or Operation.
            freq: Target frequency - "W" (weekly), "M" (monthly), "Q" (quarterly), etc.
            func: Aggregation function - "last", "first", "mean", "sum", "min", "max".
        """
        super().__init__(frame, freq=freq, func=func)

    def _apply(self, inputs: list[Any], freq: str, func: str) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            # Map frequency strings to polars format
            freq_map = {"W": "1w", "M": "1mo", "Q": "1q", "Y": "1y", "D": "1d"}
            pl_freq = freq_map.get(freq, freq)

            # Polars group_by_dynamic requires datetime column
            date_col = "as_of_date"
            if date_col not in df.columns:
                raise ValueError(f"DataFrame must have '{date_col}' column for resampling")

            # Get value columns (exclude date and id)
            id_col = "id" if "id" in df.columns else None
            value_cols = [c for c in df.columns if c not in (date_col, id_col)]

            # Build aggregation expressions based on func
            if func == "last":
                agg_exprs = [pl.col(c).last().alias(c) for c in value_cols]
            elif func == "first":
                agg_exprs = [pl.col(c).first().alias(c) for c in value_cols]
            elif func == "mean":
                agg_exprs = [pl.col(c).mean().alias(c) for c in value_cols]
            elif func == "sum":
                agg_exprs = [pl.col(c).sum().alias(c) for c in value_cols]
            elif func == "min":
                agg_exprs = [pl.col(c).min().alias(c) for c in value_cols]
            elif func == "max":
                agg_exprs = [pl.col(c).max().alias(c) for c in value_cols]
            else:
                raise ValueError(f"Unknown aggregation function: {func}")

            if id_col:
                return df.group_by_dynamic(date_col, every=pl_freq, by=id_col).agg(
                    agg_exprs
                )
            else:
                return df.group_by_dynamic(date_col, every=pl_freq).agg(agg_exprs)
        else:
            # Pandas with MultiIndex [as_of_date, id]
            # Reset index to work with resample
            df_reset = df.reset_index()

            def resample_group(group):
                group = group.set_index("as_of_date")
                resampled = getattr(group.resample(freq), func)()
                return resampled

            if "id" in df_reset.columns:
                result = df_reset.groupby("id", group_keys=False).apply(resample_group)
                # Restore MultiIndex
                result = result.reset_index()
                result = result.set_index(["as_of_date", "id"])
            else:
                result = resample_group(df_reset)

            return result
