"""Cross-sectional operations - transformations across ids per date."""

from typing import TYPE_CHECKING, Any

from frame.ops.base import Operation, _is_polars

if TYPE_CHECKING:
    from frame.core import Frame


class CsRank(Operation):
    """Cross-sectional percentile rank (0-1) across ids per date."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize CsRank operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            # Group by date and compute percentile rank within each group
            return df.with_columns(
                pl.all().rank() / pl.all().count()
            )
        else:
            # Pandas with MultiIndex [as_of_date, id]
            # Group by as_of_date and rank within each date
            return df.groupby(level="as_of_date", group_keys=False).transform(
                lambda x: x.rank(pct=True)
            )


class CsZscore(Operation):
    """Cross-sectional z-score (standardize across ids per date)."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize CsZscore operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            # Group by date and compute z-score within each group
            return df.with_columns(
                (pl.all() - pl.all().mean()) / pl.all().std()
            )
        else:
            # Pandas with MultiIndex [as_of_date, id]
            def zscore(x):
                return (x - x.mean()) / x.std()

            return df.groupby(level="as_of_date", group_keys=False).transform(zscore)


class CsDemean(Operation):
    """Cross-sectional demean (subtract mean across ids per date)."""

    def __init__(self, frame: "Frame | Operation") -> None:
        """Initialize CsDemean operation.

        Args:
            frame: Input Frame or Operation.
        """
        super().__init__(frame)

    def _apply(self, inputs: list[Any]) -> Any:
        df = inputs[0]

        if _is_polars(df):
            import polars as pl

            return df.with_columns(pl.all() - pl.all().mean())
        else:
            # Pandas with MultiIndex [as_of_date, id]
            return df.groupby(level="as_of_date", group_keys=False).transform(
                lambda x: x - x.mean()
            )


class CsWinsorize(Operation):
    """Cross-sectional winsorization (clip to percentiles across ids per date)."""

    def __init__(
        self,
        frame: "Frame | Operation",
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> None:
        """Initialize CsWinsorize operation.

        Args:
            frame: Input Frame or Operation.
            lower: Lower percentile (0-1). Values below are capped.
            upper: Upper percentile (0-1). Values above are capped.
        """
        if not 0 <= lower < upper <= 1:
            raise ValueError(
                f"Percentiles must satisfy 0 <= lower < upper <= 1, "
                f"got lower={lower}, upper={upper}"
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
            # Pandas with MultiIndex [as_of_date, id]
            result = df.copy()

            def winsorize_group(group):
                for col in group.select_dtypes(include=["number"]).columns:
                    lower_val = group[col].quantile(lower)
                    upper_val = group[col].quantile(upper)
                    group[col] = group[col].clip(lower=lower_val, upper=upper_val)
                return group

            return df.groupby(level="as_of_date", group_keys=False).apply(
                winsorize_group
            )
