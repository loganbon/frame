"""Mixin classes for shared Frame/Operation functionality."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frame.ops.binary import Add, Div, Mul, Pow, Sub
    from frame.ops.conversion import ToPandas, ToPolars
    from frame.ops.unary import (
        Abs,
        Clip,
        Diff,
        Fillna,
        Filter,
        Pct,
        Rolling,
        Select,
        Shift,
        Winsorize,
        Zscore,
    )


class APIMixin:
    """Mixin providing fluent API methods for Frame and Operation."""

    # Unary operations
    def rolling(
        self, window: int, func: str = "mean", min_periods: int | None = None
    ) -> "Rolling":
        """Apply rolling window operation.

        Args:
            window: Window size in rows.
            func: Rolling function ("mean", "sum", "std", "min", "max", etc.).
            min_periods: Minimum observations required. Defaults to window size.

        Returns:
            Rolling operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Rolling

        return Rolling(self, window=window, func=func, min_periods=min_periods)

    def shift(self, periods: int = 1) -> "Shift":
        """Apply shift/lag operation.

        Args:
            periods: Number of periods to shift. Positive shifts forward (lag),
                     negative shifts backward (lead).

        Returns:
            Shift operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Shift

        return Shift(self, periods=periods)

    def diff(self, periods: int = 1) -> "Diff":
        """Apply difference operation.

        Args:
            periods: Number of periods for difference calculation.

        Returns:
            Diff operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Diff

        return Diff(self, periods=periods)

    def abs(self) -> "Abs":
        """Apply absolute value operation.

        Returns:
            Abs operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Abs

        return Abs(self)

    def pct_change(self, periods: int = 1) -> "Pct":
        """Apply percentage change operation.

        Args:
            periods: Number of periods for percentage calculation.

        Returns:
            Pct operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Pct

        return Pct(self, periods=periods)

    def select(self, columns: list[str]) -> "Select":
        """Select specific columns.

        Args:
            columns: List of column names to select.

        Returns:
            Select operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Select

        return Select(self, columns=columns)

    def filter(self, filters: list[tuple]) -> "Filter":
        """Filter rows based on conditions.

        Args:
            filters: List of (column, operator, value) tuples.
                Operators: =, ==, !=, <, <=, >, >=, in, not in

        Returns:
            Filter operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Filter

        return Filter(self, filters=filters)

    def zscore(self, window: int, min_periods: int | None = None) -> "Zscore":
        """Apply rolling z-score (standardization) operation.

        Computes the z-score as (x - rolling_mean) / rolling_std for each value.

        Args:
            window: Window size for rolling mean and std calculation.
            min_periods: Minimum observations required. Defaults to window size.

        Returns:
            Zscore operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Zscore

        return Zscore(self, window=window, min_periods=min_periods)

    def clip(
        self, lower: float | None = None, upper: float | None = None
    ) -> "Clip":
        """Clip values to a specified range.

        Values outside the range [lower, upper] are set to the boundary values.

        Args:
            lower: Lower bound. Values below this are set to lower. None means no lower bound.
            upper: Upper bound. Values above this are set to upper. None means no upper bound.

        Returns:
            Clip operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Clip

        return Clip(self, lower=lower, upper=upper)

    def winsorize(self, lower: float = 0.01, upper: float = 0.99) -> "Winsorize":
        """Winsorize values to specified percentiles.

        Limits extreme values by capping them at the given percentiles.
        For example, winsorize(0.01, 0.99) caps values below the 1st percentile
        and above the 99th percentile.

        Args:
            lower: Lower percentile (0-1). Values below this percentile are capped.
            upper: Upper percentile (0-1). Values above this percentile are capped.

        Returns:
            Winsorize operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Winsorize

        return Winsorize(self, lower=lower, upper=upper)

    def fillna(
        self, value: float | None = None, method: str | None = None
    ) -> "Fillna":
        """Fill missing values.

        Can fill with a scalar value or use a method like forward fill or backward fill.

        Args:
            value: Scalar value to fill missing values with.
            method: Fill method - "ffill" (forward fill) or "bfill" (backward fill).
                Only one of value or method should be specified.

        Returns:
            Fillna operation wrapping this Frame/Operation.
        """
        from frame.ops.unary import Fillna

        return Fillna(self, value=value, method=method)

    # Binary operations
    def add(self, other: Any) -> "Add":
        """Add another Frame/Operation or scalar.

        Args:
            other: Right operand.

        Returns:
            Add operation.
        """
        from frame.ops.binary import Add

        return Add(self, other)

    def sub(self, other: Any) -> "Sub":
        """Subtract another Frame/Operation or scalar.

        Args:
            other: Right operand.

        Returns:
            Sub operation.
        """
        from frame.ops.binary import Sub

        return Sub(self, other)

    def mul(self, other: Any) -> "Mul":
        """Multiply by another Frame/Operation or scalar.

        Args:
            other: Right operand.

        Returns:
            Mul operation.
        """
        from frame.ops.binary import Mul

        return Mul(self, other)

    def div(self, other: Any) -> "Div":
        """Divide by another Frame/Operation or scalar.

        Args:
            other: Right operand (denominator).

        Returns:
            Div operation.
        """
        from frame.ops.binary import Div

        return Div(self, other)

    def pow(self, other: Any) -> "Pow":
        """Raise to a power.

        Args:
            other: Exponent (Frame, Operation, or scalar).

        Returns:
            Pow operation.
        """
        from frame.ops.binary import Pow

        return Pow(self, other)

    # Operator overloads
    def __add__(self, other: Any) -> "Add":
        """Add operator."""
        return self.add(other)

    def __radd__(self, other: Any) -> "Add":
        """Right add operator."""
        return self.add(other)

    def __sub__(self, other: Any) -> "Sub":
        """Subtract operator."""
        return self.sub(other)

    def __rsub__(self, other: Any) -> "Add":
        """Right subtract operator: other - self = -self + other."""
        return self.mul(-1).add(other)

    def __mul__(self, other: Any) -> "Mul":
        """Multiply operator."""
        return self.mul(other)

    def __rmul__(self, other: Any) -> "Mul":
        """Right multiply operator."""
        return self.mul(other)

    def __truediv__(self, other: Any) -> "Div":
        """Divide operator."""
        return self.div(other)

    def __neg__(self) -> "Mul":
        """Negation operator."""
        return self.mul(-1)

    def __pow__(self, other: Any) -> "Pow":
        """Power operator."""
        return self.pow(other)

    def __rpow__(self, other: Any) -> "Pow":
        """Right power operator: other ** self."""
        from frame.ops.binary import Pow

        # For rpow, we need to create a scalar raised to the power of self
        # This is tricky since we can't easily represent a scalar as a Frame
        # Instead, we'll implement it as: other ** self
        # But this requires the scalar to be the base, not the exponent
        # For now, raise NotImplementedError for scalar ** Frame
        raise NotImplementedError(
            "Scalar ** Frame/Operation is not supported. Use frame.pow(scalar) instead."
        )

    # Backend conversion
    def to_backend(self, backend: str) -> "ToPandas | ToPolars":
        """Return a conversion operation that outputs data in the specified backend format.

        This is a convenience method that wraps the Frame/Operation with a ToPandas or ToPolars
        operation, allowing easy backend switching while preserving lazy evaluation.

        Args:
            backend: Target backend - "pandas" or "polars"

        Returns:
            A ToPandas or ToPolars operation wrapping this Frame/Operation

        Example:
            polars_frame = Frame(fetch_func, backend="polars")
            pandas_view = polars_frame.to_backend("pandas")
            df = pandas_view.get_range(start, end)  # Returns pandas.DataFrame
        """
        from frame.ops.conversion import ToPandas, ToPolars

        if backend == "pandas":
            return ToPandas(self)
        elif backend == "polars":
            return ToPolars(self)
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'pandas' or 'polars'.")
