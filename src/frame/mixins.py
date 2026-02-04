"""Mixin classes for shared Frame/Operation functionality."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frame.calendar import Calendar
    from frame.core import Frame


class APIMixin:
    """Mixin providing fluent API methods for Frame and Operation."""

    # Unary operations
    def rolling(
        self, window: int, func: str = "mean", min_periods: int | None = None
    ) -> "Frame":
        """Apply rolling window operation.

        Args:
            window: Window size in rows.
            func: Rolling function ("mean", "sum", "std", "min", "max", etc.).
            min_periods: Minimum observations required. Defaults to window size.

        Returns:
            Frame with rolling window transformation applied.
        """
        from frame.ops.unary import Rolling

        return Rolling(self, window=window, func=func, min_periods=min_periods)

    def shift(self, periods: int = 1) -> "Frame":
        """Apply shift/lag operation.

        Args:
            periods: Number of periods to shift. Positive shifts forward (lag),
                     negative shifts backward (lead).

        Returns:
            Frame with shift transformation applied.
        """
        from frame.ops.unary import Shift

        return Shift(self, periods=periods)

    def dt_shift(self, periods: int = 1) -> "Frame":
        """Shift date range to return data from N periods back.

        Fetches data from N periods ago and re-labels it with the
        requested dates. Useful for getting lagged data without
        null values at the start of the range.

        Args:
            periods: Number of periods to shift backward. Positive values
                return past data, negative values return future data.

        Returns:
            Frame returning data shifted by N periods.
        """
        from frame.ops.dtshift import DtShift

        return DtShift(self, periods=periods)

    def diff(self, periods: int = 1) -> "Frame":
        """Apply difference operation.

        Args:
            periods: Number of periods for difference calculation.

        Returns:
            Frame with difference transformation applied.
        """
        from frame.ops.unary import Diff

        return Diff(self, periods=periods)

    def abs(self) -> "Frame":
        """Apply absolute value operation.

        Returns:
            Frame with absolute value transformation applied.
        """
        from frame.ops.unary import Abs

        return Abs(self)

    def pct_change(self, periods: int = 1) -> "Frame":
        """Apply percentage change operation.

        Args:
            periods: Number of periods for percentage calculation.

        Returns:
            Frame with percentage change transformation applied.
        """
        from frame.ops.unary import Pct

        return Pct(self, periods=periods)

    def select(self, columns: list[str]) -> "Frame":
        """Select specific columns.

        Args:
            columns: List of column names to select.

        Returns:
            Frame with column selection applied.
        """
        from frame.ops.unary import Select

        return Select(self, columns=columns)

    def filter(self, filters: list[tuple]) -> "Frame":
        """Filter rows based on conditions.

        Args:
            filters: List of (column, operator, value) tuples.
                Operators: =, ==, !=, <, <=, >, >=, in, not in

        Returns:
            Frame with row filtering applied.
        """
        from frame.ops.unary import Filter

        return Filter(self, filters=filters)

    def zscore(self, window: int, min_periods: int | None = None) -> "Frame":
        """Apply rolling z-score (standardization) operation.

        Computes the z-score as (x - rolling_mean) / rolling_std for each value.

        Args:
            window: Window size for rolling mean and std calculation.
            min_periods: Minimum observations required. Defaults to window size.

        Returns:
            Frame with z-score transformation applied.
        """
        from frame.ops.unary import Zscore

        return Zscore(self, window=window, min_periods=min_periods)

    def clip(
        self, lower: float | None = None, upper: float | None = None
    ) -> "Frame":
        """Clip values to a specified range.

        Values outside the range [lower, upper] are set to the boundary values.

        Args:
            lower: Lower bound. Values below this are set to lower. None means no lower bound.
            upper: Upper bound. Values above this are set to upper. None means no upper bound.

        Returns:
            Frame with clipping applied.
        """
        from frame.ops.unary import Clip

        return Clip(self, lower=lower, upper=upper)

    def winsorize(self, lower: float = 0.01, upper: float = 0.99) -> "Frame":
        """Winsorize values to specified percentiles.

        Limits extreme values by capping them at the given percentiles.
        For example, winsorize(0.01, 0.99) caps values below the 1st percentile
        and above the 99th percentile.

        Args:
            lower: Lower percentile (0-1). Values below this percentile are capped.
            upper: Upper percentile (0-1). Values above this percentile are capped.

        Returns:
            Frame with winsorization applied.
        """
        from frame.ops.unary import Winsorize

        return Winsorize(self, lower=lower, upper=upper)

    def fillna(
        self, value: float | None = None, method: str | None = None
    ) -> "Frame":
        """Fill missing values.

        Can fill with a scalar value or use a method like forward fill or backward fill.

        Args:
            value: Scalar value to fill missing values with.
            method: Fill method - "ffill" (forward fill) or "bfill" (backward fill).
                Only one of value or method should be specified.

        Returns:
            Frame with missing values filled.
        """
        from frame.ops.unary import Fillna

        return Fillna(self, value=value, method=method)

    # Alignment operations
    def align_to_calendar(
        self,
        calendar: "Calendar",
        fill_method: str | float | int | None = "ffill",
    ) -> "Frame":
        """Align data to a target calendar.

        Reindexes the data to match dates from the target calendar, using the
        input data's date range (min/max dates). Missing values can be filled
        using forward fill, backward fill, a scalar value, or left as NaN.

        Args:
            calendar: Target calendar to align to.
            fill_method: How to fill missing values:
                - "ffill": Forward fill (use last available value)
                - "bfill": Backward fill (use next available value)
                - None: Leave as NaN/null
                - float/int: Fill with constant value

        Returns:
            Frame aligned to the target calendar.
        """
        from frame.ops.align import AlignToCalendar

        return AlignToCalendar(self, calendar=calendar, fill_method=fill_method)

    def align_to(
        self,
        target: "Frame",
        fill_method: str | float | int | None = "ffill",
    ) -> "Frame":
        """Align data to match another Frame's dates/ids.

        Reindexes the data to match the target Frame's index (date/id
        combinations). Missing values can be filled using forward fill,
        backward fill, a scalar value, or left as NaN.

        Args:
            target: Target Frame whose dates/ids to align to.
            fill_method: How to fill missing values:
                - "ffill": Forward fill (use last available value)
                - "bfill": Backward fill (use next available value)
                - None: Leave as NaN/null
                - float/int: Fill with constant value

        Returns:
            Frame aligned to the target Frame's dates/ids.
        """
        from frame.ops.align import AlignTo

        return AlignTo(self, target, fill_method=fill_method)

    # Binary operations
    def add(self, other: Any) -> "Frame":
        """Add another Frame or scalar.

        Args:
            other: Right operand.

        Returns:
            Frame with addition applied.
        """
        from frame.ops.binary import Add

        return Add(self, other)

    def sub(self, other: Any) -> "Frame":
        """Subtract another Frame or scalar.

        Args:
            other: Right operand.

        Returns:
            Frame with subtraction applied.
        """
        from frame.ops.binary import Sub

        return Sub(self, other)

    def mul(self, other: Any) -> "Frame":
        """Multiply by another Frame or scalar.

        Args:
            other: Right operand.

        Returns:
            Frame with multiplication applied.
        """
        from frame.ops.binary import Mul

        return Mul(self, other)

    def div(self, other: Any) -> "Frame":
        """Divide by another Frame or scalar.

        Args:
            other: Right operand (denominator).

        Returns:
            Frame with division applied.
        """
        from frame.ops.binary import Div

        return Div(self, other)

    def pow(self, other: Any) -> "Frame":
        """Raise to a power.

        Args:
            other: Exponent (Frame or scalar).

        Returns:
            Frame with power operation applied.
        """
        from frame.ops.binary import Pow

        return Pow(self, other)

    # Operator overloads
    def __add__(self, other: Any) -> "Frame":
        """Add operator."""
        return self.add(other)

    def __radd__(self, other: Any) -> "Frame":
        """Right add operator."""
        return self.add(other)

    def __sub__(self, other: Any) -> "Frame":
        """Subtract operator."""
        return self.sub(other)

    def __rsub__(self, other: Any) -> "Frame":
        """Right subtract operator: other - self = -self + other."""
        return self.mul(-1).add(other)

    def __mul__(self, other: Any) -> "Frame":
        """Multiply operator."""
        return self.mul(other)

    def __rmul__(self, other: Any) -> "Frame":
        """Right multiply operator."""
        return self.mul(other)

    def __truediv__(self, other: Any) -> "Frame":
        """Divide operator."""
        return self.div(other)

    def __neg__(self) -> "Frame":
        """Negation operator."""
        return self.mul(-1)

    def __pow__(self, other: Any) -> "Frame":
        """Power operator."""
        return self.pow(other)

    def __rpow__(self, other: Any) -> "Frame":
        """Right power operator: other ** self."""
        # For rpow, we need to create a scalar raised to the power of self
        # This is tricky since we can't easily represent a scalar as a Frame
        # Instead, we'll implement it as: other ** self
        # But this requires the scalar to be the base, not the exponent
        # For now, raise NotImplementedError for scalar ** Frame
        raise NotImplementedError(
            "Scalar ** Frame is not supported. Use frame.pow(scalar) instead."
        )

    # Backend conversion
    def to_backend(self, backend: str) -> "Frame":
        """Return a conversion operation that outputs data in the specified backend format.

        This is a convenience method that wraps the Frame with a ToPandas or ToPolars
        operation, allowing easy backend switching while preserving lazy evaluation.

        Args:
            backend: Target backend - "pandas" or "polars"

        Returns:
            Frame that converts output to the specified backend format.

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
