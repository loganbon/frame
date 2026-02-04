"""Join operation for combining two Frames."""

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from frame.ops.base import Operation, _is_polars

if TYPE_CHECKING:
    from frame.core import Frame


class Join(Operation):
    """Join two Frames on specified keys."""

    VALID_JOIN_TYPES = ("inner", "left", "right", "outer")

    def __init__(
        self,
        left: "Frame | Operation",
        right: "Frame | Operation",
        on: str | list[str] | None = None,
        how: Literal["inner", "left", "right", "outer"] = "left",
        suffix: tuple[str, str] = ("_left", "_right"),
    ) -> None:
        self._validate_how(how)
        normalized_on = self._normalize_on(on)
        super().__init__(left, right, on=normalized_on, how=how, suffix=suffix)

    def _validate_how(self, how: str) -> None:
        if how not in self.VALID_JOIN_TYPES:
            raise ValueError(f"how must be one of {self.VALID_JOIN_TYPES}, got {how!r}")

    def _normalize_on(self, on: str | list[str] | None) -> list[str]:
        if on is None:
            return ["as_of_date", "id"]
        elif isinstance(on, str):
            return [on]
        return on

    def _apply(
        self,
        inputs: list[Any],
        on: list[str],
        how: str,
        suffix: tuple[str, str],
    ) -> Any:
        left_df, right_df = inputs[0], inputs[1]

        if _is_polars(left_df):
            return self._apply_polars(left_df, right_df, on, how, suffix)
        return self._apply_pandas(left_df, right_df, on, how, suffix)

    def _apply_pandas(self, left_df, right_df, on, how, suffix):
        # Reset index to make join keys available as columns
        left_reset = left_df.reset_index()
        right_reset = right_df.reset_index()

        result = pd.merge(left_reset, right_reset, on=on, how=how, suffixes=suffix)

        # Restore MultiIndex using join keys
        return result.set_index(on)

    def _apply_polars(self, left_df, right_df, on, how, suffix):
        polars_how = {"inner": "inner", "left": "left", "right": "right", "outer": "full"}
        return left_df.join(right_df, on=on, how=polars_how[how], suffix=suffix[1])


class AsofJoin(Operation):
    """Point-in-time (as-of) join on nearest prior date.

    Joins the left frame to the right frame by finding the most recent
    row in the right frame that is on or before the left frame's date.
    Useful for joining time-series data to slowly-changing data like
    fundamentals or reference data.
    """

    def __init__(
        self,
        left: "Frame | Operation",
        right: "Frame | Operation",
        on: str = "as_of_date",
        by: str | list[str] | None = "id",
        tolerance: str | None = None,
    ) -> None:
        """Initialize AsofJoin operation.

        Args:
            left: Left Frame to join.
            right: Right Frame to join.
            on: Column to match on (typically date column). Default "as_of_date".
            by: Column(s) to group by before matching. Default "id".
                If None, no grouping is applied.
            tolerance: Maximum time tolerance for matching (e.g., "7d").
                If None, uses nearest prior with no limit.
        """
        normalized_by = self._normalize_by(by)
        super().__init__(left, right, on=on, by=normalized_by, tolerance=tolerance)

    def _normalize_by(self, by: str | list[str] | None) -> list[str] | None:
        if by is None:
            return None
        elif isinstance(by, str):
            return [by]
        return by

    def _apply(
        self,
        inputs: list[Any],
        on: str,
        by: list[str] | None,
        tolerance: str | None,
    ) -> Any:
        left_df, right_df = inputs[0], inputs[1]

        if _is_polars(left_df):
            return self._apply_polars(left_df, right_df, on, by, tolerance)
        return self._apply_pandas(left_df, right_df, on, by, tolerance)

    def _apply_pandas(self, left_df, right_df, on, by, tolerance):
        # Reset index to get date column available
        left_reset = left_df.reset_index()
        right_reset = right_df.reset_index()

        # Ensure date columns are sorted
        left_reset = left_reset.sort_values(on)
        right_reset = right_reset.sort_values(on)

        # Parse tolerance if provided
        pd_tolerance = None
        if tolerance is not None:
            pd_tolerance = pd.Timedelta(tolerance)

        # Perform merge_asof
        result = pd.merge_asof(
            left_reset,
            right_reset,
            on=on,
            by=by,
            tolerance=pd_tolerance,
            direction="backward",
        )

        # Restore MultiIndex
        index_cols = [on]
        if by is not None:
            index_cols = [on] + by

        return result.set_index(index_cols)

    def _apply_polars(self, left_df, right_df, on, by, tolerance):
        import polars as pl

        # Ensure sorted by the join key
        left_sorted = left_df.sort(on)
        right_sorted = right_df.sort(on)

        # Polars join_asof
        strategy = "backward"

        if by is not None:
            result = left_sorted.join_asof(
                right_sorted,
                on=on,
                by=by,
                strategy=strategy,
                tolerance=tolerance,
            )
        else:
            result = left_sorted.join_asof(
                right_sorted,
                on=on,
                strategy=strategy,
                tolerance=tolerance,
            )

        return result
