"""Concat operation for combining multiple Frames."""

from typing import TYPE_CHECKING, Any

from frame.ops.base import Operation

if TYPE_CHECKING:
    from frame.core import Frame


class Concat(Operation):
    """Operation that concatenates multiple Frames/Operations.

    When get_range is called, fetches all inputs in parallel (via batching)
    and concatenates the results.

    Example:
        combined = Concat(prices, volumes, benchmark)
        result = combined.get_range(start, end)

        # Or via Frame.concat:
        combined = Frame.concat([prices, volumes, benchmark])
        result = combined.get_range(start, end)
    """

    def __init__(self, *inputs: "Frame | Operation") -> None:
        """Initialize a Concat operation.

        Args:
            *inputs: Frames or Operations to concatenate.
        """
        if not inputs:
            raise ValueError("Concat requires at least one input")
        super().__init__(*inputs)

    def _apply(self, inputs: list[Any], **params: Any) -> Any:
        """Concatenate all resolved inputs."""
        # Filter out empty dataframes
        non_empty = [df for df in inputs if not self._backend.is_empty(df)]
        if not non_empty:
            return self._backend.empty()
        return self._backend.concat(non_empty)

    def __repr__(self) -> str:
        """String representation."""
        inputs_repr = ", ".join(repr(inp) for inp in self._inputs)
        return f"Concat({inputs_repr})"
