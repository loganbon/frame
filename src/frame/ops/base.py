"""Base Operation class for Frame transformations."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from frame.executor import execute_with_batching, execute_with_batching_async, get_current_batch
from frame.mixins import APIMixin

if TYPE_CHECKING:
    from frame.core import Frame


class Operation(APIMixin):
    """Base class for Frame operations.

    Operations wrap one or more Frames and apply transformations.
    They are Frame-like objects that support get_range, get, aget_range, aget.

    Operations do NOT cache their results - input Frames already cache,
    and keeping operations stateless is simpler.
    """

    def __init__(self, *inputs: "Frame | Operation", **params: Any) -> None:
        """Initialize an Operation.

        Args:
            *inputs: Input Frames or Operations to transform.
            **params: Operation-specific parameters.
        """
        self._inputs = inputs
        self._params = params
        self._backend = inputs[0]._backend if inputs else None

    def get_range(self, start_dt: datetime, end_dt: datetime) -> Any:
        """Get transformed data for a date range.

        If called within a batching context, returns a LazyOperation proxy
        that will be resolved concurrently with other nested calls.

        Args:
            start_dt: Start datetime (inclusive).
            end_dt: End datetime (inclusive).

        Returns:
            DataFrame (or LazyOperation proxy if nested).
        """
        from frame.proxy import LazyOperation

        batch = get_current_batch()
        if batch is not None:
            return LazyOperation(self, start_dt, end_dt)

        return execute_with_batching(self._execute, start_dt, end_dt, {})

    def _execute(self, start_dt: datetime, end_dt: datetime) -> Any:
        """Execute the operation by collecting input data and applying transformation.

        This is called during batch resolution. Input Frames are fetched
        as LazyFrames/LazyOperations which are resolved before this returns.
        """
        # Get data from inputs - will be LazyFrames/LazyOperations in batch context
        lazy_inputs = [inp.get_range(start_dt, end_dt) for inp in self._inputs]
        # Resolve if needed (handles both lazy and already-resolved data)
        resolved = [
            li._resolve() if hasattr(li, "_resolve") else li for li in lazy_inputs
        ]
        return self._apply(resolved, **self._params)

    def _apply(self, inputs: list[Any], **params: Any) -> Any:
        """Apply the transformation to resolved input data.

        Override in subclasses to implement specific operations.

        Args:
            inputs: List of resolved DataFrames from input Frames/Operations.
            **params: Operation-specific parameters.

        Returns:
            Transformed DataFrame.
        """
        raise NotImplementedError("Subclasses must implement _apply")

    def get(self, dt: datetime) -> Any:
        """Get transformed data for a single date.

        Args:
            dt: The date to get data for.

        Returns:
            DataFrame for the single date.
        """
        end_dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        from frame.proxy import LazyOperation

        data = self.get_range(start_dt, end_dt)

        if isinstance(data, LazyOperation):
            data = data._resolve()

        return self._backend.drop_index_level(data, "as_of_date")

    async def aget_range(self, start_dt: datetime, end_dt: datetime) -> Any:
        """Async version of get_range.

        Args:
            start_dt: Start datetime (inclusive).
            end_dt: End datetime (inclusive).

        Returns:
            DataFrame for the date range.
        """
        return await execute_with_batching_async(self._execute, start_dt, end_dt, {})

    async def aget(self, dt: datetime) -> Any:
        """Async version of get.

        Args:
            dt: The date to get data for.

        Returns:
            DataFrame for the single date.
        """
        end_dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        data = await self.aget_range(start_dt, end_dt)
        return self._backend.drop_index_level(data, "as_of_date")

    def __repr__(self) -> str:
        """String representation."""
        inputs_repr = ", ".join(repr(inp) for inp in self._inputs)
        params_repr = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        parts = [inputs_repr]
        if params_repr:
            parts.append(params_repr)
        return f"{self.__class__.__name__}({', '.join(parts)})"
