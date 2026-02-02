"""Base Operation class for Frame transformations."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frame.cache import ChunkBy
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

    def _compute_cache_key(self) -> str:
        """Compute a stable hash key from operation definition."""
        key_data = {
            "operation": self.__class__.__name__,
            "operation_module": self.__class__.__module__,
            "params": self._serialize_params(self._params),
            "inputs": [self._get_input_cache_key(inp) for inp in self._inputs],
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _serialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serialize params to JSON-compatible format."""
        result = {}
        for key, value in params.items():
            if callable(value):
                result[key] = f"callable({value.__name__})"
            else:
                try:
                    json.dumps(value)
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = str(value)
        return result

    def _get_input_cache_key(self, inp: "Frame | Operation") -> str:
        """Get cache key for an input Frame or Operation."""
        if hasattr(inp, "_cache_key"):
            # Frame has _cache_key attribute
            return inp._cache_key
        elif isinstance(inp, Operation):
            # Recursively compute for nested operations
            return inp._compute_cache_key()
        else:
            return str(inp)

    def to_frame(
        self,
        name: str | None = None,
        cache_dir: Path | str | None = None,
        parent_cache_dirs: list[Path | str] | None = None,
        chunk_by: ChunkBy = "month",
        read_workers: int | None = None,
        fetch_workers: int | None = 1,
    ) -> "Frame":
        """Convert this Operation into a cached Frame.

        Creates a Frame that wraps this operation, enabling parquet caching
        of the operation's output. Useful for expensive operation pipelines
        that are queried repeatedly.

        Args:
            name: Optional name for the cache key. If None, auto-generates
                from operation definition (class, params, input cache keys).
            cache_dir: Directory for parquet cache. Defaults to config default.
            parent_cache_dirs: Additional read-only cache directories.
            chunk_by: Cache chunk size ("day", "week", "month", "year").
            read_workers: Concurrency for cache reads.
            fetch_workers: Concurrency for data fetches.

        Returns:
            A Frame that caches the operation's output.

        Example:
            # Create an operation pipeline
            pipeline = frame.rolling(window=20).shift(1).zscore()

            # Materialize to a cached Frame
            cached = pipeline.to_frame(name="rolling_zscore")

            # Now queries use disk cache
            data = cached.get_range(start, end)
        """
        from frame.core import Frame

        # Determine cache key
        cache_key = name if name is not None else self._compute_cache_key()

        # Determine backend name from backend class
        backend_name = "pandas"
        if self._backend is not None:
            backend_class = self._backend.__class__.__name__
            if "Polars" in backend_class:
                backend_name = "polars"

        # Create fetch function that executes this operation
        def fetch_operation(start_dt: datetime, end_dt: datetime) -> Any:
            return self.get_range(start_dt, end_dt)

        # Set function metadata for cache key generation
        fetch_operation.__name__ = f"materialized_{cache_key}"
        fetch_operation.__module__ = "frame.ops.materialized"

        return Frame(
            func=fetch_operation,
            kwargs={},
            backend=backend_name,
            cache_dir=cache_dir,
            parent_cache_dirs=parent_cache_dirs,
            chunk_by=chunk_by,
            read_workers=read_workers,
            fetch_workers=fetch_workers,
        )
