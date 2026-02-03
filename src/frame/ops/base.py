"""Base Operation class for Frame transformations."""

from __future__ import annotations

import hashlib
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frame.cache import CacheMode, ChunkBy
from frame.core import Frame
from frame.executor import execute_with_batching, execute_with_batching_async, get_current_batch

if TYPE_CHECKING:
    pass


def _is_polars(df: Any) -> bool:
    """Check if df is a polars DataFrame."""
    return hasattr(df, "lazy") and hasattr(df, "collect")


def _is_pandas(df: Any) -> bool:
    """Check if DataFrame is a pandas DataFrame."""
    return isinstance(df, pd.DataFrame)


def _apply_filters(df: Any, filters: list[tuple]) -> Any:
    """Apply filter conditions to a DataFrame.

    Args:
        df: pandas or polars DataFrame.
        filters: List of (column, operator, value) tuples.

    Returns:
        Filtered DataFrame.
    """
    for col, op, val in filters:
        df = _apply_single_filter(df, col, op, val)
    return df


def _apply_single_filter(df: Any, col: str, op: str, val: Any) -> Any:
    """Apply a single filter condition to a DataFrame.

    Args:
        df: pandas or polars DataFrame.
        col: Column name (can be in index for pandas MultiIndex).
        op: Operator (=, ==, !=, <, <=, >, >=, in, not in).
        val: Value to compare against.

    Returns:
        Filtered DataFrame.
    """
    if _is_polars(df):
        from frame.backends.polars import _build_polars_filter

        expr = _build_polars_filter(col, op, val)
        return df.filter(expr)
    else:
        # Pandas - check if column is in index or columns
        if col in df.columns:
            col_values = df[col]
        elif hasattr(df.index, "names") and col in df.index.names:
            col_values = df.index.get_level_values(col)
        else:
            raise KeyError(f"Column '{col}' not found in columns or index")

        if op in ("=", "=="):
            return df[col_values == val]
        elif op == "!=":
            return df[col_values != val]
        elif op == "<":
            return df[col_values < val]
        elif op == "<=":
            return df[col_values <= val]
        elif op == ">":
            return df[col_values > val]
        elif op == ">=":
            return df[col_values >= val]
        elif op == "in":
            return df[col_values.isin(val)]
        elif op == "not in":
            return df[~col_values.isin(val)]
        else:
            raise ValueError(f"Unknown filter operator: {op}")


class Operation(Frame):
    """Base class for Frame transformations. Inherits from Frame.

    Operations wrap one or more Frames and apply transformations.
    They provide the same interface as Frame (get_range, get, aget_range, aget)
    and ARE Frames from the user's perspective.

    Operations do NOT cache their results - input Frames already cache,
    and keeping operations stateless is simpler.
    """

    _is_source = False  # Class attribute: Operation is a transformation, not a source

    def __init__(self, *inputs: "Frame", **params: Any) -> None:
        """Initialize an Operation.

        Args:
            *inputs: Input Frames or Operations to transform.
            **params: Operation-specific parameters.
        """
        super().__init__()  # Calls Frame.__init__ with func=None, skips cache setup
        self._inputs = inputs
        self._params = params
        self._backend = inputs[0]._backend if inputs else None

    def get_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Get transformed data for a date range.

        If called within a batching context, returns a LazyFrame proxy
        that will be resolved concurrently with other nested calls.

        Args:
            start_dt: Start datetime (inclusive).
            end_dt: End datetime (inclusive).
            columns: List of column names to select from the operation's output.
                If None, returns all columns.
            filters: List of (column, operator, value) tuples for row filtering
                on the operation's output.
            cache_mode: Cache mode to pass to dependent frames.

        Returns:
            DataFrame (or LazyFrame proxy if nested).
        """
        from frame.proxy import LazyFrame

        batch = get_current_batch()
        if batch is not None:
            return LazyFrame(
                self, start_dt, end_dt,
                columns=columns, filters=filters, cache_mode=cache_mode,
            )

        return execute_with_batching(
            lambda s, e: self._execute(
                s, e, columns=columns, filters=filters, cache_mode=cache_mode
            ),
            start_dt,
            end_dt,
            {},
        )

    def _execute(
        self,
        start_dt: datetime,
        end_dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Execute the operation by collecting input data and applying transformation.

        This is called during batch resolution. Input Frames are fetched
        as LazyFrames which are resolved before this returns.

        Args:
            start_dt: Start datetime (inclusive).
            end_dt: End datetime (inclusive).
            columns: List of column names to select from output.
            filters: List of filter tuples to apply to output.
            cache_mode: Cache mode to pass to input frames.
        """
        # Get data from inputs - will be LazyFrames in batch context
        # Note: columns/filters are NOT passed to inputs - they apply to output
        lazy_inputs = [
            inp.get_range(start_dt, end_dt, cache_mode=cache_mode)
            for inp in self._inputs
        ]
        # Resolve if needed (handles both lazy and already-resolved data)
        resolved = [
            li._resolve() if hasattr(li, "_resolve") else li for li in lazy_inputs
        ]
        result = self._apply(resolved, **self._params)

        # Apply columns selection to output (works for both pandas and polars)
        if columns is not None:
            result = result[columns]

        # Apply filters to output
        if filters is not None:
            result = _apply_filters(result, filters)

        return result

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

    def get(
        self,
        dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Get transformed data for a single date.

        Args:
            dt: The date to get data for.
            columns: List of column names to read from source frames.
            filters: List of filter tuples for source frames.
            cache_mode: Cache mode to pass to dependent frames.

        Returns:
            DataFrame for the single date.
        """
        end_dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        from frame.proxy import LazyFrame

        data = self.get_range(
            start_dt, end_dt,
            columns=columns, filters=filters, cache_mode=cache_mode,
        )

        if isinstance(data, LazyFrame):
            data = data._resolve()

        return self._backend.drop_index_level(data, "as_of_date")

    async def aget_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Async version of get_range.

        Args:
            start_dt: Start datetime (inclusive).
            end_dt: End datetime (inclusive).
            columns: List of column names to read from source frames.
            filters: List of filter tuples for source frames.
            cache_mode: Cache mode to pass to dependent frames.

        Returns:
            DataFrame for the date range.
        """
        return await execute_with_batching_async(
            lambda s, e: self._execute(
                s, e, columns=columns, filters=filters, cache_mode=cache_mode
            ),
            start_dt,
            end_dt,
            {},
        )

    async def aget(
        self,
        dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Async version of get.

        Args:
            dt: The date to get data for.
            columns: List of column names to read from source frames.
            filters: List of filter tuples for source frames.
            cache_mode: Cache mode to pass to dependent frames.

        Returns:
            DataFrame for the single date.
        """
        end_dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        data = await self.aget_range(
            start_dt, end_dt,
            columns=columns, filters=filters, cache_mode=cache_mode,
        )
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

    @property
    def _cache_key(self) -> str:
        """Stable cache key for this operation."""
        if not hasattr(self, "_cached_cache_key"):
            self._cached_cache_key = self._compute_cache_key()
        return self._cached_cache_key

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

    def _get_input_cache_key(self, inp: "Frame") -> str:
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
