"""Main Frame class for temporal dataframe wrapper."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from frame.backends.pandas import PandasBackend
from frame.backends.polars import PolarsBackend
from frame.cache import CacheManager, CacheMode, ChunkBy
from frame.calendar import BDateCalendar, Calendar
from frame.config import get_default_cache_dir, get_default_parent_cache_dirs
from frame.executor import (
    execute_with_batching,
    execute_with_batching_async,
    get_current_batch,
)
from frame.logging import get_logger
from frame.mixins import APIMixin
from frame.proxy import LazyFrame


class Frame(APIMixin):
    """Temporal dataframe wrapper with caching and concurrent execution.

    Wraps a data-fetching function that takes start_dt and end_dt parameters,
    providing automatic parquet caching by date chunks and support for
    concurrent nested Frame resolution.
    """

    _is_source = True  # Class attribute: Frame is a data source

    def __init__(
        self,
        func: Callable[..., Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        backend: Literal["pandas", "polars"] = "pandas",
        cache_dir: Path | str | None = None,
        parent_cache_dirs: list[Path | str] | None = None,
        chunk_by: ChunkBy = "month",
        read_workers: int | None = None,
        fetch_workers: int | None = 1,
        calendar: Calendar | None = None,
    ) -> None:
        """Initialize a Frame.

        Args:
            func: Function that fetches data. Must accept start_dt and end_dt
                  as first two positional arguments, plus any kwargs. Can be None
                  for Operation subclasses which don't fetch data directly.
            kwargs: Additional keyword arguments to pass to func.
            backend: DataFrame backend to use ("pandas" or "polars").
            cache_dir: Directory for parquet cache. Defaults to .frame_cache/
            parent_cache_dirs: Additional read-only cache directories to check
                for missing data. Checked in order after primary cache.
            chunk_by: Granularity for cache chunks ("day", "week",
                "month", or "year"). Default is "month".
            read_workers: Concurrency for cache reads (parquet files).
                None (default) uses ThreadPoolExecutor default (high, good for I/O).
            fetch_workers: Concurrency for live data fetches.
                1 (default) is sequential, safe for rate-limited APIs.
                Set higher (2-4) for APIs that support parallel requests.
            calendar: Calendar for determining valid dates. Defaults to
                BDateCalendar (business dates, excludes weekends).
        """
        self._func = func
        self._kwargs = kwargs or {}

        # If no func provided, this is an Operation subclass - skip cache setup
        if func is None:
            return

        self._backend_name = backend
        self._chunk_by = chunk_by
        self._read_workers = read_workers
        self._fetch_workers = fetch_workers
        self._calendar = calendar or BDateCalendar()

        if backend == "pandas":
            self._backend = PandasBackend()
        elif backend == "polars":
            self._backend = PolarsBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if cache_dir is None:
            self._cache_dir = get_default_cache_dir()
        else:
            self._cache_dir = Path(cache_dir)

        # Convert parent cache dirs to Path objects, falling back to defaults
        if parent_cache_dirs is None:
            self._parent_cache_dirs = get_default_parent_cache_dirs()
        else:
            self._parent_cache_dirs = [Path(p) for p in parent_cache_dirs]

        self._cache = CacheManager(
            func=self._func,
            kwargs=self._kwargs,
            backend=self._backend,
            cache_dir=self._cache_dir,
            parent_cache_dirs=self._parent_cache_dirs,
            chunk_by=self._chunk_by,
            read_workers=self._read_workers,
            fetch_workers=self._fetch_workers,
            calendar=self._calendar,
        )
        self._cache_key = self._cache._cache_key
        self._log = get_logger(__name__).bind(
            frame_func=self._func.__name__, cache_key=self._cache_key
        )
        self._log.info(
            "frame_created", backend=self._backend_name, cache_dir=str(self._cache_dir)
        )

    def _fetch_data(
        self,
        start_dt: datetime,
        end_dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Fetch data using cache manager.

        Args:
            start_dt: Start datetime.
            end_dt: End datetime.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
            cache_mode: Cache behavior mode ("a", "l", "r", "w").
        """
        return self._cache.get_data(
            start_dt, end_dt, columns=columns, filters=filters, cache_mode=cache_mode
        )

    def get_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Get data for a date range.

        If called within a batching context (nested Frame call), returns
        a LazyFrame proxy that will be resolved concurrently with other
        nested calls.

        Args:
            start_dt: Start datetime (inclusive).
            end_dt: End datetime (inclusive).
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
                Operators: =, ==, !=, <, <=, >, >=, in, not in
            cache_mode: Cache behavior mode:
                - "a" (append): Read from cache, fetch missing, cache new data (default)
                - "l" (live): Bypass cache entirely, always fetch fresh
                - "r" (read): Only read from cache, raise CacheMissError if missing
                - "w" (write): Force refresh, always fetch and overwrite cache

        Returns:
            DataFrame (or LazyFrame proxy if nested).
        """
        self._log.info(
            "get_range_called",
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            cache_mode=cache_mode,
        )
        batch = get_current_batch()
        if batch is not None:
            self._log.debug("returning_lazy_frame")
            return LazyFrame(
                self,
                start_dt,
                end_dt,
                columns=columns,
                filters=filters,
                cache_mode=cache_mode,
            )

        return execute_with_batching(
            lambda s, e: self._fetch_data(
                s, e, columns=columns, filters=filters, cache_mode=cache_mode
            ),
            start_dt,
            end_dt,
            {},
        )

    def _execute_func(self, start_dt: datetime, end_dt: datetime) -> Any:
        """Execute the wrapped function with caching."""
        return self._fetch_data(start_dt, end_dt)

    def get(
        self,
        dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ) -> Any:
        """Get data for a single date.

        Returns data without as_of_date in the index (for pandas) or
        without as_of_date column (for polars).

        Args:
            dt: The date to fetch data for.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
                Operators: =, ==, !=, <, <=, >, >=, in, not in
            cache_mode: Cache behavior mode:
                - "a" (append): Read from cache, fetch missing, cache new data (default)
                - "l" (live): Bypass cache entirely, always fetch fresh
                - "r" (read): Only read from cache, raise CacheMissError if missing
                - "w" (write): Force refresh, always fetch and overwrite cache

        Returns:
            DataFrame for the single date.
        """
        end_dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        data = self.get_range(
            start_dt, end_dt, columns=columns, filters=filters, cache_mode=cache_mode
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
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
                Operators: =, ==, !=, <, <=, >, >=, in, not in
            cache_mode: Cache behavior mode:
                - "a" (append): Read from cache, fetch missing, cache new data (default)
                - "l" (live): Bypass cache entirely, always fetch fresh
                - "r" (read): Only read from cache, raise CacheMissError if missing
                - "w" (write): Force refresh, always fetch and overwrite cache

        Returns:
            DataFrame for the date range.
        """
        return await execute_with_batching_async(
            lambda s, e: self._fetch_data(
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
            dt: The date to fetch data for.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
                Operators: =, ==, !=, <, <=, >, >=, in, not in
            cache_mode: Cache behavior mode:
                - "a" (append): Read from cache, fetch missing, cache new data (default)
                - "l" (live): Bypass cache entirely, always fetch fresh
                - "r" (read): Only read from cache, raise CacheMissError if missing
                - "w" (write): Force refresh, always fetch and overwrite cache

        Returns:
            DataFrame for the single date.
        """
        end_dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        data = await self.aget_range(
            start_dt, end_dt, columns=columns, filters=filters, cache_mode=cache_mode
        )
        return self._backend.drop_index_level(data, "as_of_date")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Frame(func={self._func.__name__}, "
            f"backend={self._backend_name}, "
            f"cache_key={self._cache_key})"
        )

    @property
    def cache_key(self) -> str:
        """Return the cache key for this Frame."""
        return self._cache_key

    @staticmethod
    def concat(frames: list["Frame | APIMixin"]) -> "Frame":
        """Create a Concat operation that combines multiple frames.

        Returns a Frame that, when get_range is called, fetches all
        input frames in parallel (via batching) and concatenates the results.

        Args:
            frames: List of Frame objects to concatenate.

        Returns:
            Frame that concatenates the inputs.

        Example:
            prices = Frame(fetch_prices, {"ticker": "AAPL"})
            volumes = Frame(fetch_volumes, {"ticker": "AAPL"})

            # Create concat operation
            combined = Frame.concat([prices, volumes])

            # Fetch data (frames are fetched in parallel)
            result = combined.get_range(start, end)

            # Can be chained with other operations
            scaled = Frame.concat([prices, volumes]) * 2
            result = scaled.get_range(start, end)
        """
        from frame.ops.concat import Concat

        return Concat(*frames)
