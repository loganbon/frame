"""Parquet caching with hashing for Frame."""

import concurrent.futures
import hashlib
import json
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple

from frame.logging import get_logger
from frame.memory_cache import get_memory_cache
from frame.validation import validate_dataframe

if TYPE_CHECKING:
    from frame.backends.base import Backend
    from frame.calendar import Calendar

CacheMode = Literal["l", "a", "r", "w"]
ChunkBy = Literal["day", "week", "month", "year"]


class ChunkResult(NamedTuple):
    """Result from resolving a single chunk."""

    chunk_start: datetime
    chunk_end: datetime
    df: Any
    missing_dates: set[date]
    requested_dates: set[date]


class CacheMissError(Exception):
    """Raised when cache_mode='r' and requested data is not in cache."""
    pass


class CacheManager:
    """Manages chunked parquet caching for Frame data."""

    def __init__(
        self,
        func: Callable,
        kwargs: dict[str, Any],
        backend: "Backend",
        cache_dir: Path,
        parent_cache_dirs: list[Path] | None = None,
        chunk_by: ChunkBy = "month",
        read_workers: int | None = None,
        fetch_workers: int | None = 1,
        calendar: "Calendar | None" = None,
    ):
        from frame.calendar import BDateCalendar

        self._func = func
        self._kwargs = kwargs
        self._backend = backend
        self._cache_dir = cache_dir
        self._chunk_by = chunk_by
        self._parent_cache_dirs = parent_cache_dirs or []
        self._read_workers = read_workers  # None = ThreadPoolExecutor default (high)
        self._fetch_workers = fetch_workers  # Default 1 = sequential (safe for APIs)
        self._calendar = calendar or BDateCalendar()
        self._cache_key = self._compute_cache_key()
        self._chunk_dir = self._cache_dir / self._cache_key
        self._parent_chunk_dirs = [
            parent / self._cache_key for parent in self._parent_cache_dirs
        ]
        self._log = get_logger(__name__).bind(
            cache_key=self._cache_key,
            func_name=self._func.__name__,
        )
        self._log.info(
            "cache_manager_initialized",
            cache_dir=str(self._cache_dir),
            granularity=self._chunk_by,
            read_workers=self._read_workers,
            fetch_workers=self._fetch_workers,
        )

    def _compute_cache_key(self) -> str:
        """Compute a stable cache key as {name}/{cache_id}.

        The name is the function's __name__ attribute.
        The cache_id is a 16-char hash of the serialized kwargs.
        """
        if hasattr(self, "__class__") and self.__class__.__name__ != "Frame":
            name = self.__class__.__name__
        else:
            name = self._func.__name__

        # Hash only the kwargs (with Frames converted to their cache keys)
        kwargs_data = self._serialize_kwargs(self._kwargs)
        kwargs_str = json.dumps(kwargs_data, sort_keys=True, default=str)
        cache_id = hashlib.sha256(kwargs_str.encode()).hexdigest()[:16]

        return f"{name}/{cache_id}"

    def _serialize_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Serialize kwargs to JSON-compatible format."""
        result = {}
        for key, value in kwargs.items():
            if hasattr(value, "__class__") and value.__class__.__name__ == "Frame":
                result[key] = f"Frame({value._cache_key})"
            elif hasattr(value, "_cache_key"):
                # Operations and other objects with cache keys
                result[key] = f"{value.__class__.__name__}({value._cache_key})"
            elif callable(value):
                result[key] = f"callable({value.__name__})"
            else:
                try:
                    json.dumps(value)
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = str(value)
        return result

    def _get_chunk_start(self, dt: datetime) -> datetime:
        """Get the start datetime of the chunk containing the given date."""
        if self._chunk_by == "day":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif self._chunk_by == "week":
            # ISO week starts on Monday
            weekday = dt.weekday()
            start = dt - timedelta(days=weekday)
            return start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif self._chunk_by == "month":
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif self._chunk_by == "year":
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unknown chunk granularity: {self._chunk_by}")

    def _get_chunk_end(self, dt: datetime) -> datetime:
        """Get the end datetime of the chunk containing the given date."""
        if self._chunk_by == "day":
            return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif self._chunk_by == "week":
            # ISO week ends on Sunday
            weekday = dt.weekday()
            end = dt + timedelta(days=6 - weekday)
            return end.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif self._chunk_by == "month":
            last_day = monthrange(dt.year, dt.month)[1]
            return dt.replace(
                day=last_day, hour=23, minute=59, second=59, microsecond=999999
            )
        elif self._chunk_by == "year":
            return dt.replace(
                month=12, day=31, hour=23, minute=59, second=59, microsecond=999999
            )
        else:
            raise ValueError(f"Unknown chunk granularity: {self._chunk_by}")

    def get_chunk_ranges(
        self, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Split a date range into chunk-sized ranges based on granularity."""
        chunks = []
        current = self._get_chunk_start(start)

        while current <= end:
            chunk_end = self._get_chunk_end(current)
            chunks.append((current, chunk_end))

            # Move to next chunk
            if self._chunk_by == "day":
                current = current + timedelta(days=1)
            elif self._chunk_by == "week":
                current = current + timedelta(days=7)
            elif self._chunk_by == "month":
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1, day=1)
                else:
                    current = current.replace(month=current.month + 1, day=1)
            elif self._chunk_by == "year":
                current = current.replace(year=current.year + 1, month=1, day=1)

        return chunks

    def _chunk_path(self, dt: datetime, chunk_dir: Path | None = None) -> Path:
        """Get the parquet file path for a chunk containing the given date."""
        if chunk_dir is None:
            chunk_dir = self._chunk_dir
        if self._chunk_by == "day":
            return chunk_dir / f"{dt.year}/{dt.month:02d}/{dt.day:02d}.prq"
        elif self._chunk_by == "week":
            week = dt.isocalendar()[1]
            return chunk_dir / f"{dt.year}/W{week:02d}.prq"
        elif self._chunk_by == "month":
            return chunk_dir / f"{dt.year}/{dt.month:02d}.prq"
        elif self._chunk_by == "year":
            return chunk_dir / f"{dt.year}.prq"
        else:
            raise ValueError(f"Unknown chunk granularity: {self._chunk_by}")

    def _dates_in_range(self, start: datetime, end: datetime) -> set[date]:
        """Get all valid dates in a range as a set."""
        return set(self._calendar.dt_range(start, end))

    def _get_dates_in_df(self, df: Any) -> set[date]:
        """Extract unique dates from DataFrame's as_of_date column/index."""
        if self._backend.is_empty(df):
            return set()

        # Handle pandas (MultiIndex) vs polars (column)
        try:
            # Pandas path: as_of_date in index
            if hasattr(df, "index") and hasattr(df.index, "names"):
                if "as_of_date" in df.index.names:
                    dates = df.index.get_level_values("as_of_date")
                    return {d.date() if hasattr(d, "date") else d for d in dates.unique()}
            # Polars path: as_of_date as column
            if hasattr(df, "columns") and "as_of_date" in (
                df.columns if hasattr(df.columns, "__iter__") else []
            ):
                unique_dates = df["as_of_date"].unique()
                if hasattr(unique_dates, "to_list"):
                    unique_dates = unique_dates.to_list()
                return {d.date() if hasattr(d, "date") else d for d in unique_dates}
        except Exception:
            pass

        return set()

    def _filter_to_dates(self, df: Any, dates: set[date]) -> Any:
        """Filter DataFrame to only include rows with dates in the given set."""
        if self._backend.is_empty(df):
            return df

        # Handle pandas (MultiIndex)
        if hasattr(df, "index") and hasattr(df.index, "names"):
            if "as_of_date" in df.index.names:
                import pandas as pd

                idx_dates = df.index.get_level_values("as_of_date")
                mask = pd.Series(
                    [d.date() if hasattr(d, "date") else d for d in idx_dates]
                ).isin(dates)
                return df[mask.values]

        # Handle polars (column)
        if hasattr(df, "columns") and "as_of_date" in (
            df.columns if hasattr(df.columns, "__iter__") else []
        ):
            import polars as pl

            date_list = list(dates)
            return df.filter(pl.col("as_of_date").dt.date().is_in(date_list))

        return df

    def _filter_out_dates(self, df: Any, dates: set[date]) -> Any:
        """Filter DataFrame to exclude rows with dates in the given set."""
        if self._backend.is_empty(df) or not dates:
            return df

        # Handle pandas (MultiIndex)
        if hasattr(df, "index") and hasattr(df.index, "names"):
            if "as_of_date" in df.index.names:
                import pandas as pd

                idx_dates = df.index.get_level_values("as_of_date")
                mask = pd.Series(
                    [d.date() if hasattr(d, "date") else d for d in idx_dates]
                ).isin(dates)
                return df[~mask.values]

        # Handle polars (column)
        if hasattr(df, "columns") and "as_of_date" in (
            df.columns if hasattr(df.columns, "__iter__") else []
        ):
            import polars as pl

            date_list = list(dates)
            return df.filter(~pl.col("as_of_date").dt.date().is_in(date_list))

        return df

    def _resolve_chunk_hierarchically(
        self,
        chunk_start: datetime,
        chunk_end: datetime,
        requested_dates: set[date],
        columns: list[str] | None,
        filters: list[tuple] | None,
    ) -> tuple[Any, set[date]]:
        """
        Resolve chunk data across cache hierarchy, tracking which dates are found.

        Returns: (DataFrame with found data, set of still-missing dates)
        """
        frames = []
        remaining_dates = requested_dates.copy()

        # Check all cache levels: primary first, then parents
        cache_dirs = self._parent_chunk_dirs + [self._chunk_dir]

        for cache_dir in cache_dirs:
            chunk_path = self._chunk_path(chunk_start, chunk_dir=cache_dir)
            if not chunk_path.exists():
                continue

            level = "primary" if cache_dir == self._chunk_dir else "parent"

            # Read chunk once, then extract dates from loaded data
            df = self._backend.read_parquet(chunk_path, columns=columns, filters=filters)
            if self._backend.is_empty(df):
                self._log.debug("cache_miss_at_level", level=level)
                continue

            chunk_dates = self._get_dates_in_df(df)
            found_dates = remaining_dates & chunk_dates

            if found_dates:
                self._log.debug(
                    "cache_hit_at_level", level=level, found_count=len(found_dates)
                )
                df = self._filter_to_dates(df, found_dates)
                if not self._backend.is_empty(df):
                    frames.append(df)
                remaining_dates -= found_dates
            else:
                self._log.debug("cache_miss_at_level", level=level)

            if not remaining_dates:
                break  # All dates found

        if frames:
            result = self._backend.concat(frames)
        else:
            result = self._backend.empty()

        return result, remaining_dates

    def _fetch_and_cache_dates(
        self,
        chunk_start: datetime,
        chunk_end: datetime,
        dates: set[date],
    ) -> Any:
        """Fetch specific dates and merge into primary cache chunk."""
        if not dates:
            return self._backend.empty()

        # Fetch live data for the date range
        min_date = min(dates)
        max_date = max(dates)
        self._log.info(
            "fetching_live_data",
            date_count=len(dates),
            min_date=str(min_date),
            max_date=str(max_date),
        )
        data = self._func(
            datetime.combine(min_date, datetime.min.time()),
            datetime.combine(max_date, datetime.max.time().replace(microsecond=999999)),
            **self._kwargs,
        )

        if self._backend.is_empty(data):
            return data

        # Filter to only the dates we requested (in case func returns more)
        data = self._filter_to_dates(data, dates)

        # Merge with existing primary cache chunk
        # For each date in new data, we replace ALL rows for that date
        # This ensures each date's data comes from a single source
        chunk_path = self._chunk_path(chunk_start)
        memory_cache = get_memory_cache()
        new_dates = self._get_dates_in_df(data)

        if chunk_path.exists():
            try:
                existing = self._backend.read_parquet(chunk_path)
                if not self._backend.is_empty(existing):
                    # Remove all rows for dates that are in new data
                    existing_filtered = self._filter_out_dates(existing, new_dates)
                    if not self._backend.is_empty(existing_filtered):
                        merged = self._backend.concat([existing_filtered, data])
                    else:
                        merged = data
                    self._backend.to_parquet(merged, chunk_path)
                else:
                    self._backend.to_parquet(data, chunk_path)
            except Exception:
                self._backend.to_parquet(data, chunk_path)
        else:
            self._backend.to_parquet(data, chunk_path)

        # Invalidate memory cache entries for this path after write
        memory_cache.invalidate(chunk_path)

        self._log.debug("cache_chunk_written", chunk_path=str(chunk_path))
        return data

    def has_chunk(self, chunk_start: datetime, chunk_end: datetime) -> bool:
        """Check if a chunk exists in cache (primary or any parent)."""
        chunk_path = self._chunk_path(chunk_start)
        if chunk_path.exists():
            return True
        for parent_dir in self._parent_chunk_dirs:
            parent_path = self._chunk_path(chunk_start, chunk_dir=parent_dir)
            if parent_path.exists():
                return True
        return False

    def read_chunk(
        self,
        chunk_start: datetime,
        chunk_end: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ):
        """Read a cached chunk from disk (checking primary then parents).

        Args:
            chunk_start: Chunk start datetime.
            chunk_end: Chunk end datetime.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
        """
        memory_cache = get_memory_cache()

        # Check primary first
        chunk_path = self._chunk_path(chunk_start)
        if chunk_path.exists():
            # Try memory cache first (key includes columns/filters)
            df = memory_cache.get(chunk_path, columns=columns, filters=filters)
            if df is not None:
                self._log.debug("memory_cache_hit", path=str(chunk_path))
                return df

            # Memory miss - read from disk
            self._log.debug("reading_chunk", chunk_path=str(chunk_path), source="primary")
            df = self._backend.read_parquet(chunk_path, columns=columns, filters=filters)

            # Cache result (with columns/filters in key)
            if not self._backend.is_empty(df):
                memory_cache.put(chunk_path, df, columns=columns, filters=filters)

            return df

        # Check parents
        for parent_dir in self._parent_chunk_dirs:
            parent_path = self._chunk_path(chunk_start, chunk_dir=parent_dir)
            if parent_path.exists():
                # Try memory cache first
                df = memory_cache.get(parent_path, columns=columns, filters=filters)
                if df is not None:
                    self._log.debug("memory_cache_hit", path=str(parent_path))
                    return df

                self._log.debug("reading_chunk", chunk_path=str(parent_path), source="parent")
                df = self._backend.read_parquet(
                    parent_path, columns=columns, filters=filters
                )

                if not self._backend.is_empty(df):
                    memory_cache.put(parent_path, df, columns=columns, filters=filters)

                return df

        return self._backend.empty()

    def write_chunk(self, df, chunk_start: datetime, chunk_end: datetime) -> None:
        """Write a chunk to primary cache."""
        validate_dataframe(df)
        path = self._chunk_path(chunk_start)
        self._log.debug("writing_chunk", chunk_path=str(path))
        self._backend.to_parquet(df, path)
        # Invalidate ALL memory cache entries for this path (any columns/filters)
        get_memory_cache().invalidate(path)

    def fetch_and_cache_chunk(
        self,
        chunk_start: datetime,
        chunk_end: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ):
        """Fetch data for a chunk, cache it, then apply filters/columns.

        The cache always stores the full unfiltered data. Filters and column
        selection are applied to the result before returning.

        Args:
            chunk_start: Chunk start datetime.
            chunk_end: Chunk end datetime.
            columns: List of column names to select after caching.
            filters: List of (column, operator, value) tuples to apply after caching.
        """
        data = self._func(chunk_start, chunk_end, **self._kwargs)
        if not self._backend.is_empty(data):
            self.write_chunk(data, chunk_start, chunk_end)
            # Read back with filters/columns applied for efficiency
            return self.read_chunk(chunk_start, chunk_end, columns=columns, filters=filters)
        return data

    def _fetch_and_write_chunk(
        self,
        chunk_start: datetime,
        chunk_end: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ):
        """Fetch data for a chunk and overwrite cache (for write mode).

        Used by concurrent write mode to fetch and cache chunks in parallel.
        """
        self._log.info(
            "fetching_chunk_for_write",
            chunk_start=chunk_start.isoformat(),
            chunk_end=chunk_end.isoformat(),
        )
        data = self._func(chunk_start, chunk_end, **self._kwargs)
        if not self._backend.is_empty(data):
            self.write_chunk(data, chunk_start, chunk_end)
            return self.read_chunk(chunk_start, chunk_end, columns=columns, filters=filters)
        return data

    def get_data(
        self,
        start: datetime,
        end: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: CacheMode = "a",
    ):
        """Get data for a date range, using cache when available.

        Args:
            start: Start datetime.
            end: End datetime.
            columns: List of column names to read. If None, reads all columns.
            filters: List of (column, operator, value) tuples for row filtering.
            cache_mode: Cache behavior mode:
                - "a" (append): Read from cache, fetch missing, cache new data (default)
                - "l" (live): Bypass cache entirely, always fetch fresh
                - "r" (read): Only read from cache, raise CacheMissError if missing
                - "w" (write): Force refresh, always fetch and overwrite cache

        Returns:
            DataFrame with the requested data, optionally filtered.
        """
        self._log.info(
            "get_data_started",
            start=start.isoformat(),
            end=end.isoformat(),
            cache_mode=cache_mode,
        )
        # Handle live mode - bypass cache entirely
        if cache_mode == "l":
            data = self._func(start, end, **self._kwargs)
            if columns or filters:
                # Apply column selection and filters
                if not self._backend.is_empty(data):
                    # Write to temp, read back with filters (reuse parquet filter logic)
                    # For simplicity, just return the data and let caller filter
                    pass
            return self._backend.filter_date_range(data, start, end)

        chunks = self.get_chunk_ranges(start, end)
        self._log.debug("chunks_computed", chunk_count=len(chunks))

        # Single chunk: use sequential (no threading overhead)
        if len(chunks) == 1:
            return self._get_data_sequential(
                chunks, start, end, columns, filters, cache_mode
            )

        # Multiple chunks: use concurrent processing
        return self._get_data_concurrent(
            chunks, start, end, columns, filters, cache_mode
        )

    def _get_data_sequential(
        self,
        chunks: list[tuple[datetime, datetime]],
        start: datetime,
        end: datetime,
        columns: list[str] | None,
        filters: list[tuple] | None,
        cache_mode: CacheMode,
    ):
        """Original sequential chunk processing."""
        frames = []
        cache_hits = 0
        cache_misses = 0

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            self._log.debug("processing_chunk", chunk_index=i, cache_mode=cache_mode)
            effective_start = max(start, chunk_start)
            effective_end = min(end, chunk_end)
            requested_dates = self._dates_in_range(effective_start, effective_end)

            if cache_mode == "w":
                # Write mode: always fetch live and cache
                data = self._func(chunk_start, chunk_end, **self._kwargs)
                if not self._backend.is_empty(data):
                    self.write_chunk(data, chunk_start, chunk_end)
                    df = self.read_chunk(
                        chunk_start, chunk_end, columns=columns, filters=filters
                    )
                else:
                    df = data
            else:
                # Append or Read mode: try cache hierarchy first
                df, missing_dates = self._resolve_chunk_hierarchically(
                    chunk_start,
                    chunk_end,
                    requested_dates,
                    columns,
                    filters,
                )

                if missing_dates:
                    cache_misses += 1
                    if cache_mode == "r":
                        raise CacheMissError(
                            f"Missing dates {sorted(missing_dates)} not found in cache"
                        )
                    # Append mode: fetch missing and cache
                    self._fetch_and_cache_dates(chunk_start, chunk_end, missing_dates)
                    # Re-read from cache with filters/columns applied
                    # This ensures consistent filter application
                    live_df = self.read_chunk(
                        chunk_start, chunk_end, columns=columns, filters=filters
                    )
                    live_df = self._filter_to_dates(live_df, missing_dates)

                    if not self._backend.is_empty(live_df):
                        if not self._backend.is_empty(df):
                            df = self._backend.concat([df, live_df])
                        else:
                            df = live_df
                else:
                    cache_hits += 1

            if not self._backend.is_empty(df):
                frames.append(df)

        self._log.info(
            "sequential_complete", cache_hits=cache_hits, cache_misses=cache_misses
        )

        if not frames:
            return self._backend.empty()

        result = self._backend.concat(frames)
        return self._backend.filter_date_range(result, start, end)

    def _get_data_concurrent(
        self,
        chunks: list[tuple[datetime, datetime]],
        start: datetime,
        end: datetime,
        columns: list[str] | None,
        filters: list[tuple] | None,
        cache_mode: CacheMode,
    ):
        """Get data with concurrent chunk reading and fetching."""
        frames = []

        # Handle write mode: concurrent fetch and overwrite all chunks
        if cache_mode == "w":
            self._log.debug("concurrent_write_mode", chunk_count=len(chunks))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._fetch_workers
            ) as executor:
                futures = {}
                for chunk_start, chunk_end in chunks:
                    future = executor.submit(
                        self._fetch_and_write_chunk,
                        chunk_start,
                        chunk_end,
                        columns,
                        filters,
                    )
                    futures[future] = (chunk_start, chunk_end)

                for future in concurrent.futures.as_completed(futures):
                    df = future.result()
                    if not self._backend.is_empty(df):
                        frames.append(df)

            self._log.info("concurrent_write_complete", chunks_written=len(chunks))

            if not frames:
                return self._backend.empty()

            result = self._backend.concat(frames)
            result = self._backend.sort_by_date(result)
            return self._backend.filter_date_range(result, start, end)

        # Phase 1: Concurrent initial read of all chunks (uses read_workers)
        self._log.debug("concurrent_phase1_read", chunk_count=len(chunks))
        chunk_results: list[ChunkResult] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._read_workers
        ) as executor:
            futures = {}
            for chunk_start, chunk_end in chunks:
                effective_start = max(start, chunk_start)
                effective_end = min(end, chunk_end)
                future = executor.submit(
                    self._resolve_chunk_task,
                    chunk_start,
                    chunk_end,
                    effective_start,
                    effective_end,
                    columns,
                    filters,
                )
                futures[future] = (chunk_start, chunk_end)

            for future in concurrent.futures.as_completed(futures):
                chunk_results.append(future.result())

        # Phase 2: Handle missing dates
        chunks_needing_fetch = [r for r in chunk_results if r.missing_dates]
        self._log.debug(
            "concurrent_phase2_fetch", chunks_needing_fetch=len(chunks_needing_fetch)
        )

        if chunks_needing_fetch:
            if cache_mode == "r":
                all_missing: set[date] = set()
                for r in chunks_needing_fetch:
                    all_missing.update(r.missing_dates)
                raise CacheMissError(
                    f"Missing dates {sorted(all_missing)} not found in cache"
                )

            # Concurrent fetching with fetch_workers limit (respects API rate limits)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._fetch_workers
            ) as executor:
                fetch_futures = {}
                for result in chunks_needing_fetch:
                    future = executor.submit(
                        self._fetch_and_cache_dates,
                        result.chunk_start,
                        result.chunk_end,
                        result.missing_dates,
                    )
                    fetch_futures[future] = result

                # Wait for all fetches to complete
                for future in concurrent.futures.as_completed(fetch_futures):
                    future.result()  # Raise any exceptions

        # Phase 3: Concurrent final read (uses read_workers)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._read_workers
        ) as executor:
            final_futures = {}

            for result in chunk_results:
                if result.missing_dates:
                    # Re-read after fetching
                    future = executor.submit(
                        self._read_chunk_filtered,
                        result.chunk_start,
                        result.chunk_end,
                        result.requested_dates,
                        columns,
                        filters,
                    )
                    final_futures[future] = result
                else:
                    # Already have complete data
                    if not self._backend.is_empty(result.df):
                        frames.append(result.df)

            for future in concurrent.futures.as_completed(final_futures):
                df = future.result()
                if not self._backend.is_empty(df):
                    frames.append(df)

        cache_count = len(chunk_results) - len(chunks_needing_fetch)
        fetch_count = len(chunks_needing_fetch)
        self._log.info(
            "concurrent_complete", chunks_from_cache=cache_count, chunks_fetched=fetch_count
        )

        if not frames:
            return self._backend.empty()

        result = self._backend.concat(frames)
        result = self._backend.sort_by_date(result)
        return self._backend.filter_date_range(result, start, end)

    def _resolve_chunk_task(
        self,
        chunk_start: datetime,
        chunk_end: datetime,
        effective_start: datetime,
        effective_end: datetime,
        columns: list[str] | None,
        filters: list[tuple] | None,
    ) -> ChunkResult:
        """Task for concurrent chunk resolution."""
        requested_dates = self._dates_in_range(effective_start, effective_end)
        df, missing_dates = self._resolve_chunk_hierarchically(
            chunk_start, chunk_end, requested_dates, columns, filters
        )
        return ChunkResult(
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            df=df,
            missing_dates=missing_dates,
            requested_dates=requested_dates,
        )

    def _read_chunk_filtered(
        self,
        chunk_start: datetime,
        chunk_end: datetime,
        dates: set[date],
        columns: list[str] | None,
        filters: list[tuple] | None,
    ) -> Any:
        """Read a chunk and filter to specific dates."""
        df = self.read_chunk(chunk_start, chunk_end, columns=columns, filters=filters)
        return self._filter_to_dates(df, dates)
