"""In-memory cache layer for Frame chunk system."""

import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from frame.logging import get_logger


class CacheKey(NamedTuple):
    """Composite cache key including path, columns, and filters."""

    path: Path
    columns: tuple[str, ...] | None  # Sorted tuple for consistency
    filters: tuple[tuple, ...] | None  # Tuple of filter tuples

    @staticmethod
    def _make_hashable(value: Any) -> Any:
        """Recursively convert lists to tuples for hashability."""
        if isinstance(value, list):
            return tuple(CacheKey._make_hashable(v) for v in value)
        return value

    @classmethod
    def make(
        cls,
        path: Path,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> "CacheKey":
        hashable_filters = None
        if filters:
            hashable_filters = tuple(
                tuple(cls._make_hashable(v) for v in f) for f in filters
            )
        return cls(
            path=path,
            columns=tuple(sorted(columns)) if columns else None,
            filters=hashable_filters,
        )


@dataclass
class CacheConfig:
    enabled: bool = True
    max_entries: int = 128  # 0 = unlimited
    max_memory_bytes: int = 0  # 0 = unlimited
    track_stats: bool = True


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    current_entries: int = 0
    current_memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


@dataclass
class CacheEntry:
    df: Any
    size_bytes: int
    path: Path  # Track path for invalidation
    access_count: int = 0


class MemoryCache:
    """Thread-safe LRU memory cache for DataFrame chunks.

    Cache keys are composite: (path, columns, filters).
    Invalidation by path removes ALL entries for that path.
    """

    def __init__(self, config: CacheConfig | None = None):
        self._config = config or CacheConfig()
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._path_to_keys: dict[Path, set[CacheKey]] = {}  # For path-based invalidation
        self._stats = CacheStats()
        self._global_lock = threading.Lock()
        self._path_locks: dict[Path, threading.RLock] = {}
        self._path_locks_lock = threading.Lock()
        self._log = get_logger(__name__)

    def _get_path_lock(self, path: Path) -> threading.RLock:
        with self._path_locks_lock:
            if path not in self._path_locks:
                self._path_locks[path] = threading.RLock()
            return self._path_locks[path]

    def _estimate_df_size(self, df: Any) -> int:
        try:
            if hasattr(df, "memory_usage"):  # pandas
                return int(df.memory_usage(deep=True).sum())
            if hasattr(df, "estimated_size"):  # polars
                return df.estimated_size()
        except Exception:
            pass
        return sys.getsizeof(df)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if limits exceeded. Must hold _global_lock."""
        while self._config.max_entries > 0 and len(self._cache) > self._config.max_entries:
            key, entry = self._cache.popitem(last=False)
            self._path_to_keys.get(entry.path, set()).discard(key)
            self._stats.evictions += 1
            self._stats.current_entries -= 1
            self._stats.current_memory_bytes -= entry.size_bytes

        while (
            self._config.max_memory_bytes > 0
            and self._stats.current_memory_bytes > self._config.max_memory_bytes
            and self._cache
        ):
            key, entry = self._cache.popitem(last=False)
            self._path_to_keys.get(entry.path, set()).discard(key)
            self._stats.evictions += 1
            self._stats.current_entries -= 1
            self._stats.current_memory_bytes -= entry.size_bytes

    def get(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> Any | None:
        if not self._config.enabled:
            return None
        key = CacheKey.make(path, columns, filters)
        path_lock = self._get_path_lock(path)
        with path_lock:
            with self._global_lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    entry = self._cache[key]
                    entry.access_count += 1
                    self._stats.hits += 1
                    return entry.df
                self._stats.misses += 1
                return None

    def put(
        self,
        path: Path,
        df: Any,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> None:
        if not self._config.enabled:
            return
        key = CacheKey.make(path, columns, filters)
        size_bytes = self._estimate_df_size(df)
        path_lock = self._get_path_lock(path)
        with path_lock:
            with self._global_lock:
                if key in self._cache:
                    old = self._cache.pop(key)
                    self._stats.current_memory_bytes -= old.size_bytes
                    self._stats.current_entries -= 1
                # Evict to make room before adding
                self._evict_if_needed()
                # Add new entry
                self._cache[key] = CacheEntry(df=df, size_bytes=size_bytes, path=path)
                self._stats.current_entries += 1
                self._stats.current_memory_bytes += size_bytes
                # Evict again if this entry pushed us over memory limit
                self._evict_if_needed()
                # Track key for path-based invalidation
                if path not in self._path_to_keys:
                    self._path_to_keys[path] = set()
                self._path_to_keys[path].add(key)

    def invalidate(self, path: Path) -> int:
        """Invalidate ALL entries for a given path. Returns count removed."""
        path_lock = self._get_path_lock(path)
        with path_lock:
            with self._global_lock:
                keys_to_remove = self._path_to_keys.pop(path, set())
                count = 0
                for key in keys_to_remove:
                    if key in self._cache:
                        entry = self._cache.pop(key)
                        self._stats.invalidations += 1
                        self._stats.current_entries -= 1
                        self._stats.current_memory_bytes -= entry.size_bytes
                        count += 1
                return count

    def clear(self) -> int:
        with self._global_lock:
            count = len(self._cache)
            self._cache.clear()
            self._path_to_keys.clear()
            self._stats.current_entries = 0
            self._stats.current_memory_bytes = 0
            return count

    def reset(self) -> None:
        """Reset cache to initial state including stats."""
        with self._global_lock:
            self._cache.clear()
            self._path_to_keys.clear()
            self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        with self._global_lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                invalidations=self._stats.invalidations,
                current_entries=self._stats.current_entries,
                current_memory_bytes=self._stats.current_memory_bytes,
            )

    def configure(self, **kwargs) -> None:
        with self._global_lock:
            for k, value in kwargs.items():
                if hasattr(self._config, k):
                    setattr(self._config, k, value)
            self._evict_if_needed()


# Module-level singleton
_memory_cache: MemoryCache | None = None
_cache_lock = threading.Lock()


def get_memory_cache() -> MemoryCache:
    global _memory_cache
    if _memory_cache is None:
        with _cache_lock:
            if _memory_cache is None:
                _memory_cache = MemoryCache()
    return _memory_cache


def configure_memory_cache(**kwargs) -> None:
    get_memory_cache().configure(**kwargs)


def clear_memory_cache() -> int:
    return get_memory_cache().clear()


def reset_memory_cache() -> None:
    """Reset the memory cache to initial state including stats."""
    get_memory_cache().reset()


def get_memory_cache_stats() -> CacheStats:
    return get_memory_cache().get_stats()
