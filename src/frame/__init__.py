"""Frame - Temporal dataframe wrapper with caching and concurrent execution."""

from frame.backends.pandas import PandasBackend
from frame.backends.polars import PolarsBackend
from frame.cache import CacheManager, CacheMissError, CacheMode, ChunkGranularity
from frame.memory_cache import (
    CacheConfig,
    CacheStats,
    clear_memory_cache,
    configure_memory_cache,
    get_memory_cache_stats,
)
from frame.calendar import BDateCalendar, Calendar, DateCalendar
from frame.core import Frame
from frame.logging import configure_logging, get_logger
from frame.ops import (
    Abs,
    Add,
    Clip,
    Diff,
    Div,
    Fillna,
    Filter,
    Mul,
    Operation,
    Pct,
    Pow,
    Rolling,
    Select,
    Shift,
    Sub,
    ToPandas,
    ToPolars,
    Winsorize,
    Zscore,
)
from frame.ops import __all__ as ops_all
from frame.proxy import LazyFrame, LazyOperation

__all__ = [
    "Frame",
    "PandasBackend",
    "PolarsBackend",
    "LazyFrame",
    "LazyOperation",
    "Operation",
    "CacheManager",
    "CacheMissError",
    "CacheMode",
    "ChunkGranularity",
    "Calendar",
    "BDateCalendar",
    "DateCalendar",
    "configure_logging",
    "get_logger",
    "CacheConfig",
    "CacheStats",
    "clear_memory_cache",
    "configure_memory_cache",
    "get_memory_cache_stats",
    *ops_all,
]
__version__ = "0.1.0"
