"""Frame - Temporal dataframe wrapper with caching and concurrent execution."""

from frame.backends.pandas import PandasBackend
from frame.backends.polars import PolarsBackend
from frame.cache import CacheManager, CacheMissError, CacheMode, ChunkBy
from frame.calendar import BDateCalendar, Calendar, DateCalendar
from frame.config import (
    configure_frame,
    get_default_cache_dir,
    get_default_parent_cache_dirs,
    reset_frame_config,
)
from frame.core import Frame
from frame.executor import async_batch, batch
from frame.memory_cache import (
    CacheConfig,
    CacheStats,
    clear_memory_cache,
    configure_memory_cache,
    get_memory_cache_stats,
)
from frame.logging import configure_logging, get_logger
from frame.ops import (
    Abs,
    Add,
    Clip,
    Diff,
    Div,
    DtShift,
    Fillna,
    Filter,
    Mul,
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
from frame.ops.concat import Concat
# Import Operation for backwards compatibility but don't expose prominently
from frame.ops.base import Operation
from frame.proxy import LazyFrame

# LazyOperation is now an alias for LazyFrame (backwards compatibility)
LazyOperation = LazyFrame

__all__ = [
    # Primary API - users interact with Frame
    "Frame",
    "LazyFrame",
    # Backends
    "PandasBackend",
    "PolarsBackend",
    # Cache
    "CacheManager",
    "CacheMissError",
    "CacheMode",
    "ChunkBy",
    # Calendar
    "Calendar",
    "BDateCalendar",
    "DateCalendar",
    # Logging
    "configure_logging",
    "get_logger",
    # Memory cache
    "CacheConfig",
    "CacheStats",
    "clear_memory_cache",
    "configure_memory_cache",
    "get_memory_cache_stats",
    # Config
    "configure_frame",
    "get_default_cache_dir",
    "get_default_parent_cache_dirs",
    "reset_frame_config",
    # Batching
    "batch",
    "async_batch",
    # Operations (exported for direct use, all are Frame subclasses)
    "Abs",
    "Add",
    "Clip",
    "Concat",
    "Diff",
    "Div",
    "DtShift",
    "Fillna",
    "Filter",
    "Mul",
    "Pct",
    "Pow",
    "Rolling",
    "Select",
    "Shift",
    "Sub",
    "ToPandas",
    "ToPolars",
    "Winsorize",
    "Zscore",
    # Backwards compatibility (internal implementation detail)
    "Operation",
    "LazyOperation",
]
__version__ = "0.1.0"
