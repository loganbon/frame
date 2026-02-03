"""Sync threading and async execution for Frame."""

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from frame.logging import get_logger

if TYPE_CHECKING:
    from frame.proxy import LazyFrame

_log = get_logger(__name__)

# Context variable to track pending frame requests during execution
_current_batch: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "batch", default=None
)


def get_current_batch() -> list | None:
    """Get the current batch context."""
    return _current_batch.get()


def set_batch_context(batch: list | None) -> contextvars.Token:
    """Set the batch context and return token for reset."""
    return _current_batch.set(batch)


def reset_batch_context(token: contextvars.Token) -> None:
    """Reset batch context using token."""
    _current_batch.reset(token)


def _is_lazy_operation(item: Any) -> bool:
    """Check if item is a LazyFrame for an operation (transformation).

    Returns True if the item is a LazyFrame wrapping a transformation frame
    (i.e., has _input_lazies populated).
    """
    d = getattr(item, "__dict__", {})
    if "_frame" not in d:
        return False
    # Check if it's a transformation (has _input_lazies with items)
    input_lazies = d.get("_input_lazies", [])
    return len(input_lazies) > 0


def _is_lazy_frame(item: Any) -> bool:
    """Check if item is a LazyFrame for a source frame.

    Returns True if the item is a LazyFrame wrapping a source frame
    (i.e., does not have _input_lazies populated).
    """
    d = getattr(item, "__dict__", {})
    if "_frame" not in d:
        return False
    # Check if it's a source frame (no _input_lazies or empty)
    input_lazies = d.get("_input_lazies", [])
    return len(input_lazies) == 0


def _resolve_lazy_frame(item: "LazyFrame") -> Any:
    """Resolve a LazyFrame by fetching its data or applying transformation."""
    # Skip if already resolved (e.g., by _execute calling _resolve() directly)
    if getattr(item, '_resolved', False):
        return item._data

    frame = item._frame
    is_source = getattr(frame, '_is_source', True)

    if is_source:
        # Source frame: fetch data
        return frame._fetch_data(
            item._start,
            item._end,
            columns=item._columns,
            filters=item._filters,
            cache_mode=item._cache_mode,
        )
    else:
        # Transformation frame: apply to resolved inputs
        resolved_inputs = [li._data for li in item._input_lazies]
        return frame._apply(resolved_inputs, **frame._params)


def resolve_batch_sync(batch: list) -> None:
    """Resolve batch with dependency ordering using threads.

    Uses topological ordering to resolve items in the correct order:
    1. Source LazyFrames have no dependencies and are resolved first in parallel
    2. Transformation LazyFrames depend on their input_lazies and are resolved
       once all dependencies are resolved
    """
    if not batch:
        return

    _log.debug("resolving_batch", batch_size=len(batch))

    # Build dependency map: item -> set of items it depends on (within batch)
    batch_set = set(batch)
    deps: dict[Any, set] = {}
    for item in batch:
        # Get input_lazies if present (transformation frames have these)
        input_lazies = getattr(item, '_input_lazies', [])
        deps[item] = set(input_lazies) & batch_set

    resolved: set = set()
    round_num = 0

    while len(resolved) < len(batch):
        round_num += 1
        # Find items whose dependencies are all resolved
        ready = [
            item
            for item in batch
            if item not in resolved and deps[item] <= resolved
        ]

        if not ready:
            raise RuntimeError("Circular dependency detected in batch")

        _log.debug("batch_resolution_round", round=round_num, ready_count=len(ready))

        # Resolve ready items in parallel using unified resolver
        with ThreadPoolExecutor() as pool:
            futures = []
            for item in ready:
                future = pool.submit(_resolve_lazy_frame, item)
                futures.append((item, future))

            for item, future in futures:
                item._data = future.result()
                item._resolved = True

        resolved.update(ready)

    _log.debug("batch_resolution_complete", total_rounds=round_num)


async def resolve_batch_async(batch: list) -> None:
    """Resolve batch with dependency ordering using asyncio.

    Uses topological ordering to resolve items in the correct order.
    """
    if not batch:
        return

    loop = asyncio.get_event_loop()

    # Build dependency map
    batch_set = set(batch)
    deps: dict[Any, set] = {}
    for item in batch:
        # Get input_lazies if present (transformation frames have these)
        input_lazies = getattr(item, '_input_lazies', [])
        deps[item] = set(input_lazies) & batch_set

    resolved: set = set()

    async def resolve_one(item: Any) -> tuple[Any, Any]:
        # Use unified resolver for both source and transformation frames
        data = await loop.run_in_executor(None, _resolve_lazy_frame, item)
        return item, data

    while len(resolved) < len(batch):
        # Find items whose dependencies are all resolved
        ready = [
            item
            for item in batch
            if item not in resolved and deps[item] <= resolved
        ]

        if not ready:
            raise RuntimeError("Circular dependency detected in batch")

        # Resolve ready items in parallel
        results = await asyncio.gather(*[resolve_one(item) for item in ready])
        for item, data in results:
            item._data = data
            item._resolved = True

        resolved.update(ready)


def execute_with_batching(
    func, start_dt: datetime, end_dt: datetime, kwargs: dict
) -> Any:
    """Execute function with batch tracking for nested Frame calls."""
    _log.debug("batch_context_created")
    batch: list = []
    token = set_batch_context(batch)

    try:
        result = func(start_dt, end_dt, **kwargs)
        resolve_batch_sync(batch)
        return result
    finally:
        reset_batch_context(token)


async def execute_with_batching_async(
    func, start_dt: datetime, end_dt: datetime, kwargs: dict
) -> Any:
    """Execute function with async batch tracking for nested Frame calls."""
    batch: list = []
    token = set_batch_context(batch)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, func, start_dt, end_dt, **kwargs)
        await resolve_batch_async(batch)
        return result
    finally:
        reset_batch_context(token)


@contextmanager
def batch():
    """Context manager for batching lazy frame operations.

    All LazyFrame and LazyOperation objects created within this context
    will be collected and resolved in parallel when the context exits.

    Example:
        with batch():
            lazy1 = frame1.get_range(start, end)
            lazy2 = frame2.get_range(start, end)
        # Both resolved in parallel here
    """
    batch_list: list = []
    token = set_batch_context(batch_list)
    try:
        yield batch_list
    finally:
        resolve_batch_sync(batch_list)
        reset_batch_context(token)


@asynccontextmanager
async def async_batch():
    """Async context manager for batching lazy frame operations.

    All LazyFrame and LazyOperation objects created within this context
    will be collected and resolved in parallel using asyncio when the context exits.

    Example:
        async with async_batch():
            lazy1 = frame1.get_range(start, end)
            lazy2 = frame2.get_range(start, end)
        # Both resolved in parallel here
    """
    batch_list: list = []
    token = set_batch_context(batch_list)
    try:
        yield batch_list
    finally:
        await resolve_batch_async(batch_list)
        reset_batch_context(token)
