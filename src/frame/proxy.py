"""LazyFrame and LazyOperation proxies for nested frame batching."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from frame.executor import get_current_batch

if TYPE_CHECKING:
    from frame.core import Frame
    from frame.ops.base import Operation


class _LazyMixins:
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying DataFrame."""
        return getattr(self._resolve(), name)

    def __getitem__(self, key: Any) -> Any:
        """Proxy item access to underlying DataFrame."""
        return self._resolve()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Proxy item setting to underlying DataFrame."""
        self._resolve()[key] = value

    def __len__(self) -> int:
        """Return length of underlying DataFrame."""
        return len(self._resolve())

    def __iter__(self):
        """Iterate over underlying DataFrame."""
        return iter(self._resolve())

    def __bool__(self) -> bool:
        """Check if DataFrame is non-empty."""
        return len(self._resolve()) > 0

    def __str__(self) -> str:
        """String representation."""
        return str(self._resolve())

    def __add__(self, other: Any) -> Any:
        """Add operation."""
        return self._resolve() + other

    def __radd__(self, other: Any) -> Any:
        """Right add operation."""
        return other + self._resolve()

    def __sub__(self, other: Any) -> Any:
        """Subtract operation."""
        return self._resolve() - other

    def __rsub__(self, other: Any) -> Any:
        """Right subtract operation."""
        return other - self._resolve()

    def __mul__(self, other: Any) -> Any:
        """Multiply operation."""
        return self._resolve() * other

    def __rmul__(self, other: Any) -> Any:
        """Right multiply operation."""
        return other * self._resolve()

    def __truediv__(self, other: Any) -> Any:
        """Divide operation."""
        return self._resolve() / other

    def __rtruediv__(self, other: Any) -> Any:
        """Right divide operation."""
        return other / self._resolve()

    def __eq__(self, other: Any) -> Any:
        """Equality comparison."""
        return self._resolve() == other

    def __ne__(self, other: Any) -> Any:
        """Inequality comparison."""
        return self._resolve() != other

    def __lt__(self, other: Any) -> Any:
        """Less than comparison."""
        return self._resolve() < other

    def __le__(self, other: Any) -> Any:
        """Less than or equal comparison."""
        return self._resolve() <= other

    def __gt__(self, other: Any) -> Any:
        """Greater than comparison."""
        return self._resolve() > other

    def __ge__(self, other: Any) -> Any:
        """Greater than or equal comparison."""
        return self._resolve() >= other

    def __neg__(self) -> Any:
        """Negation."""
        return -self._resolve()

    def __pos__(self) -> Any:
        """Positive."""
        return +self._resolve()

    def __abs__(self) -> Any:
        """Absolute value."""
        return abs(self._resolve())

    def __contains__(self, item: Any) -> bool:
        """Containment check."""
        return item in self._resolve()

    def __hash__(self) -> int:
        """Hash based on object identity for use in sets/dicts."""
        return id(self)


class LazyFrame(_LazyMixins):
    """Proxy that defers data fetch until actually needed.

    When created within a batching context, registers itself for
    concurrent resolution. Otherwise resolves immediately on access.
    """

    def __init__(
        self,
        frame: "Frame",
        start_dt: datetime,
        end_dt: datetime,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        cache_mode: str = "a",
    ):
        self._frame = frame
        self._start = start_dt
        self._end = end_dt
        self._columns = columns
        self._filters = filters
        self._cache_mode = cache_mode
        self._data: Any = None
        self._resolved = False

        batch = get_current_batch()
        if batch is not None:
            batch.append(self)

    def _resolve(self) -> Any:
        """Fetch data if not already resolved."""
        if not self._resolved:
            if self._data is None:
                self._data = self._frame._fetch_data(
                    self._start,
                    self._end,
                    columns=self._columns,
                    filters=self._filters,
                    cache_mode=self._cache_mode,
                )
            self._resolved = True
        return self._data

    def __repr__(self) -> str:
        """String representation."""
        if self._resolved:
            return repr(self._data)
        return f"LazyFrame(frame={self._frame}, start={self._start}, end={self._end}, resolved=False)"


class LazyOperation(_LazyMixins):
    """Proxy that defers operation execution until actually needed.

    When created within a batching context, registers itself for
    concurrent resolution with dependency tracking.
    """

    def __init__(
        self, operation: "Operation", start_dt: datetime, end_dt: datetime
    ) -> None:
        self._operation = operation
        self._start = start_dt
        self._end = end_dt
        self._data: Any = None
        self._resolved = False
        self._input_lazies: list[Any] = []

        batch = get_current_batch()
        if batch is not None:
            batch.append(self)
            # Collect LazyFrames/LazyOperations for inputs
            for inp in operation._inputs:
                lazy = inp.get_range(start_dt, end_dt)
                self._input_lazies.append(lazy)

    def _resolve(self) -> Any:
        """Resolve by applying operation to already-resolved inputs."""
        if not self._resolved:
            if self._data is None:
                resolved_inputs = [
                    li._resolve() if hasattr(li, "_resolve") else li
                    for li in self._input_lazies
                ]
                self._data = self._operation._apply(
                    resolved_inputs, **self._operation._params
                )
            self._resolved = True
        return self._data

    def __repr__(self) -> str:
        """String representation."""
        if self._resolved:
            return repr(self._data)
        return (
            f"LazyOperation(operation={self._operation.__class__.__name__}, "
            f"start={self._start}, end={self._end}, resolved=False)"
        )
