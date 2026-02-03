"""Tests for LazyFrame proxy and nested frame batching."""

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from frame import Frame, LazyFrame
from frame.executor import (
    set_batch_context,
    reset_batch_context,
    resolve_batch_sync,
    _is_lazy_operation,
    _is_lazy_frame,
)


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


@pytest.fixture
def call_log():
    """Track function calls with timing."""
    return []


@pytest.fixture
def price_func(call_log):
    """Price data function."""
    def fetch_prices(start_dt: datetime, end_dt: datetime, ticker: str = "AAPL"):
        call_log.append(("prices", ticker, start_dt, end_dt))
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for dt in dates:
            records.append({
                "as_of_date": dt.to_pydatetime(),
                "id": ticker,
                "price": 100.0 + len(records),
            })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return fetch_prices


class TestLazyFrame:
    """Test LazyFrame proxy behavior."""

    def test_lazy_frame_not_resolved_until_accessed(self, price_func, cache_dir, call_log):
        """Test that LazyFrame doesn't fetch until data is accessed."""
        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)

        batch = []
        token = set_batch_context(batch)
        try:
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 5)

            lazy = frame.get_range(start, end)

            assert isinstance(lazy, LazyFrame)
            assert not lazy._resolved
            assert len(batch) == 1

            _ = len(lazy)

            assert lazy._resolved
        finally:
            reset_batch_context(token)

    def test_lazy_frame_proxy_methods(self, price_func, cache_dir):
        """Test that LazyFrame correctly proxies DataFrame methods."""
        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)

        batch = []
        token = set_batch_context(batch)
        try:
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 5)

            lazy = frame.get_range(start, end)

            assert len(lazy) == 5
            assert "price" in lazy.columns
            assert lazy["price"].sum() > 0

            head = lazy.head(2)
            assert len(head) == 2
        finally:
            reset_batch_context(token)

    def test_lazy_frame_arithmetic(self, price_func, cache_dir):
        """Test LazyFrame arithmetic operations."""
        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)

        batch = []
        token = set_batch_context(batch)
        try:
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 5)

            lazy = frame.get_range(start, end)

            result = lazy["price"] * 2
            assert result.iloc[0] == (100.0 * 2)
        finally:
            reset_batch_context(token)


class TestNestedFrames:
    """Test nested Frame calls with batching."""

    def test_nested_frame_batching(self, price_func, cache_dir, call_log):
        """Test that nested Frame calls are batched."""
        prices = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)

        def compute_returns(start_dt: datetime, end_dt: datetime, price_frame: Frame):
            p = price_frame.get_range(start_dt, end_dt)
            df = p.copy()
            df["return"] = df["price"].pct_change()
            return df

        returns = Frame(compute_returns, {"price_frame": prices}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = returns.get_range(start, end)

        assert len(result) == 5
        assert "return" in result.columns

    def test_multiple_nested_frames(self, price_func, cache_dir, call_log):
        """Test multiple nested Frame calls are batched together."""
        aapl = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        googl = Frame(price_func, {"ticker": "GOOGL"}, cache_dir=cache_dir)

        def compute_spread(start_dt: datetime, end_dt: datetime, frame1: Frame, frame2: Frame):
            p1 = frame1.get_range(start_dt, end_dt)
            p2 = frame2.get_range(start_dt, end_dt)
            df = p1.copy()
            df["spread"] = p1["price"].values - p2["price"].values
            return df

        spread = Frame(
            compute_spread,
            {"frame1": aapl, "frame2": googl},
            cache_dir=cache_dir
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = spread.get_range(start, end)

        assert len(result) == 5
        assert "spread" in result.columns

    def test_deep_nesting(self, price_func, cache_dir, call_log):
        """Test 3+ levels of nested Frame calls."""
        prices = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)

        def compute_returns(start_dt: datetime, end_dt: datetime, price_frame: Frame):
            p = price_frame.get_range(start_dt, end_dt)
            df = p.copy()
            df["return"] = df["price"].pct_change()
            return df

        returns_frame = Frame(compute_returns, {"price_frame": prices}, cache_dir=cache_dir)

        def compute_volatility(start_dt: datetime, end_dt: datetime, returns_frame: Frame):
            r = returns_frame.get_range(start_dt, end_dt)
            vol = r["return"].std()
            dates = r.index.get_level_values("as_of_date").unique()
            df = pd.DataFrame({
                "as_of_date": [dates[0]],
                "id": ["vol"],
                "volatility": [vol]
            })
            return df.set_index(["as_of_date", "id"])

        volatility_frame = Frame(
            compute_volatility,
            {"returns_frame": returns_frame},
            cache_dir=cache_dir
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = volatility_frame.get_range(start, end)

        assert "volatility" in result.columns


class TestAsyncNestedFrames:
    """Test async nested Frame operations."""

    @pytest.mark.asyncio
    async def test_async_nested_frames(self, price_func, cache_dir):
        """Test async API with nested frames."""
        import asyncio

        from frame import DateCalendar

        prices = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir, calendar=DateCalendar())

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        results = await asyncio.gather(
            prices.aget_range(start, end),
            prices.aget_range(datetime(2024, 2, 1), datetime(2024, 2, 5)),
        )

        assert len(results) == 2
        assert len(results[0]) == 5
        assert len(results[1]) == 5


class TestBatchResolutionTiming:
    """Test that batch resolution doesn't trigger premature execution."""

    def test_type_checking_does_not_resolve(self, price_func, cache_dir):
        """_is_lazy_operation and _is_lazy_frame should not trigger resolution."""
        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        batch = []
        token = set_batch_context(batch)
        try:
            lazy = frame.get_range(start, end)

            # These checks should NOT trigger resolution
            assert _is_lazy_frame(lazy)
            assert not _is_lazy_operation(lazy)

            # LazyFrame should still be unresolved
            assert not lazy._resolved
            assert lazy._data is None
        finally:
            reset_batch_context(token)

    def test_batch_items_stay_lazy_until_resolution(self, price_func, cache_dir):
        """Items in batch should not resolve until resolve_batch_sync is called."""
        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        batch = []
        token = set_batch_context(batch)
        try:
            lazy1 = frame.get_range(start, end)
            lazy2 = frame.get_range(start, end)

            # Both should be in batch and unresolved
            assert len(batch) == 2
            assert not lazy1._resolved
            assert not lazy2._resolved

            # Now resolve
            resolve_batch_sync(batch)

            # Both should now be resolved
            assert lazy1._resolved
            assert lazy2._resolved
        finally:
            reset_batch_context(token)


class TestBatchContextManager:
    """Test the batch context manager."""

    def test_batch_context_manager_resolves_on_exit(self, price_func, cache_dir):
        """Items should be resolved when exiting the context."""
        from frame import batch

        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        with batch():
            lazy1 = frame.get_range(start, end)
            lazy2 = frame.get_range(start, end)

            # Should be unresolved inside context
            assert not lazy1._resolved
            assert not lazy2._resolved

        # Should be resolved after context exits
        assert lazy1._resolved
        assert lazy2._resolved

    def test_batch_context_manager_yields_batch_list(self, price_func, cache_dir):
        """Context manager should yield the batch list."""
        from frame import batch

        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        with batch() as batch_list:
            lazy = frame.get_range(start, end)
            assert len(batch_list) == 1
            assert batch_list[0] is lazy

    def test_batch_context_manager_parallel_resolution(self, price_func, cache_dir, call_log):
        """Multiple frames should be fetched (enables parallel resolution)."""
        from frame import batch

        frame1 = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        frame2 = Frame(price_func, {"ticker": "GOOGL"}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        with batch():
            lazy1 = frame1.get_range(start, end)
            lazy2 = frame2.get_range(start, end)

        # Both should have been called
        assert len(call_log) == 2
        tickers = [c[1] for c in call_log]
        assert "AAPL" in tickers
        assert "GOOGL" in tickers


class TestAsyncBatchContextManager:
    """Test the async batch context manager."""

    @pytest.mark.asyncio
    async def test_async_batch_resolves_on_exit(self, price_func, cache_dir):
        """Items should be resolved when exiting the async context."""
        from frame import async_batch

        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        async with async_batch():
            lazy1 = frame.get_range(start, end)
            lazy2 = frame.get_range(start, end)

            # Should be unresolved inside context
            assert not lazy1._resolved
            assert not lazy2._resolved

        # Should be resolved after context exits
        assert lazy1._resolved
        assert lazy2._resolved

    @pytest.mark.asyncio
    async def test_async_batch_yields_batch_list(self, price_func, cache_dir):
        """Async context manager should yield the batch list."""
        from frame import async_batch

        frame = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        async with async_batch() as batch_list:
            lazy = frame.get_range(start, end)
            assert len(batch_list) == 1
            assert batch_list[0] is lazy
