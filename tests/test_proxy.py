"""Tests for LazyFrame proxy and nested frame batching."""

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from frame import Frame, LazyFrame
from frame.executor import set_batch_context, reset_batch_context


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

        prices = Frame(price_func, {"ticker": "AAPL"}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        results = await asyncio.gather(
            prices.aget_range(start, end),
            prices.aget_range(datetime(2024, 2, 1), datetime(2024, 2, 5)),
        )

        assert len(results) == 2
        assert len(results[0]) == 5
        assert len(results[1]) == 5
