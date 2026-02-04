"""Tests for Frame operations layer."""

import shutil
from datetime import datetime

import pandas as pd
import pytest

from frame import (
    Abs,
    Add,
    Apply,
    Clip,
    Diff,
    Div,
    Dropna,
    Fillna,
    Filter,
    Frame,
    LazyFrame,
    Mask,
    Mul,
    Operation,
    Pct,
    Pow,
    Rename,
    Rolling,
    Select,
    Shift,
    Sub,
    ToPandas,
    ToPolars,
    Where,
    Winsorize,
    Zscore,
)


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / ".frame_cache"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


@pytest.fixture
def sample_data_func():
    """Sample data function that returns DataFrame with as_of_date/id index."""
    def fetch_data(start_dt: datetime, end_dt: datetime, multiplier: int = 1):
        dates = pd.date_range(start_dt, end_dt, freq="D")
        records = []
        for dt in dates:
            for id_ in range(3):
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": id_,
                    "value": (id_ + 1) * multiplier,
                    "price": 100.0 + id_ * 10,
                })
        df = pd.DataFrame(records)
        return df.set_index(["as_of_date", "id"])
    return fetch_data


@pytest.fixture
def prices_frame(sample_data_func, cache_dir):
    """Create a prices Frame for testing."""
    return Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)


class TestOperationBase:
    """Test Operation base class."""

    def test_operation_init(self, prices_frame):
        """Test Operation initialization."""
        op = Rolling(prices_frame, window=3)
        assert op._inputs == (prices_frame,)
        assert op._params == {"window": 3, "func": "mean", "min_periods": None}
        assert op._backend is not None

    def test_operation_repr(self, prices_frame):
        """Test Operation string representation."""
        op = Rolling(prices_frame, window=3)
        repr_str = repr(op)
        assert "Rolling" in repr_str
        assert "window=3" in repr_str


class TestUnaryOperations:
    """Test unary operations (single input)."""

    def test_rolling_mean(self, prices_frame):
        """Test Rolling operation with mean."""
        rolling = Rolling(prices_frame, window=3)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = rolling.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # First two rows per group should be NaN (window=3)
        assert result["value"].isna().sum() > 0

    def test_rolling_sum(self, prices_frame):
        """Test Rolling operation with sum."""
        rolling = Rolling(prices_frame, window=2, func="sum")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = rolling.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_shift_forward(self, prices_frame):
        """Test Shift operation (lag)."""
        shift = Shift(prices_frame, periods=1)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = shift.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # First row should be NaN after shift
        assert result["value"].isna().any()

    def test_shift_backward(self, prices_frame):
        """Test Shift operation (lead)."""
        shift = Shift(prices_frame, periods=-1)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = shift.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Last row should be NaN after backward shift
        assert result["value"].isna().any()

    def test_diff(self, prices_frame):
        """Test Diff operation."""
        diff = Diff(prices_frame, periods=1)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = diff.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_abs(self, prices_frame):
        """Test Abs operation."""
        abs_op = Abs(prices_frame)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = abs_op.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert (result["value"] >= 0).all()

    def test_pct_change(self, prices_frame):
        """Test Pct operation."""
        pct = Pct(prices_frame, periods=1)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = pct.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_zscore(self, cache_dir):
        """Test Zscore operation."""
        # Use data that varies over time within each id (needed for meaningful z-score)
        def fetch_varying_data(start_dt: datetime, end_dt: datetime):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for i, dt in enumerate(dates):
                for id_ in range(3):
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": float(i + id_),  # Varies over time
                        "price": 100.0 + i * 10 + id_,
                    })
            df = pd.DataFrame(records)
            return df.set_index(["as_of_date", "id"])

        frame = Frame(fetch_varying_data, cache_dir=cache_dir)
        zscore = Zscore(frame, window=3)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = zscore.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # First two rows per group should be NaN (window=3, need at least window for std)
        assert result["value"].isna().sum() > 0
        # Z-scores should be roughly centered around 0 for normalized data
        non_nan = result["value"].dropna()
        assert len(non_nan) > 0

    def test_clip(self, prices_frame):
        """Test Clip operation."""
        clip = Clip(prices_frame, lower=1.5, upper=2.5)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = clip.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # All values should be between 1.5 and 2.5
        assert (result["value"] >= 1.5).all()
        assert (result["value"] <= 2.5).all()

    def test_clip_lower_only(self, prices_frame):
        """Test Clip operation with only lower bound."""
        clip = Clip(prices_frame, lower=2.0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = clip.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert (result["value"] >= 2.0).all()

    def test_winsorize(self, prices_frame):
        """Test Winsorize operation."""
        winsorize = Winsorize(prices_frame, lower=0.1, upper=0.9)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = winsorize.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Result should have same shape as original
        original = prices_frame.get_range(start, end)
        assert result.shape == original.shape

    def test_fillna_value(self, prices_frame):
        """Test Fillna operation with scalar value."""
        # First create some NaNs with diff
        diff = Diff(prices_frame, periods=1)
        fillna = Fillna(diff, value=0.0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = fillna.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # Should have no NaN values
        assert result["value"].isna().sum() == 0

    def test_fillna_ffill(self, prices_frame):
        """Test Fillna operation with forward fill."""
        diff = Diff(prices_frame, periods=1)
        fillna = Fillna(diff, method="ffill")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = fillna.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_fillna_bfill(self, prices_frame):
        """Test Fillna operation with backward fill."""
        diff = Diff(prices_frame, periods=1)
        fillna = Fillna(diff, method="bfill")
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = fillna.get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestBinaryOperations:
    """Test binary operations (two inputs)."""

    def test_add_frames(self, sample_data_func, cache_dir):
        """Test Add operation with two Frames."""
        frame1 = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir / "f1")
        frame2 = Frame(sample_data_func, {"multiplier": 2}, cache_dir=cache_dir / "f2")

        add = Add(frame1, frame2)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = add.get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # multiplier 1 + multiplier 2 = multiplier 3
        # For id=0: (0+1)*1 + (0+1)*2 = 1 + 2 = 3
        # For id=1: (1+1)*1 + (1+1)*2 = 2 + 4 = 6
        # For id=2: (2+1)*1 + (2+1)*2 = 3 + 6 = 9

    def test_add_scalar(self, prices_frame):
        """Test Add operation with scalar."""
        add = Add(prices_frame, 10.0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = add.get_range(start, end)

        assert (result["value"] == original["value"] + 10.0).all()

    def test_sub_frames(self, sample_data_func, cache_dir):
        """Test Sub operation with two Frames."""
        frame1 = Frame(sample_data_func, {"multiplier": 3}, cache_dir=cache_dir / "f1")
        frame2 = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir / "f2")

        sub = Sub(frame1, frame2)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = sub.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_sub_scalar(self, prices_frame):
        """Test Sub operation with scalar."""
        sub = Sub(prices_frame, 5.0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = sub.get_range(start, end)

        assert (result["value"] == original["value"] - 5.0).all()

    def test_mul_frames(self, sample_data_func, cache_dir):
        """Test Mul operation with two Frames."""
        frame1 = Frame(sample_data_func, {"multiplier": 2}, cache_dir=cache_dir / "f1")
        frame2 = Frame(sample_data_func, {"multiplier": 3}, cache_dir=cache_dir / "f2")

        mul = Mul(frame1, frame2)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = mul.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_mul_scalar(self, prices_frame):
        """Test Mul operation with scalar."""
        mul = Mul(prices_frame, 2.0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = mul.get_range(start, end)

        assert (result["value"] == original["value"] * 2.0).all()

    def test_div_frames(self, sample_data_func, cache_dir):
        """Test Div operation with two Frames."""
        frame1 = Frame(sample_data_func, {"multiplier": 6}, cache_dir=cache_dir / "f1")
        frame2 = Frame(sample_data_func, {"multiplier": 2}, cache_dir=cache_dir / "f2")

        div = Div(frame1, frame2)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = div.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_div_scalar(self, prices_frame):
        """Test Div operation with scalar."""
        div = Div(prices_frame, 2.0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = div.get_range(start, end)

        assert (result["value"] == original["value"] / 2.0).all()

    def test_pow_frames(self, sample_data_func, cache_dir):
        """Test Pow operation with two Frames."""
        frame1 = Frame(sample_data_func, {"multiplier": 2}, cache_dir=cache_dir / "f1")
        frame2 = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir / "f2")

        pow_op = Pow(frame1, frame2)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = pow_op.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_pow_scalar(self, prices_frame):
        """Test Pow operation with scalar."""
        pow_op = Pow(prices_frame, 2.0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = pow_op.get_range(start, end)

        assert (result["value"] == original["value"] ** 2.0).all()


class TestChainedOperations:
    """Test chaining multiple operations."""

    def test_operation_on_operation(self, prices_frame):
        """Test chaining operations: Rolling of Diff."""
        diff = Diff(prices_frame, periods=1)
        rolling = Rolling(diff, window=3)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = rolling.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_binary_with_unary(self, prices_frame):
        """Test: prices - Shift(prices)."""
        shifted = Shift(prices_frame, periods=1)
        sub = Sub(prices_frame, shifted)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = sub.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_deep_nesting(self, prices_frame):
        """Test deep nesting: Rolling(Rolling(prices))."""
        rolling1 = Rolling(prices_frame, window=2)
        rolling2 = Rolling(rolling1, window=2)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = rolling2.get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestConcurrentResolution:
    """Test concurrent resolution of Frames and Operations."""

    def test_frame_and_operation_concurrent(self, sample_data_func, cache_dir):
        """Test that Frame and Operation on same Frame resolve concurrently."""
        from frame.executor import set_batch_context, reset_batch_context, resolve_batch_sync

        prices = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        rolling = Rolling(prices, window=3)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # Create a batch context and collect lazy items
        batch = []
        token = set_batch_context(batch)

        try:
            # This should create LazyFrame for prices
            lazy_prices = prices.get_range(start, end)
            # This should create LazyFrame for rolling, which adds LazyFrame for prices
            lazy_rolling = rolling.get_range(start, end)

            # Verify both are in batch
            assert len(batch) >= 2  # At least prices and rolling

            # Resolve the batch
            resolve_batch_sync(batch)

            # Both should now be resolved
            assert lazy_prices._resolved
            assert lazy_rolling._resolved

            # Verify we can access the data
            assert len(lazy_prices) > 0
            assert len(lazy_rolling) > 0

        finally:
            reset_batch_context(token)

    def test_shared_dependency(self, sample_data_func, cache_dir):
        """Test operations sharing the same input Frame."""
        from frame.executor import set_batch_context, reset_batch_context, resolve_batch_sync

        prices = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)
        rolling = Rolling(prices, window=3)
        shifted = Shift(prices, periods=1)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # Create a batch context
        batch = []
        token = set_batch_context(batch)

        try:
            lazy_rolling = rolling.get_range(start, end)
            lazy_shifted = shifted.get_range(start, end)

            # Both operations should be in batch, plus their shared dependency
            # rolling creates: LazyFrame(rolling), LazyFrame(prices)
            # shifted creates: LazyFrame(shifted), LazyFrame(prices)
            assert len(batch) >= 2  # At least rolling and shifted operations

            # Resolve the batch
            resolve_batch_sync(batch)

            # Both should be resolved
            assert lazy_rolling._resolved
            assert lazy_shifted._resolved

            # Verify we can access the data
            assert len(lazy_rolling) > 0
            assert len(lazy_shifted) > 0

        finally:
            reset_batch_context(token)


class TestLazyOperation:
    """Test LazyFrame proxy behavior for operations."""

    def test_lazy_operation_creation(self, prices_frame):
        """Test LazyFrame is created in batch context for operations."""
        from frame.executor import set_batch_context, reset_batch_context

        batch = []
        token = set_batch_context(batch)

        try:
            rolling = Rolling(prices_frame, window=3)
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 5)

            result = rolling.get_range(start, end)

            assert isinstance(result, LazyFrame)
            assert result in batch
        finally:
            reset_batch_context(token)

    def test_lazy_operation_proxy_methods(self, prices_frame):
        """Test LazyFrame proxy methods work correctly for operations."""
        rolling = Rolling(prices_frame, window=3)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = rolling.get_range(start, end)

        # Test various proxy operations
        assert len(result) > 0
        assert "value" in result.columns
        assert result["value"].isna().any()

    def test_lazy_operation_repr_unresolved(self, prices_frame):
        """Test LazyFrame repr when unresolved for operations."""
        from frame.executor import set_batch_context, reset_batch_context

        batch = []
        token = set_batch_context(batch)

        try:
            rolling = Rolling(prices_frame, window=3)
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 5)

            result = rolling.get_range(start, end)
            repr_str = repr(result)

            assert "LazyFrame" in repr_str
            assert "Rolling" in repr_str
            assert "resolved=False" in repr_str
        finally:
            reset_batch_context(token)


class TestOperationGet:
    """Test Operation.get() for single date access."""

    def test_operation_get_single_date(self, prices_frame):
        """Test get() returns data for single date."""
        rolling = Rolling(prices_frame, window=3)
        dt = datetime(2024, 1, 15)

        result = rolling.get(dt)

        assert isinstance(result, pd.DataFrame)
        assert "as_of_date" not in result.index.names


class TestAsyncOperations:
    """Test async operation methods."""

    @pytest.mark.asyncio
    async def test_aget_range(self, prices_frame):
        """Test async get_range."""
        rolling = Rolling(prices_frame, window=3)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = await rolling.aget_range(start, end)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_aget_single_date(self, prices_frame):
        """Test async get for single date."""
        rolling = Rolling(prices_frame, window=3)
        dt = datetime(2024, 1, 15)

        result = await rolling.aget(dt)

        assert isinstance(result, pd.DataFrame)
        assert "as_of_date" not in result.index.names


class TestDependencyResolution:
    """Test dependency-aware batch resolution."""

    def test_topological_ordering(self, prices_frame):
        """Test that dependencies are resolved in correct order."""
        # Create a chain: prices -> rolling -> rolling2
        rolling1 = Rolling(prices_frame, window=2)
        rolling2 = Rolling(rolling1, window=2)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # This should work - dependencies resolved in order
        result = rolling2.get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_diamond_dependency(self, prices_frame):
        """Test diamond-shaped dependency graph.

        prices
          /  \\
      rolling shifted
          \\  /
           sub
        """
        rolling = Rolling(prices_frame, window=2)
        shifted = Shift(prices_frame, periods=1)
        sub = Sub(rolling, shifted)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = sub.get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestFluentAPI:
    """Test fluent/method-style API for operations."""

    def test_frame_rolling_method(self, prices_frame):
        """Test Frame.rolling() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.rolling(window=3).get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # First two rows should be NaN for window=3
        assert result["value"].isna().sum() > 0

    def test_frame_shift_method(self, prices_frame):
        """Test Frame.shift() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.shift(periods=1).get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_frame_diff_method(self, prices_frame):
        """Test Frame.diff() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.diff().get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_frame_abs_method(self, prices_frame):
        """Test Frame.abs() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.abs().get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert (result["value"] >= 0).all()

    def test_frame_pct_change_method(self, prices_frame):
        """Test Frame.pct_change() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.pct_change().get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_frame_add_method(self, prices_frame):
        """Test Frame.add() method with scalar."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = prices_frame.add(10).get_range(start, end)

        assert (result["value"] == original["value"] + 10).all()

    def test_frame_sub_method(self, prices_frame):
        """Test Frame.sub() method with scalar."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = prices_frame.sub(5).get_range(start, end)

        assert (result["value"] == original["value"] - 5).all()

    def test_frame_mul_method(self, prices_frame):
        """Test Frame.mul() method with scalar."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = prices_frame.mul(2).get_range(start, end)

        assert (result["value"] == original["value"] * 2).all()

    def test_frame_div_method(self, prices_frame):
        """Test Frame.div() method with scalar."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = prices_frame.div(2).get_range(start, end)

        assert (result["value"] == original["value"] / 2).all()

    def test_frame_pow_method(self, prices_frame):
        """Test Frame.pow() method with scalar."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = prices_frame.pow(2).get_range(start, end)

        assert (result["value"] == original["value"] ** 2).all()

    def test_frame_zscore_method(self, prices_frame):
        """Test Frame.zscore() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.zscore(window=3).get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        # First two rows per group should be NaN (window=3)
        assert result["value"].isna().sum() > 0

    def test_frame_clip_method(self, prices_frame):
        """Test Frame.clip() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.clip(lower=1.5, upper=2.5).get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert (result["value"] >= 1.5).all()
        assert (result["value"] <= 2.5).all()

    def test_frame_winsorize_method(self, prices_frame):
        """Test Frame.winsorize() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = prices_frame.winsorize(lower=0.1, upper=0.9).get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_frame_fillna_method(self, prices_frame):
        """Test Frame.fillna() method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # Create NaNs with diff, then fill with 0
        result = prices_frame.diff().fillna(value=0.0).get_range(start, end)

        assert isinstance(result, pd.DataFrame)
        assert result["value"].isna().sum() == 0

    def test_chaining_methods(self, prices_frame):
        """Test chaining multiple methods."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        # Chain: pct_change -> rolling -> abs
        result = prices_frame.pct_change().rolling(3).abs().get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_operator_add(self, prices_frame):
        """Test + operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = (prices_frame + 10).get_range(start, end)

        assert (result["value"] == original["value"] + 10).all()

    def test_operator_sub(self, prices_frame):
        """Test - operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = (prices_frame - 5).get_range(start, end)

        assert (result["value"] == original["value"] - 5).all()

    def test_operator_mul(self, prices_frame):
        """Test * operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = (prices_frame * 2).get_range(start, end)

        assert (result["value"] == original["value"] * 2).all()

    def test_operator_div(self, prices_frame):
        """Test / operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = (prices_frame / 2).get_range(start, end)

        assert (result["value"] == original["value"] / 2).all()

    def test_operator_neg(self, prices_frame):
        """Test negation operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = (-prices_frame).get_range(start, end)

        assert (result["value"] == -original["value"]).all()

    def test_operator_rmul(self, prices_frame):
        """Test right multiply: 2 * frame."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = (2 * prices_frame).get_range(start, end)

        assert (result["value"] == original["value"] * 2).all()

    def test_operator_pow(self, prices_frame):
        """Test ** operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = (prices_frame ** 2).get_range(start, end)

        assert (result["value"] == original["value"] ** 2).all()

    def test_operation_chaining_methods(self, prices_frame):
        """Test that operations can also use method chaining."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        rolling = Rolling(prices_frame, window=3)
        # Chain method on Operation
        result = rolling.shift(1).abs().get_range(start, end)

        assert isinstance(result, pd.DataFrame)

    def test_complex_expression(self, sample_data_func, cache_dir):
        """Test complex expression: (prices - prices.shift(1)).abs()."""
        prices = Frame(sample_data_func, {"multiplier": 1}, cache_dir=cache_dir)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # Complex expression using operators and methods
        result = (prices - prices.shift(1)).abs().get_range(start, end)

        assert isinstance(result, pd.DataFrame)


class TestSelectOperation:
    """Test Select operation for column selection."""

    def test_select_single_column(self, prices_frame):
        """Test Select operation with single column."""
        selected = Select(prices_frame, columns=["value"])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = selected.get_range(start, end)

        assert list(result.columns) == ["value"]

    def test_select_multiple_columns(self, prices_frame):
        """Test Select operation with multiple columns."""
        selected = Select(prices_frame, columns=["value", "price"])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = selected.get_range(start, end)

        assert list(result.columns) == ["value", "price"]

    def test_select_fluent_api(self, prices_frame):
        """Test Frame.select() fluent API."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.select(["value"]).get_range(start, end)

        assert list(result.columns) == ["value"]

    def test_select_chained_with_operations(self, prices_frame):
        """Test Select chained with other operations."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # Select columns first, then apply rolling
        result = prices_frame.select(["value"]).rolling(window=3).get_range(start, end)

        assert list(result.columns) == ["value"]
        assert result["value"].isna().any()  # Rolling produces NaN


class TestFilterOperation:
    """Test Filter operation for row filtering."""

    def test_filter_greater_than(self, prices_frame):
        """Test Filter with > operator."""
        filtered = Filter(prices_frame, filters=[("value", ">", 1)])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = filtered.get_range(start, end)

        assert (result["value"] > 1).all()

    def test_filter_equal(self, prices_frame):
        """Test Filter with = operator."""
        filtered = Filter(prices_frame, filters=[("value", "=", 2)])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = filtered.get_range(start, end)

        assert (result["value"] == 2).all()

    def test_filter_less_than(self, prices_frame):
        """Test Filter with < operator."""
        filtered = Filter(prices_frame, filters=[("value", "<", 3)])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = filtered.get_range(start, end)

        assert (result["value"] < 3).all()

    def test_filter_not_equal(self, prices_frame):
        """Test Filter with != operator."""
        filtered = Filter(prices_frame, filters=[("value", "!=", 2)])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = filtered.get_range(start, end)

        assert (result["value"] != 2).all()

    def test_filter_in(self, prices_frame):
        """Test Filter with 'in' operator."""
        filtered = Filter(prices_frame, filters=[("value", "in", [1, 3])])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = filtered.get_range(start, end)

        assert result["value"].isin([1, 3]).all()

    def test_filter_not_in(self, prices_frame):
        """Test Filter with 'not in' operator."""
        filtered = Filter(prices_frame, filters=[("value", "not in", [2])])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = filtered.get_range(start, end)

        assert (~result["value"].isin([2])).all()

    def test_filter_multiple_conditions(self, prices_frame):
        """Test Filter with multiple conditions (AND)."""
        filtered = Filter(prices_frame, filters=[("value", ">", 1), ("value", "<", 3)])
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = filtered.get_range(start, end)

        assert (result["value"] == 2).all()

    def test_filter_fluent_api(self, prices_frame):
        """Test Frame.filter() fluent API."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.filter([("value", ">", 1)]).get_range(start, end)

        assert (result["value"] > 1).all()


class TestSelectFilterChaining:
    """Test chaining Select and Filter operations."""

    def test_filter_then_select(self, prices_frame):
        """Test filtering then selecting columns."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = (
            prices_frame
            .filter([("value", ">", 1)])
            .select(["value"])
            .get_range(start, end)
        )

        assert list(result.columns) == ["value"]
        assert (result["value"] > 1).all()

    def test_select_then_filter(self, prices_frame):
        """Test selecting columns then filtering."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = (
            prices_frame
            .select(["value"])
            .filter([("value", ">", 1)])
            .get_range(start, end)
        )

        assert list(result.columns) == ["value"]
        assert (result["value"] > 1).all()

    def test_filter_select_rolling(self, prices_frame):
        """Test chaining filter, select, and rolling."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        result = (
            prices_frame
            .filter([("value", ">=", 2)])
            .select(["value"])
            .rolling(window=3)
            .get_range(start, end)
        )

        assert list(result.columns) == ["value"]


class TestConversionOperations:
    """Test ToPandas and ToPolars conversion operations."""

    def test_to_pandas_from_polars(self, tmp_path):
        """ToPandas converts polars DataFrame to pandas."""
        import polars as pl

        def fetch_polars(start, end):
            return pl.DataFrame({
                "as_of_date": [start, end],
                "id": ["A", "A"],
                "value": [1.0, 2.0],
            })

        frame = Frame(fetch_polars, backend="polars", cache_dir=tmp_path)
        converted = ToPandas(frame)

        result = converted.get_range(datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert isinstance(result, pd.DataFrame)

    def test_to_polars_from_pandas(self, tmp_path):
        """ToPolars converts pandas DataFrame to polars."""
        import polars as pl

        def fetch_pandas(start, end):
            return pd.DataFrame({
                "as_of_date": [start, end],
                "id": ["A", "A"],
                "value": [1.0, 2.0],
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_pandas, backend="pandas", cache_dir=tmp_path)
        converted = ToPolars(frame)

        result = converted.get_range(datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert isinstance(result, pl.DataFrame)

    def test_to_pandas_passthrough(self, tmp_path):
        """ToPandas returns pandas DataFrame unchanged."""
        def fetch_pandas(start, end):
            return pd.DataFrame({
                "as_of_date": [start],
                "id": ["A"],
                "value": [1.0],
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_pandas, backend="pandas", cache_dir=tmp_path)
        converted = ToPandas(frame)

        result = converted.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))
        assert isinstance(result, pd.DataFrame)

    def test_to_polars_passthrough(self, tmp_path):
        """ToPolars returns polars DataFrame unchanged."""
        import polars as pl

        def fetch_polars(start, end):
            return pl.DataFrame({
                "as_of_date": [start],
                "id": ["A"],
                "value": [1.0],
            })

        frame = Frame(fetch_polars, backend="polars", cache_dir=tmp_path)
        converted = ToPolars(frame)

        result = converted.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))
        assert isinstance(result, pl.DataFrame)

    def test_conversion_preserves_batching(self, tmp_path):
        """Conversion operations work with lazy batching."""
        import polars as pl

        call_count = {"n": 0}

        def fetch_polars(start, end):
            call_count["n"] += 1
            return pl.DataFrame({
                "as_of_date": [start],
                "id": ["A"],
                "value": [1.0],
            })

        frame = Frame(fetch_polars, backend="polars", cache_dir=tmp_path)
        converted = ToPandas(frame)

        # Just get the converted result directly to test batching works
        result = converted.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        assert call_count["n"] == 1
        assert isinstance(result, pd.DataFrame)

    def test_frame_to_backend_pandas(self, tmp_path):
        """Frame.to_backend('pandas') returns ToPandas operation."""
        import polars as pl

        def fetch_polars(start, end):
            return pl.DataFrame({
                "as_of_date": [start],
                "id": ["A"],
                "value": [1.0],
            })

        frame = Frame(fetch_polars, backend="polars", cache_dir=tmp_path)
        pandas_view = frame.to_backend("pandas")

        assert isinstance(pandas_view, ToPandas)
        result = pandas_view.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))
        assert isinstance(result, pd.DataFrame)

    def test_frame_to_backend_polars(self, tmp_path):
        """Frame.to_backend('polars') returns ToPolars operation."""
        import polars as pl

        def fetch_pandas(start, end):
            return pd.DataFrame({
                "as_of_date": [start],
                "id": ["A"],
                "value": [1.0],
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_pandas, backend="pandas", cache_dir=tmp_path)
        polars_view = frame.to_backend("polars")

        assert isinstance(polars_view, ToPolars)
        result = polars_view.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))
        assert isinstance(result, pl.DataFrame)

    def test_frame_to_backend_invalid(self, tmp_path):
        """Frame.to_backend() raises error for invalid backend."""
        def fetch_pandas(start, end):
            return pd.DataFrame({
                "as_of_date": [start],
                "id": ["A"],
                "value": [1.0],
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_pandas, backend="pandas", cache_dir=tmp_path)

        with pytest.raises(ValueError, match="Unknown backend"):
            frame.to_backend("invalid")

    def test_operation_to_backend(self, tmp_path):
        """Operation.to_backend() works for chaining."""
        import polars as pl

        def fetch_polars(start, end):
            return pl.DataFrame({
                "as_of_date": [start],
                "id": ["A"],
                "value": [1.0],
            })

        frame = Frame(fetch_polars, backend="polars", cache_dir=tmp_path)
        # Use Select operation instead of Rolling to avoid numeric type issues
        selected = Select(frame, columns=["value"])
        pandas_view = selected.to_backend("pandas")

        assert isinstance(pandas_view, ToPandas)
        result = pandas_view.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))
        assert isinstance(result, pd.DataFrame)

    def test_chaining_with_conversion(self, tmp_path):
        """Test chaining other operations after conversion."""
        import polars as pl

        def fetch_polars(start, end):
            dates = pd.date_range(start, end, freq="D")
            return pl.DataFrame({
                "as_of_date": dates.to_list(),
                "id": ["A"] * len(dates),
                "value": [float(i) for i in range(len(dates))],
            })

        frame = Frame(fetch_polars, backend="polars", cache_dir=tmp_path)
        # polars -> convert to pandas -> select (to only keep numeric columns) -> rolling
        result = ToPandas(frame).select(["value"]).rolling(window=2).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        assert isinstance(result, pd.DataFrame)
        # First row should be NaN from rolling
        assert result["value"].isna().any()

    def test_to_pandas_type_error(self, prices_frame):
        """ToPandas raises TypeError for unconvertible input."""
        class MockOperation(Operation):
            def _apply(self, inputs, **params):
                return "not a dataframe"

        mock_op = MockOperation(prices_frame)
        converted = ToPandas(mock_op)

        with pytest.raises(TypeError, match="Cannot convert"):
            converted.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

    def test_to_polars_type_error(self, prices_frame):
        """ToPolars raises TypeError for unconvertible input."""
        class MockOperation(Operation):
            def _apply(self, inputs, **params):
                return "not a dataframe"

        mock_op = MockOperation(prices_frame)
        converted = ToPolars(mock_op)

        with pytest.raises(TypeError, match="Cannot convert"):
            converted.get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))


class TestOperationCacheMode:
    """Test cache_mode propagation through Operations."""

    def test_cache_mode_propagates_to_source_frame(self, tmp_path):
        """cache_mode passed to operation.get_range() reaches source frame."""
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 1.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # First call populates cache
        op.get_range(start, end)
        assert call_counter["count"] == 1

        # Default cache_mode="a" uses cache, no new fetch
        op.get_range(start, end)
        assert call_counter["count"] == 1

        # cache_mode="w" forces refresh
        op.get_range(start, end, cache_mode="w")
        assert call_counter["count"] == 2

    def test_cache_mode_propagates_through_chained_operations(self, tmp_path):
        """cache_mode propagates through multiple chained operations."""
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 1.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        # Chain multiple operations
        op = frame.rolling(2).shift(1).abs()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # First call populates cache
        op.get_range(start, end)
        assert call_counter["count"] == 1

        # cache_mode="w" propagates through the chain
        op.get_range(start, end, cache_mode="w")
        assert call_counter["count"] == 2

    def test_cache_mode_live_skips_cache(self, tmp_path):
        """cache_mode='l' (live) always fetches fresh data."""
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            return pd.DataFrame({
                "as_of_date": [start_dt],
                "id": ["A"],
                "value": [1.0],
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # Live mode always fetches
        op.get_range(start, end, cache_mode="l")
        assert call_counter["count"] == 1

        op.get_range(start, end, cache_mode="l")
        assert call_counter["count"] == 2

        op.get_range(start, end, cache_mode="l")
        assert call_counter["count"] == 3

    def test_operation_get_with_cache_mode(self, tmp_path):
        """Operation.get() accepts cache_mode parameter."""
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 1.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        dt = datetime(2024, 1, 15)

        # First call populates cache
        op.get(dt)
        assert call_counter["count"] == 1

        # cache_mode="w" forces refresh
        op.get(dt, cache_mode="w")
        assert call_counter["count"] == 2

    @pytest.mark.asyncio
    async def test_async_operation_cache_mode(self, tmp_path):
        """Async operation methods accept cache_mode parameter."""
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 1.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # First call populates cache
        await op.aget_range(start, end)
        assert call_counter["count"] == 1

        # cache_mode="w" forces refresh
        await op.aget_range(start, end, cache_mode="w")
        assert call_counter["count"] == 2

    @pytest.mark.asyncio
    async def test_async_operation_get_cache_mode(self, tmp_path):
        """Async operation.aget() accepts cache_mode parameter."""
        call_counter = {"count": 0}

        def fetch_data(start_dt, end_dt):
            call_counter["count"] += 1
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 1.0,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        dt = datetime(2024, 1, 15)

        # First call populates cache
        await op.aget(dt)
        assert call_counter["count"] == 1

        # cache_mode="w" forces refresh
        await op.aget(dt, cache_mode="w")
        assert call_counter["count"] == 2

    def test_operation_columns_parameter(self, tmp_path):
        """Operation.get_range() accepts columns parameter."""
        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                records.append({
                    "as_of_date": dt.to_pydatetime(),
                    "id": "A",
                    "value": 1.0,
                    "price": 100.0,
                    "volume": 1000,
                })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # Request only specific columns
        result = op.get_range(start, end, columns=["value"])

        # Rolling operation should only have the requested column
        assert "value" in result.columns
        assert "price" not in result.columns
        assert "volume" not in result.columns

    def test_operation_filters_parameter(self, tmp_path):
        """Operation.get_range() accepts filters parameter."""
        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in ["A", "B", "C"]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": 1.0 if id_ == "A" else 2.0,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # Filter at the source level
        result = op.get_range(start, end, filters=[("id", "=", "A")])

        # Result should only contain id="A"
        ids = result.index.get_level_values("id").unique().tolist()
        assert ids == ["A"]

    def test_operation_get_with_columns_and_filters(self, tmp_path):
        """Operation.get() accepts columns and filters parameters."""
        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in ["A", "B"]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": 1.0,
                        "price": 100.0,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        op = Rolling(frame, window=2)

        dt = datetime(2024, 1, 15)

        result = op.get(dt, columns=["value"], filters=[("id", "=", "A")])

        assert "value" in result.columns
        assert "price" not in result.columns
        ids = result.index.get_level_values("id").unique().tolist()
        assert ids == ["A"]

    def test_operation_columns_filters_propagate_through_chain(self, tmp_path):
        """columns and filters propagate through chained operations."""
        def fetch_data(start_dt, end_dt):
            dates = pd.date_range(start_dt, end_dt, freq="D")
            records = []
            for dt in dates:
                for id_ in ["A", "B", "C"]:
                    records.append({
                        "as_of_date": dt.to_pydatetime(),
                        "id": id_,
                        "value": 1.0,
                        "price": 100.0,
                        "volume": 1000,
                    })
            return pd.DataFrame(records).set_index(["as_of_date", "id"])

        frame = Frame(fetch_data, {}, cache_dir=tmp_path)
        # Chain multiple operations
        op = frame.rolling(2).shift(1).abs()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = op.get_range(
            start, end,
            columns=["value"],
            filters=[("id", "in", ["A", "B"])],
        )

        # Check columns
        assert "value" in result.columns
        assert "price" not in result.columns

        # Check filters
        ids = result.index.get_level_values("id").unique().tolist()
        assert set(ids) == {"A", "B"}


class TestWhereOperation:
    """Test Where operation."""

    def test_where_basic(self, cache_dir):
        """Test Where replaces values where condition is False."""
        def fetch_values(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [10.0, 20.0, 30.0],
            }).set_index(["as_of_date", "id"])

        def fetch_cond(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [True, False, True],
            }).set_index(["as_of_date", "id"])

        values_frame = Frame(fetch_values, cache_dir=cache_dir / "values")
        cond_frame = Frame(fetch_cond, cache_dir=cache_dir / "cond")

        result = Where(values_frame, cond_frame, other=0.0).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )

        # id=0: True -> 10, id=1: False -> 0, id=2: True -> 30
        assert result.loc[(slice(None), 0), "value"].iloc[0] == 10.0
        assert result.loc[(slice(None), 1), "value"].iloc[0] == 0.0
        assert result.loc[(slice(None), 2), "value"].iloc[0] == 30.0

    def test_where_fluent_api(self, cache_dir):
        """Test Frame.where() fluent method."""
        def fetch_values(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [10.0, 20.0, 30.0],
            }).set_index(["as_of_date", "id"])

        def fetch_cond(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [True, True, False],
            }).set_index(["as_of_date", "id"])

        values_frame = Frame(fetch_values, cache_dir=cache_dir / "values")
        cond_frame = Frame(fetch_cond, cache_dir=cache_dir / "cond")

        result = values_frame.where(cond_frame, other=-1.0).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )

        assert result.loc[(slice(None), 2), "value"].iloc[0] == -1.0


class TestMaskOperation:
    """Test Mask operation."""

    def test_mask_basic(self, cache_dir):
        """Test Mask replaces values where condition is True."""
        def fetch_values(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [10.0, 20.0, 30.0],
            }).set_index(["as_of_date", "id"])

        def fetch_cond(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [True, False, True],
            }).set_index(["as_of_date", "id"])

        values_frame = Frame(fetch_values, cache_dir=cache_dir / "values")
        cond_frame = Frame(fetch_cond, cache_dir=cache_dir / "cond")

        result = Mask(values_frame, cond_frame, other=0.0).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )

        # id=0: True -> 0 (masked), id=1: False -> 20 (kept), id=2: True -> 0 (masked)
        assert result.loc[(slice(None), 0), "value"].iloc[0] == 0.0
        assert result.loc[(slice(None), 1), "value"].iloc[0] == 20.0
        assert result.loc[(slice(None), 2), "value"].iloc[0] == 0.0

    def test_mask_fluent_api(self, cache_dir):
        """Test Frame.mask() fluent method."""
        def fetch_values(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [10.0, 20.0, 30.0],
            }).set_index(["as_of_date", "id"])

        def fetch_cond(start_dt, end_dt):
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [False, True, False],
            }).set_index(["as_of_date", "id"])

        values_frame = Frame(fetch_values, cache_dir=cache_dir / "values")
        cond_frame = Frame(fetch_cond, cache_dir=cache_dir / "cond")

        result = values_frame.mask(cond_frame, other=-1.0).get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )

        assert result.loc[(slice(None), 1), "value"].iloc[0] == -1.0


class TestDropnaOperation:
    """Test Dropna operation."""

    def test_dropna_any(self, cache_dir):
        """Test Dropna with how='any'."""
        def fetch_with_na(start_dt, end_dt):
            import numpy as np
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [10.0, np.nan, 30.0],
                "other": [1.0, 2.0, 3.0],
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_with_na, cache_dir=cache_dir)
        result = Dropna(frame, how="any").get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )

        # Row with id=1 should be dropped (has NaN in value)
        ids = result.index.get_level_values("id").tolist()
        assert 1 not in ids
        assert len(ids) == 2

    def test_dropna_all(self, cache_dir):
        """Test Dropna with how='all'."""
        def fetch_with_na(start_dt, end_dt):
            import numpy as np
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [10.0, np.nan, np.nan],
                "other": [1.0, np.nan, 3.0],  # id=1 has all NaN
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_with_na, cache_dir=cache_dir)
        result = Dropna(frame, how="all").get_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )

        # Only row with id=1 should be dropped (all values are NaN)
        ids = result.index.get_level_values("id").tolist()
        assert 1 not in ids
        assert 0 in ids
        assert 2 in ids

    def test_dropna_invalid_how(self, prices_frame):
        """Test Dropna with invalid how raises error."""
        with pytest.raises(ValueError, match="how must be"):
            Dropna(prices_frame, how="invalid")

    def test_dropna_fluent_api(self, cache_dir):
        """Test Frame.dropna() fluent method."""
        def fetch_with_na(start_dt, end_dt):
            import numpy as np
            return pd.DataFrame({
                "as_of_date": [start_dt] * 3,
                "id": [0, 1, 2],
                "value": [10.0, np.nan, 30.0],
            }).set_index(["as_of_date", "id"])

        frame = Frame(fetch_with_na, cache_dir=cache_dir)
        result = frame.dropna().get_range(datetime(2024, 1, 1), datetime(2024, 1, 1))

        ids = result.index.get_level_values("id").tolist()
        assert 1 not in ids


class TestRenameOperation:
    """Test Rename operation."""

    def test_rename_basic(self, prices_frame):
        """Test basic column rename."""
        renamed = Rename(prices_frame, mapping={"value": "new_value"})
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = renamed.get_range(start, end)

        assert "new_value" in result.columns
        assert "value" not in result.columns
        assert "price" in result.columns  # Unchanged

    def test_rename_multiple_columns(self, prices_frame):
        """Test renaming multiple columns."""
        renamed = Rename(prices_frame, mapping={"value": "val", "price": "prc"})
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = renamed.get_range(start, end)

        assert "val" in result.columns
        assert "prc" in result.columns
        assert "value" not in result.columns
        assert "price" not in result.columns

    def test_rename_fluent_api(self, prices_frame):
        """Test Frame.rename() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.rename({"value": "renamed_value"}).get_range(start, end)

        assert "renamed_value" in result.columns
        assert "value" not in result.columns


class TestApplyOperation:
    """Test Apply operation."""

    def test_apply_basic(self, prices_frame):
        """Test basic apply with custom function."""
        def double_values(df):
            result = df.copy()
            result["value"] = df["value"] * 2
            return result

        applied = Apply(prices_frame, func=double_values)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        original = prices_frame.get_range(start, end)
        result = applied.get_range(start, end)

        assert (result["value"] == original["value"] * 2).all()

    def test_apply_add_column(self, prices_frame):
        """Test apply that adds a new column."""
        def add_ratio(df):
            result = df.copy()
            result["ratio"] = df["value"] / df["price"]
            return result

        applied = Apply(prices_frame, func=add_ratio)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = applied.get_range(start, end)

        assert "ratio" in result.columns
        assert "value" in result.columns
        assert "price" in result.columns

    def test_apply_fluent_api(self, prices_frame):
        """Test Frame.apply() fluent method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        result = prices_frame.apply(lambda df: df * 2).get_range(start, end)

        original = prices_frame.get_range(start, end)
        assert (result["value"] == original["value"] * 2).all()

    def test_apply_chained(self, prices_frame):
        """Test chaining apply with other operations."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = (
            prices_frame
            .rolling(window=3)
            .apply(lambda df: df.fillna(0))
            .get_range(start, end)
        )

        assert isinstance(result, pd.DataFrame)
        assert result["value"].isna().sum() == 0
