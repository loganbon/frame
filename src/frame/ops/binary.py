"""Binary operations - two-input Frame transformations."""

from typing import TYPE_CHECKING, Any, Union

from frame.ops.base import Operation
from frame.utils import _is_polars

if TYPE_CHECKING:
    from frame.core import Frame


def _apply_binary_op_polars(left: Any, right: Any, op: str) -> Any:
    """Apply binary operation to polars DataFrames on numeric columns only.

    Args:
        left: Left polars DataFrame.
        right: Right polars DataFrame (or scalar).
        op: Operation - "add", "sub", "mul", "div", "pow".

    Returns:
        Result polars DataFrame with index columns preserved.
    """
    import polars as pl

    # Index columns to preserve (not operate on)
    index_cols = {"as_of_date", "id"}

    # Get numeric columns (exclude index columns)
    numeric_cols = [
        c for c in left.columns
        if c not in index_cols and left[c].dtype in (
            pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64
        )
    ]

    if isinstance(right, (int, float)):
        # Scalar operation
        if op == "add":
            exprs = [pl.col(c) + right for c in numeric_cols]
        elif op == "sub":
            exprs = [pl.col(c) - right for c in numeric_cols]
        elif op == "mul":
            exprs = [pl.col(c) * right for c in numeric_cols]
        elif op == "div":
            exprs = [pl.col(c) / right for c in numeric_cols]
        elif op == "pow":
            exprs = [pl.col(c).pow(right) for c in numeric_cols]
        else:
            raise ValueError(f"Unknown operation: {op}")
        return left.with_columns(exprs)
    else:
        # DataFrame operation - apply to matching numeric columns
        if op == "add":
            exprs = [pl.col(c) + right[c] for c in numeric_cols if c in right.columns]
        elif op == "sub":
            exprs = [pl.col(c) - right[c] for c in numeric_cols if c in right.columns]
        elif op == "mul":
            exprs = [pl.col(c) * right[c] for c in numeric_cols if c in right.columns]
        elif op == "div":
            exprs = [pl.col(c) / right[c] for c in numeric_cols if c in right.columns]
        elif op == "pow":
            exprs = [pl.col(c).pow(right[c]) for c in numeric_cols if c in right.columns]
        else:
            raise ValueError(f"Unknown operation: {op}")
        return left.with_columns(exprs)


class Add(Operation):
    """Element-wise addition of two Frames or a Frame and a scalar."""

    def __init__(
        self,
        left: "Frame | Operation",
        right: "Union[Frame, Operation, int, float]",
    ) -> None:
        """Initialize Add operation.

        Args:
            left: Left operand Frame or Operation.
            right: Right operand Frame, Operation, or scalar.
        """
        if isinstance(right, (int, float)):
            super().__init__(left, scalar=right)
        else:
            super().__init__(left, right)

    def _apply(self, inputs: list[Any], scalar: float | None = None) -> Any:
        left = inputs[0]
        if _is_polars(left):
            right = scalar if scalar is not None else inputs[1]
            return _apply_binary_op_polars(left, right, "add")
        if scalar is not None:
            return left + scalar
        return left + inputs[1]


class Sub(Operation):
    """Element-wise subtraction of two Frames or a Frame and a scalar."""

    def __init__(
        self,
        left: "Frame | Operation",
        right: "Union[Frame, Operation, int, float]",
    ) -> None:
        """Initialize Sub operation.

        Args:
            left: Left operand Frame or Operation.
            right: Right operand Frame, Operation, or scalar.
        """
        if isinstance(right, (int, float)):
            super().__init__(left, scalar=right)
        else:
            super().__init__(left, right)

    def _apply(self, inputs: list[Any], scalar: float | None = None) -> Any:
        left = inputs[0]
        if _is_polars(left):
            right = scalar if scalar is not None else inputs[1]
            return _apply_binary_op_polars(left, right, "sub")
        if scalar is not None:
            return left - scalar
        return left - inputs[1]


class Mul(Operation):
    """Element-wise multiplication of two Frames or a Frame and a scalar."""

    def __init__(
        self,
        left: "Frame | Operation",
        right: "Union[Frame, Operation, int, float]",
    ) -> None:
        """Initialize Mul operation.

        Args:
            left: Left operand Frame or Operation.
            right: Right operand Frame, Operation, or scalar.
        """
        if isinstance(right, (int, float)):
            super().__init__(left, scalar=right)
        else:
            super().__init__(left, right)

    def _apply(self, inputs: list[Any], scalar: float | None = None) -> Any:
        left = inputs[0]
        if _is_polars(left):
            right = scalar if scalar is not None else inputs[1]
            return _apply_binary_op_polars(left, right, "mul")
        if scalar is not None:
            return left * scalar
        return left * inputs[1]


class Div(Operation):
    """Element-wise division of two Frames or a Frame and a scalar."""

    def __init__(
        self,
        left: "Frame | Operation",
        right: "Union[Frame, Operation, int, float]",
    ) -> None:
        """Initialize Div operation.

        Args:
            left: Left operand Frame or Operation (numerator).
            right: Right operand Frame, Operation, or scalar (denominator).
        """
        if isinstance(right, (int, float)):
            super().__init__(left, scalar=right)
        else:
            super().__init__(left, right)

    def _apply(self, inputs: list[Any], scalar: float | None = None) -> Any:
        left = inputs[0]
        if _is_polars(left):
            right = scalar if scalar is not None else inputs[1]
            return _apply_binary_op_polars(left, right, "div")
        if scalar is not None:
            return left / scalar
        return left / inputs[1]


class Pow(Operation):
    """Element-wise power of a Frame raised to a scalar or another Frame."""

    def __init__(
        self,
        left: "Frame | Operation",
        right: "Union[Frame, Operation, int, float]",
    ) -> None:
        """Initialize Pow operation.

        Args:
            left: Base Frame or Operation.
            right: Exponent Frame, Operation, or scalar.
        """
        if isinstance(right, (int, float)):
            super().__init__(left, scalar=right)
        else:
            super().__init__(left, right)

    def _apply(self, inputs: list[Any], scalar: float | None = None) -> Any:
        left = inputs[0]
        if _is_polars(left):
            right = scalar if scalar is not None else inputs[1]
            return _apply_binary_op_polars(left, right, "pow")
        if scalar is not None:
            return left ** scalar
        return left ** inputs[1]
