"""Binary operations - two-input Frame transformations."""

from typing import TYPE_CHECKING, Any, Union

from frame.ops.base import Operation

if TYPE_CHECKING:
    from frame.core import Frame


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
        if scalar is not None:
            return inputs[0] + scalar
        return inputs[0] + inputs[1]


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
        if scalar is not None:
            return inputs[0] - scalar
        return inputs[0] - inputs[1]


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
        if scalar is not None:
            return inputs[0] * scalar
        return inputs[0] * inputs[1]


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
        if scalar is not None:
            return inputs[0] / scalar
        return inputs[0] / inputs[1]


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
        if scalar is not None:
            return inputs[0] ** scalar
        return inputs[0] ** inputs[1]
