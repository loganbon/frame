"""Frame operations module.

Provides declarative transformations on Frames while preserving concurrent data fetching.
"""

from frame.ops.align import AlignTo, AlignToCalendar
from frame.ops.base import Operation
from frame.ops.binary import Add, Div, Mul, Pow, Sub
from frame.ops.concat import Concat
from frame.ops.conversion import ToPandas, ToPolars
from frame.ops.dtshift import DtShift
from frame.ops.unary import (
    Abs,
    Clip,
    Diff,
    Fillna,
    Filter,
    Pct,
    Rolling,
    Select,
    Shift,
    Winsorize,
    Zscore,
)

__all__ = [
    "Operation",
    # Unary operations
    "Rolling",
    "Shift",
    "DtShift",
    "Diff",
    "Abs",
    "Pct",
    "Select",
    "Filter",
    "Zscore",
    "Clip",
    "Winsorize",
    "Fillna",
    # Binary operations
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    # Multi-input operations
    "Concat",
    # Alignment operations
    "AlignTo",
    "AlignToCalendar",
    # Conversion operations
    "ToPandas",
    "ToPolars",
]
