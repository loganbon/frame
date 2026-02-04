"""Frame operations module.

Provides declarative transformations on Frames while preserving concurrent data fetching.
"""

from frame.ops.align import AlignTo, AlignToCalendar
from frame.ops.base import Operation
from frame.ops.binary import Add, Div, Mul, Pow, Sub
from frame.ops.concat import Concat
from frame.ops.conversion import ToPandas, ToPolars
from frame.ops.cross_sectional import CsDemean, CsRank, CsWinsorize, CsZscore
from frame.ops.dtshift import DtShift
from frame.ops.join import AsofJoin, Join
from frame.ops.timeseries import (
    Cummax,
    Cummin,
    Cumprod,
    Cumsum,
    Ewm,
    Expanding,
    Resample,
)
from frame.ops.unary import (
    Abs,
    Apply,
    Clip,
    Diff,
    Dropna,
    Fillna,
    Filter,
    Mask,
    Pct,
    Rename,
    Rolling,
    Select,
    Shift,
    Where,
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
    "Where",
    "Mask",
    "Dropna",
    "Rename",
    "Apply",
    # Binary operations
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    # Cross-sectional operations
    "CsRank",
    "CsZscore",
    "CsDemean",
    "CsWinsorize",
    # Time series operations
    "Ewm",
    "Expanding",
    "Cumsum",
    "Cumprod",
    "Cummax",
    "Cummin",
    "Resample",
    # Multi-input operations
    "Concat",
    "Join",
    "AsofJoin",
    # Alignment operations
    "AlignTo",
    "AlignToCalendar",
    # Conversion operations
    "ToPandas",
    "ToPolars",
]
