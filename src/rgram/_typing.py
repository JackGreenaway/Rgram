"""Shared input, callback and result types for the estimator API."""

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Protocol, Union

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd

Array = NDArray[Any]
FloatArray = NDArray[np.float64]
Input = Union[str, ArrayLike, pl.Series, pl.DataFrame, "pd.Series", "pd.DataFrame"]
Frame = Union[pl.DataFrame, pl.LazyFrame, "pd.DataFrame"]
Prediction = FloatArray
NumericReducer = Union[
    Callable[[Array], Union[float, int, np.number]],
    Callable[[Array, Array], Union[float, int, np.number]],
]
ExpressionReducer = Callable[[pl.Expr], pl.Expr]
WeightedReducer = Callable[[pl.Expr, pl.Expr], pl.Expr]


class Splitter(Protocol):
    """Structural CV protocol for splitters accepting X, optional y and groups."""

    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Iterable[tuple[ArrayLike, ArrayLike]]:
        """Yield paired row indices from features and optional responses/groups."""
        ...


CV = Union[int, Splitter, Iterable[tuple[ArrayLike, ArrayLike]]]
