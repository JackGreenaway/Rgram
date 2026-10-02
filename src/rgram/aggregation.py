"""Named fast reductions and explicit adapters for custom scalar aggregations."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from rgram._typing import ExpressionReducer, NumericReducer, WeightedReducer


class ArrayAggregation(BaseEstimator):
    """Adapt func(values) or func(values, weights) to a per-bin numeric reduction.

    Each argument is a copy of a bin's rows in their original relative order.
    The function must return one real numeric scalar. No values are sorted,
    filtered or imputed by this adapter. NumPy/Python callbacks are usually
    slower than native Polars expressions. Set weighted=True explicitly for
    a two-argument callable; provided weights are never silently ignored.
    """

    def __init__(self, func: NumericReducer, *, weighted: bool = False) -> None:
        self.func = func
        self.weighted = weighted

    def __call__(self, values: ArrayLike, weights: Optional[ArrayLike] = None) -> float:
        if not callable(self.func):
            raise TypeError("func must be callable")
        if not isinstance(self.weighted, bool):
            raise TypeError("weighted must be boolean")
        if weights is not None and not self.weighted:
            raise ValueError(
                "sample_weight requires array_aggregation(..., weighted=True)"
            )
        arguments = [np.asarray(values).copy()]
        if self.weighted:
            arguments.append(
                np.ones(len(arguments[0]))
                if weights is None
                else np.asarray(weights).copy()
            )
        result = np.asarray(self.func(*arguments))
        if (
            result.ndim != 0
            or not np.issubdtype(result.dtype, np.number)
            or np.iscomplexobj(result)
        ):
            raise ValueError(
                "Custom aggregation must return one real numeric scalar per bin"
            )
        scalar = float(result)
        if np.issubdtype(result.dtype, np.integer) and int(result) != int(scalar):
            raise ValueError(
                "Custom aggregation would lose integer precision in float64"
            )
        return scalar


class WeightedAggregation(BaseEstimator):
    """Adapt a native Polars func(values_expr, weights_expr) reduction."""

    def __init__(self, func: WeightedReducer) -> None:
        self.func = func

    def __call__(self, values: pl.Expr, weights: Optional[pl.Expr] = None) -> pl.Expr:
        if not callable(self.func):
            raise TypeError("func must be callable")
        if weights is None:
            weights = values.is_not_null().cast(pl.Float64)
        return self.func(values, weights)


class Quantile(BaseEstimator):
    """A native Polars quantile aggregation, with explicit interpolation."""

    def __init__(self, q: float, *, interpolation: str = "linear") -> None:
        self.q = q
        self.interpolation = interpolation

    def __call__(self, values: pl.Expr) -> pl.Expr:
        if (
            isinstance(self.q, (bool, str))
            or not np.isscalar(self.q)
            or not 0 <= self.q <= 1
        ):
            raise ValueError("q must be in [0, 1]")
        if self.interpolation not in (
            "nearest",
            "higher",
            "lower",
            "midpoint",
            "linear",
            "equiprobable",
        ):
            raise ValueError("Unsupported quantile interpolation")
        return values.quantile(self.q, interpolation=self.interpolation)


def array_aggregation(
    func: NumericReducer, *, weighted: bool = False
) -> ArrayAggregation:
    """Wrap an ordinary numeric function for agg or either ci endpoint."""
    return ArrayAggregation(func, weighted=weighted)


def weighted_aggregation(func: WeightedReducer) -> WeightedAggregation:
    """Wrap a native Polars two-expression weighted reduction."""
    return WeightedAggregation(func)


def quantile(q: float, *, interpolation: str = "linear") -> Quantile:
    """Build a quantile reducer, e.g. Regressogram(agg=quantile(0.9))."""
    return Quantile(q, interpolation=interpolation)


AGGREGATIONS = (
    "mean",
    "median",
    "min",
    "max",
    "sum",
    "count",
    "std",
    "var",
    "first",
    "last",
)


def aggregation_expr(
    aggregation: Aggregation, values: pl.Expr, weights: Optional[pl.Expr] = None
) -> pl.Expr:
    """Resolve a declared reduction without guessing its callable signature."""
    if isinstance(aggregation, str):
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {aggregation!r}; choose from {AGGREGATIONS} or supply a callable"
            )
        if weights is not None:
            if aggregation not in ("mean", "sum"):
                raise ValueError(
                    f"sample_weight is not defined for agg={aggregation!r}; use an explicit weighted callable"
                )
            total = weights.sum()
            value = (values * weights).sum()
            if aggregation == "mean":
                return pl.when(total > 0).then(value / total).otherwise(None)
            return value
        return getattr(values, aggregation)()
    if isinstance(aggregation, ArrayAggregation):
        raise TypeError(
            "Numeric adapters are evaluated directly on each bin, not as Polars expressions"
        )
    if isinstance(aggregation, WeightedAggregation):
        expression = aggregation(values, weights)
    elif callable(aggregation):
        if weights is not None:
            raise ValueError(
                "sample_weight requires agg='mean', agg='sum', or an explicit weighted aggregation adapter"
            )
        expression = aggregation(values)
    else:
        raise TypeError("agg must be a named aggregation or callable")
    if not isinstance(expression, pl.Expr):
        raise TypeError(
            "Aggregation must return a Polars expression; wrap numeric functions with array_aggregation(func)"
        )
    return expression


Aggregation = Union[str, ExpressionReducer, ArrayAggregation, WeightedAggregation]
