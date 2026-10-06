"""Named fast reductions and explicit adapters for custom scalar aggregations."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from rgram._typing import ExpressionReducer, NumericReducer, WeightedReducer


class ArrayAggregation(BaseEstimator):
    """Adapt a numeric callback to a cloneable per-bin scalar reduction.

    Parameters
    ----------
    func : callable
        Function returning one real numeric scalar from values, or values and
        weights when weighted=True. Receives independent arrays in bin-row order.
    weighted : bool, default=False
        Explicitly require a two-argument callback. If fitting weights are absent,
        the callback receives ones. Unweighted callbacks reject supplied weights.

    Notes
    -----
    Usually constructed with array_aggregation. No values are sorted, filtered, or
    imputed. Callback exceptions propagate. NaN marks unsupported results; infinity
    is rejected by Regressogram. This adapter inherits sklearn get_params/set_params.
    """

    def __init__(self, func: NumericReducer, *, weighted: bool = False) -> None:
        self.func = func
        self.weighted = weighted

    def __call__(self, values: ArrayLike, weights: Optional[ArrayLike] = None) -> float:
        """Apply the numeric callback to independent input copies.

        Parameters
        ----------
        values : array-like
            Responses in one bin.
        weights : array-like or None, default=None
            Paired influence weights. A weighted adapter uses ones when absent.

        Returns
        -------
        value : float
            One real scalar. Nonscalar, complex, and imprecisely converted integer
            callback results raise ValueError.
        """
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
    """Adapt a native Polars weighted reduction.

    Parameters
    ----------
    func : callable
        Function accepting (values_expr, weights_expr) and returning one scalar
        reduction as a Polars expression. Usually use weighted_aggregation.

    Notes
    -----
    Without supplied weights, nonmissing values receive unit weights. The adapter
    inherits sklearn get_params/set_params and is cloneable.
    """

    def __init__(self, func: WeightedReducer) -> None:
        self.func = func

    def __call__(self, values: pl.Expr, weights: Optional[pl.Expr] = None) -> pl.Expr:
        """Build the weighted Polars reduction expression.

        Parameters
        ----------
        values : polars.Expr
            Expression selecting bin responses.
        weights : polars.Expr or None, default=None
            Weight expression. None gives unit weights for nonmissing values.

        Returns
        -------
        expression : polars.Expr
            The expression returned by func; Regressogram checks the result type.
        """
        if not callable(self.func):
            raise TypeError("func must be callable")
        if weights is None:
            weights = values.is_not_null().cast(pl.Float64)
        return self.func(values, weights)


class Quantile(BaseEstimator):
    """Cloneable Polars quantile reduction with explicit interpolation.

    Parameters
    ----------
    q : float
        Quantile probability in [0, 1], excluding booleans.
    interpolation : str, default='linear'
        One of nearest, higher, lower, midpoint, linear, equiprobable.
        Passed directly to Polars quantile.

    Notes
    -----
    Usually constructed with quantile. Observation weights are unsupported;
    choose an explicit weighted adapter if you require weighted quantiles.
    Inherits sklearn get_params/set_params.
    """

    def __init__(self, q: float, *, interpolation: str = "linear") -> None:
        self.q = q
        self.interpolation = interpolation

    def __call__(self, values: pl.Expr) -> pl.Expr:
        """Build an unweighted quantile expression.

        Parameters
        ----------
        values : polars.Expr
            Expression selecting bin responses.

        Returns
        -------
        expression : polars.Expr
            Quantile reduction with the configured probability and interpolation.

        Raises
        ------
        ValueError
            If q or interpolation is invalid.
        """
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
    """Wrap a NumPy/Python scalar function for agg or a ci endpoint.

    Parameters
    ----------
    func : callable
        Callback func(values), or func(values, weights) with weighted=True.
    weighted : bool, default=False
        Whether the callback explicitly accepts paired observation weights.

    Returns
    -------
    adapter : ArrayAggregation
        Cloneable reducer, retaining the callable and weighting option.

    Examples
    --------
    >>> import numpy as np
    >>> from rgram import Regressogram, array_aggregation
    >>> model = Regressogram(n_bins=1, agg=array_aggregation(np.median))
    >>> model.fit([0., 1., 2.], [1., 3., 9.]).predict([1.])
    array([3.])
    """
    return ArrayAggregation(func, weighted=weighted)


def weighted_aggregation(func: WeightedReducer) -> WeightedAggregation:
    """Wrap a native weighted Polars scalar reduction.

    Parameters
    ----------
    func : callable
        Callback func(values_expr, weights_expr) returning a Polars expression.

    Returns
    -------
    adapter : WeightedAggregation
        Cloneable native reducer. With no fitting weights, the adapter uses ones.

    Examples
    --------
    >>> from rgram import Regressogram, weighted_aggregation
    >>> reducer = weighted_aggregation(lambda values, weights:
    ...     (values * weights).sum() / weights.sum())
    >>> model = Regressogram(n_bins=1, agg=reducer)
    >>> model.fit([0., 1.], [2., 8.], sample_weight=[1., 2.]).predict([0.5])
    array([6.])
    """
    return WeightedAggregation(func)


def quantile(q: float, *, interpolation: str = "linear") -> Quantile:
    """Create an unweighted native Polars quantile reducer.

    Parameters
    ----------
    q : float
        Probability in [0, 1]. Validated when the reducer is evaluated.
    interpolation : str, default='linear'
        nearest, higher, lower, midpoint, linear, or equiprobable, passed to Polars.

    Returns
    -------
    reducer : Quantile
        Cloneable quantile callable for agg or either descriptive ci endpoint.

    Examples
    --------
    >>> from rgram import Regressogram, quantile
    >>> model = Regressogram(n_bins=1, agg=quantile(0.5))
    >>> model.fit([0., 1., 2.], [1., 3., 9.]).predict([1.])
    array([3.])
    """
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
    """Resolve a named/native reduction into a Polars expression.

    This is an implementation helper; users should configure Regressogram.agg.
    Numeric adapters run directly per bin and are rejected by this resolver.

    Parameters
    ----------
    aggregation : str or callable
        Declared native aggregation or expression adapter.
    values : polars.Expr
        Bin-response expression.
    weights : polars.Expr or None, default=None
        Observation-weight expression. Only mean/sum names or explicit weighted
        adapters accept weights.

    Returns
    -------
    expression : polars.Expr
        One declared scalar reduction. Invalid callback types/signatures raise.
    """
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
