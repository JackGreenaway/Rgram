# Aggregations and custom reducers

`Regressogram.agg` and each optional descriptive `ci` endpoint accept named
statistics, native Polars expression functions, or the adapters below.

## Named reductions

| Name | Unweighted meaning | With observation weights |
|---|---|---|
| `mean` | Arithmetic mean | Weighted mean, undefined at zero total mass. |
| `sum` | Response total | Sum of weight × response; zero at zero mass. |
| `median` | Median | Requires an explicit weighted callback. |
| `min`, `max` | Extremal observed response | Requires an explicit weighted callback. |
| `count` | Number of non-null responses (all fitted responses are validated) | Requires an explicit weighted callback. |
| `std`, `var` | Polars sample standard deviation/variance with `ddof=1`; singleton gives null/unsupported | Requires an explicit weighted callback. |
| `first`, `last` | Response in original within-bin row order | Requires an explicit weighted callback. |

Callbacks must return one real numeric scalar per bin, or one Polars scalar
reduction expression for native callbacks. NaN is an explicit missing estimate;
infinity and nonscalar outputs are rejected. A bare callable means a native
Polars-expression callback, not an automatically detected NumPy function.

## Factory functions

```{eval-rst}
.. autofunction:: rgram.array_aggregation
```

```{eval-rst}
.. autofunction:: rgram.weighted_aggregation
```

```{eval-rst}
.. autofunction:: rgram.quantile
```

## Returned adapter objects

These classes are importable from `rgram.aggregation`. The factories are the
recommended way to create them. Their parameters participate in cloning and
nested tuning, for example `model.set_params(agg__q=0.9)` followed by refitting.
A custom callback must implement any desired zero-mass or weighted-quantile
convention explicitly; the library does not infer it.

```{eval-rst}
.. autoclass:: rgram.aggregation.ArrayAggregation
   :members:
   :inherited-members:
   :special-members: __call__
```

```{eval-rst}
.. autoclass:: rgram.aggregation.WeightedAggregation
   :members:
   :inherited-members:
   :special-members: __call__
```

```{eval-rst}
.. autoclass:: rgram.aggregation.Quantile
   :members:
   :inherited-members:
   :special-members: __call__
```

A callback may transform its independent input copies, but Rgram adds no sorting
or filtering around it. Named reducers and adapters with importable top-level
callbacks can be pickled. Lambdas and closures may not work with standard pickle.

Implementation source: {download}`aggregation.py <../../src/rgram/aggregation.py>`.
