# Custom aggregations and smoothers

Use a custom aggregation when a standard mean or median does not express the
response statistic you need. Use a custom kernel when you want to change how
distance determines local influence. These examples use small deterministic data
and can be run independently. See the [aggregation API](api/aggregation.md) and
[smoother API](api/kernel_smoother.md) for the complete contracts.

## Reduce a bin with a Polars expression

A bare aggregation callback receives a `polars.Expr` and returns one scalar
reduction expression. Here the statistic is root mean square: it measures
response magnitude and is sensitive to large values.

```python
import numpy as np
from rgram import Regressogram

def root_mean_square(values):
    return (values ** 2).mean().sqrt()

X = np.arange(4.0).reshape(-1, 1)
y = np.array([0.0, 0.0, 3.0, 4.0])
model = Regressogram(n_bins=1, agg=root_mean_square).fit(X, y)
np.testing.assert_allclose(model.predict([[1.5]]), [2.5])
print(model.bins_)
```

The callback runs in the native Polars aggregation engine. It must reduce the
bin to one real numeric value; returning the original expression without a
reduction produces a nonscalar result. A NumPy function needs the adapter below.

## Use a NumPy statistic with explicit trimming

`array_aggregation` passes a copy of each occupied bin's responses to a Python
function. This example discards one value from each end when the bin has at
least five observations, then averages the remaining values.

```python
import numpy as np
from rgram import Regressogram, array_aggregation

def trimmed_mean(values):
    if values.size < 5:
        return float(np.mean(values))
    ordered = np.sort(values)
    return float(np.mean(ordered[1:-1]))

X = np.arange(6.0).reshape(-1, 1)
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
trimmed = Regressogram(n_bins=1, agg=array_aggregation(trimmed_mean)).fit(X, y)
ordinary = Regressogram(n_bins=1, agg="mean").fit(X, y)
np.testing.assert_allclose(trimmed.predict([[2.5]]), [3.5])
assert ordinary.predict([[2.5]])[0] > trimmed.predict([[2.5]])[0]
```

Trimming changes the statistical target. Rgram supplies values in original
within-bin order; the sorting and removal here are explicit choices in the
callback. Numeric callbacks run once per occupied bin in Python, so prefer a
native expression when it can express your statistic efficiently.

## Define how observation weights affect a statistic

Weights need an explicit convention for a custom statistic. The two adapters
below calculate the same weighted root mean square, using NumPy and Polars
respectively. Both return NaN when the total weight is zero.

```python
import numpy as np
import polars as pl
from rgram import Regressogram, array_aggregation, weighted_aggregation

def weighted_rms_array(values, weights):
    mass = weights.sum()
    return float(np.sqrt(np.sum(weights * values ** 2) / mass)) if mass > 0 else np.nan

def weighted_rms_expr(values, weights):
    mass = weights.sum()
    return pl.when(mass > 0).then(
        ((values ** 2 * weights).sum() / mass).sqrt()
    ).otherwise(float("nan"))

X = np.arange(3.0).reshape(-1, 1)
y = np.array([3.0, 4.0, 100.0])
weights = np.array([1.0, 3.0, 0.0])
reducers = (
    array_aggregation(weighted_rms_array, weighted=True),
    weighted_aggregation(weighted_rms_expr),
)
for reducer in reducers:
    model = Regressogram(n_bins=1, agg=reducer).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(model.predict([[1.0]]), [np.sqrt(57.0 / 4.0)])
```

Zero-weight rows still reach your callback; this statistic gives them zero
contribution. With no supplied `sample_weight`, weighted adapters receive ones.
NaN follows the estimator's `unsupported` policy. Unweighted adapters do not
automatically acquire weighted semantics when fitting weights are supplied.

## Add descriptive response endpoints

Custom reductions also work as the two `ci` endpoints. Here a NumPy median is
paired with lower and upper response quartiles.

```python
import numpy as np
from rgram import Regressogram, array_aggregation, quantile

X = np.arange(5.0).reshape(-1, 1)
y = np.array([1.0, 2.0, 3.0, 4.0, 20.0])
model = Regressogram(
    n_bins=1,
    agg=array_aggregation(np.median),
    ci=(quantile(0.25), quantile(0.75)),
).fit(X, y)
np.testing.assert_allclose(model.predict([[2.0]]), [3.0])
print(model.bins_)
print(model.predict_diagnostics([[2.0]]))
```

These endpoints describe the spread of responses in a bin. They are not
confidence bounds for the median. Use `predict_interval` for bootstrap
uncertainty about the fitted statistic, with the sampling assumptions described
in [advanced usage](advanced.md#bootstrap-intervals-and-their-assumptions).
The `quantile` helper is unweighted; weighted quantiles require your own explicit
weighted convention and callback.

## Supply a custom kernel and compare local fits

A kernel callback receives a Polars expression for standardized distance
`u = (training_feature - query) / bandwidth_`. It must return a nonnegative,
finite weight for every distance, rather than reducing the expression. This
quartic compact kernel gives zero weight outside one bandwidth.

```python
import numpy as np
import polars as pl
from rgram import KernelSmoother

def quartic_kernel(u):
    return pl.when(u.abs() < 1.0).then((1.0 - u ** 2) ** 2).otherwise(0.0)

X = np.linspace(0.0, 4.0, 41).reshape(-1, 1)
y = 1.0 + 2.0 * X[:, 0]
query = np.array([[0.0], [2.0], [4.0]])
for regression in ("local_constant", "local_linear"):
    model = KernelSmoother(
        kernel=quartic_kernel,
        bandwidth="manual",
        bandwidth_value=0.8,
        regression=regression,
        algorithm="brute",
    ).fit(X, y)
    prediction = model.predict(query)
    assert np.isfinite(prediction).all()
    if regression == "local_linear":
        np.testing.assert_allclose(prediction, 1.0 + 2.0 * query[:, 0], atol=1e-12)
    print(regression, prediction)
    print(model.predict_diagnostics(query))
```

Local constant fitting averages nearby responses and bends away from this
straight line at the boundaries. Local linear fitting reproduces the line here
because each neighborhood supports a slope. On noisy data, that extra flexibility
can increase variance; inspect diagnostics for singular-fit fallbacks.

Kernel weights are multiplied by observation weights supplied to `fit` and
normalized by the estimator. Custom kernels use bounded dense computation;
`algorithm="neighbors"` is available only for eligible named compact kernels.
A compact kernel can leave queries without any positive weight, even inside the
training range. Check support diagnostics and choose `unsupported="raise"` if
such gaps should fail explicitly. See the [support example](gallery/support.md)
and [boundary comparison](gallery/boundary.md) for further exploration.

Top-level named callbacks, as used here, also work with standard pickle.
The aggregation adapters support scikit-learn cloning; see
[estimators and pipelines](sklearn.md) for fitting and parameter-search workflows.
