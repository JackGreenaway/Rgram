# Component workflows

These workflows complement the [conceptual guide](user_guide.md).

## Regressogram

Use this estimator for compact, interpretable summaries of a feature-response
relationship. [Theory and interpretation](theory.md#regressograms-fixed-cells-and-local-summaries)
and [bin/aggregation parameters](statistical_parameters.md) explain its
statistical choices. See [result schemas](api/results.md) for `bins_` and diagnostics.

## What it solves

A scatter plot can contain too many individual responses to summarize clearly.
`Regressogram` groups a feature into intervals and computes one response statistic
per interval, giving you a compact table and a piecewise constant curve.

## When to use it

Use it when interval boundaries and counts help explain the relationship, or when
you need a median, percentile, or custom per-bin statistic. Increasing the bin
count retains more local detail while reducing the observations pooled in each
cell. Compare [bin counts](gallery/bins.md) and
[response summaries](gallery/summaries.md) before settling on a configuration.

## Example: summarize a typical response

```python
import numpy as np
from rgram import Regressogram

X = np.arange(8.0).reshape(-1, 1)
y = np.array([1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 30.0])
model = Regressogram(n_bins=4, agg="median").fit(X, y)
print(model.bins_)
print(model.predict([[1.0], [5.0]]))
assert model.predict([[1.0], [5.0]]).shape == (2,)
```

The table exposes each interval's estimate and count. Here the median defines the
statistical target; it is a different summary from a mean, and a small bin still
contains little evidence. Refit with `agg="mean"` to compare the effect of the
large response. See [result schemas](api/results.md) for the table columns.


## KernelSmoother

Use this estimator to inspect a local response curve. [Kernel theory](theory.md#kernel-regression-a-moving-neighborhood)
and [bandwidth/kernel parameters](statistical_parameters.md) explain the
smoothing controls. See [result schemas](api/results.md) for diagnostics and snapshots.

## What it solves

You want to see a changing local response without imposing a global straight line
or polynomial. `KernelSmoother` pools observations with distance-based weights,
producing a smooth curve you can evaluate at chosen feature locations.

## When to use it

Use it when a continuous local mean is more useful than fixed interval summaries.
Bandwidth sets the smoothing scale in feature units. Compare bandwidths and
inspect support; choose local-linear fitting when a local slope is useful, with
the [boundary example](gallery/boundary.md) explaining its behavior.

## Example: compare smoothing scales

```python
import numpy as np
from rgram import KernelSmoother

X = np.linspace(0.0, 6.0, 40).reshape(-1, 1)
y = np.sin(X[:, 0])
query = np.linspace(0.0, 6.0, 12).reshape(-1, 1)
for bandwidth in (0.3, 1.0):
    model = KernelSmoother(
        kernel="gaussian", bandwidth="manual", bandwidth_value=bandwidth,
    ).fit(X, y)
    print(bandwidth, model.predict(query))
    assert np.isfinite(model.predict(query)).all()
```

The narrower bandwidth follows more local shape; the broader one averages across
more of the feature range. This noiseless illustration makes that difference easy
to inspect. With real data, compare residuals and observation influence as well.
The [bandwidth gallery](gallery/bandwidth.md) repeats the comparison with noise.


## CoverageSearchCV

This optional search object is deliberately narrower than `GridSearchCV`.
It uses pooled unweighted validation MSE and rejects candidates with inadequate
support in any fold. See [search result fields](api/results.md#search-results) and
[the search contracts](advanced.md#optional-cv-fit-weights-and-inspectable-splits).

## What it solves

A candidate smoother can look accurate on the rows where it predicts while
leaving other validation rows unsupported. `CoverageSearchCV` makes the required
fraction of finite predictions explicit when comparing parameter settings.

## When to use it

Use it to select a single-feature regressor by pooled validation MSE while
requiring sufficient prediction coverage in every fold. The default requires
full coverage. Use ordinary `GridSearchCV` when you need its broader scoring and
search options; this helper has its own [search contracts](advanced.md).

## Example: select a bandwidth with full coverage

```python
import numpy as np
from rgram import CoverageSearchCV, KernelSmoother

X = np.linspace(0.0, 6.0, 40).reshape(-1, 1)
y = np.sin(X[:, 0])
search = CoverageSearchCV(
    KernelSmoother(kernel="gaussian", bandwidth="manual"),
    {"bandwidth_value": [0.3, 0.6, 1.0]},
    cv=3,
    min_coverage=1.0,
    random_state=42,
).fit(X, y)
print(search.best_params_)
print(search.cv_results_)
assert np.isfinite(search.predict([[2.0], [3.0]])).all()
```

The search evaluates each bandwidth using shuffled folds and refits the winner.
Gaussian weighting usually avoids support gaps; compact kernels can expose them.
Review coverage and MSE together in `cv_results_`. The winning validation score
supports parameter selection; use a separate test set for a final performance claim.

