# User guide

```{toctree}
:maxdepth: 2

theory
sklearn
statistical_parameters
advanced
```

## Choose the summary you need

Use `Regressogram` when feature intervals and a small table of counts and local
response summaries make the relationship easiest to explain. Choose
`KernelSmoother` when a smooth curve over the observed feature range is more
useful. Mean aggregation describes a local average; a bin median or quantile
answers a different question about the response distribution.

| Goal | Model or option | Guide |
|---|---|---|
| Piecewise constant response summary | `Regressogram`, `agg="mean"` | [Bin rules and aggregation](statistical_parameters.md) |
| Typical response or percentile within bins | `agg="median"` or `agg=quantile(q)` | [Aggregation API](api/aggregation.md) |
| Smooth response curve | `KernelSmoother`, local constant | [Theory](theory.md) |
| Smooth curve with a local slope | `regression="local_linear"` | [Boundary behavior](theory.md#local-linear-regression-and-boundaries) |
| Inspect the fit | `bins_`, `predict_diagnostics`, `regression_diagnostics`, `get_weights` | [Result schemas](api/results.md) |
| Display residuals | `plot_diagnostics` | [Plotting API](api/plotting.md) |
| Bootstrap pointwise curve intervals | `predict_interval` | [Uncertainty assumptions](theory.md#uncertainty-dependence-and-evaluation) |
| Preprocessing and held-out evaluation | sklearn `Pipeline`, `GridSearchCV` | [scikit-learn guide](sklearn.md) |
| Select parameters while requiring prediction support | `CoverageSearchCV` | [Search API](api/coverage_search.md) |

## Exploratory workflow

1. Choose a numeric feature and response and inspect the observations.
2. Fit on the observations you want to describe; select subgroups explicitly.
3. Compare several bin counts or bandwidths to assess smoothing sensitivity.
4. Inspect local counts/weights and residuals alongside the curve.
5. Record the summary, smoothing settings, feature range, and sampling assumptions.

Repeat for other features rather than passing a multivariate feature matrix.
A fit on all rows is a descriptive summary. Use held-out evaluation if you want
to claim predictive performance. Separate subgroup curves need enough local data
and overlapping feature ranges to support a meaningful comparison.

## Input, ordering, and missing data

Use a numeric one-column feature array/DataFrame and a numeric single response.
One-dimensional feature inputs remain convenient for exploration. With pandas or Polars,
column-name selection through `data=` is also available. NaN, null, infinity,
complex values, and multivariate inputs are rejected. Impute missing features
explicitly in a pipeline if that is appropriate; decide how to handle missing
responses before fitting.

Training snapshots and prediction order are preserved, including duplicates.
No automatic sorting or row dropping occurs. Unsorted warnings are advisory:
use Python warning filters if unordered data are normal for your analysis.
Sort a **separate paired plotting view** explicitly if you want to connect points
into an ordered line. See [data integrity contracts](advanced.md).

## Weights and custom statistics

`sample_weight` controls observation influence. It does not automatically supply
survey-design inference, robust fitting, or weighted validation scoring. Zero-weight
rows remain in stored data and influence feature-based bin/bandwidth selection.

Use named aggregations for simple summaries, `quantile` for an explicit percentile,
`array_aggregation` for a NumPy/Python scalar function, or `weighted_aggregation`
for a native Polars expression with weights. See [the full adapter reference](api/aggregation.md)
for callable contracts and cloning behavior.

## Unsupported locations and extrapolation

A bin without an estimate or a query without positive kernel weight is unsupported.
The default returns NaN with a warning; `unsupported="raise"` fails explicitly.
Ordinary sklearn scorers reject non-finite predictions, so inspect coverage or
choose parameters with broader support when evaluating prediction error.

Outside-range queries have a separate `extrapolation` policy. A finite estimate
outside the observed range is a model calculation, not evidence for the
relationship there. Diagnostics distinguish support from range membership.
See [theory and suitability](theory.md) for interpretation and
[the parameter reference](statistical_parameters.md) for exact defaults.
