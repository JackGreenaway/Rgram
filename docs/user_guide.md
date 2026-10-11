# User guide

Choose the question before the model: a bin summary explains intervals, a smooth
curve describes local shape, and a validation workflow evaluates prediction.
This guide connects those goals to the settings and diagnostics you can inspect.

- [Theory](theory.md)
- [Estimators and pipelines](sklearn.md)
- [Statistical Parameters](statistical_parameters.md)
- [Advanced](advanced.md)

## Configure the classes

The [class settings guide](statistical_parameters.md) explains every constructor
setting, its default, available choices, and its effect on the fit. Start with
the walkthrough below, then use the tables for individual options:

- [Regressogram settings](statistical_parameters.md#regressogram-parameters):
  binning, response statistics, bin counts and widths, validation, and prediction
  policies.
- [KernelSmoother settings](statistical_parameters.md#kernelsmoother-parameters):
  bandwidth, kernel, local regression, support, extrapolation, and computation.
- [CoverageSearchCV settings](statistical_parameters.md#optional-cross-validation-parameters):
  candidate models, parameter grids, fold design, and minimum prediction coverage.
- [Aggregation helper settings](statistical_parameters.md#aggregation-helper-classes):
  custom reducers, weighting conventions, and quantile interpolation.

### Set up a regressogram

Choose `binning` first: `"dist"` adapts widths to observation density, `"width"`
keeps feature intervals equally wide, `"none"` groups exact values, and `"int"`
groups values after truncation toward zero. For distribution or width binning,
choose `n_bins`; alternatively, width binning accepts `bin_width` with
`n_bins=None`. More bins or narrower widths expose detail but leave fewer
observations in each estimate.

Next choose the response summary with `agg`. Use `ci` when you also want two
descriptive endpoints, such as response percentiles within each bin. These
endpoints describe response spread; bootstrap curve intervals are a separate
operation. `max_bins` bounds allocation, while `min_samples_bin` guides the
default distribution count and automatic CV candidates without guaranteeing a
minimum occupancy in every bin.

Set `n_bins="cv"` to select counts using held-out responses. Only then do `cv`,
`bin_candidates`, and `random_state` control internal selection. Finally choose
`extrapolation` for queries outside the observed range and `unsupported` for bins
without a usable estimate. These policies address different conditions.

### Set up a kernel smoother

Choose `bandwidth` to use an automatic feature-based rule or `"manual"` with
`bandwidth_value`. `bandwidth_adjust` multiplies either choice. Smaller widths
give nearby observations more concentrated influence; larger widths smooth over
more of the relationship. Choose `kernel` to control how influence falls with
distance. Compact kernels can leave gaps when the width is too small.

Choose `regression="local_constant"` for a local weighted average or
`"local_linear"` for a local fitted line. With local-linear fitting, `singular`
decides whether an unresolved slope falls back to a constant or raises.
`unsupported` controls queries without usable local weight; `extrapolation`
controls outside-range queries even when they have support.

The remaining settings control evaluation: `algorithm` chooses the calculation
path, `batch_size` limits brute-force query batches, and `n_eval_samples` sets
the default grid resolution. Increasing grid resolution adds evaluation points;
change bandwidth to alter smoothing.

### Set up a coverage-aware search

Pass an `estimator` and a `param_grid` of settings to compare. `cv` defines the
validation design, and `random_state` seeds its shuffled folds when `cv` is an
integer. `min_coverage` requires a minimum fraction of finite predictions in
every fold before a candidate can win. Keep the default full coverage unless
scoring only supported rows answers your intended question. The winner is
always refitted on all training rows.

Constructor settings can be inspected with `get_params()` and changed with
`set_params(...)`. Refit after changing model settings. Fitted attributes such as
`n_bins_` and `bandwidth_` report what was learned; they are results rather than
constructor options. The settings guide also distinguishes fitting inputs and
method options, including weights, diagnostics, and bootstrap intervals.

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
