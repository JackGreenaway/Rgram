# Implementation guide

This guide covers every package module and the responsibilities of its helpers.
The supported user interface is the [API reference](api/index.md). Private names,
internal state and module functions can change without being additional
public estimator features. Source links on API pages expose the implementation.

## Package map

| Module | Responsibility |
|---|---|
| `rgram.__init__` | Top-level exports in `__all__`; importing does not load Matplotlib. |
| `rgram.rgram` | Regressogram configuration, bin selection, aggregation and ordered prediction joins. |
| `rgram.smoothing` | Kernel shapes, local coefficients, algorithm selection and query diagnostics. |
| `rgram.base` | Shared input validation, snapshots, names, warnings, diagnostics and bootstrap delegation. |
| `rgram.aggregation` | Named reductions, numeric/native adapters, quantile helper and expression resolution. |
| `rgram.model_selection` | Serial candidate/fold evaluation, coverage eligibility and refit. |
| `rgram.uncertainty` | Paired resampling and interval calculations. |
| `rgram.plotting` | Optional Matplotlib diagnostic panels. |
| `rgram.warnings` | Filterable advisory categories. |
| `rgram._typing` | Shared input, output, callback and CV annotations. |

## Estimator internals

### Regressogram

`_validate_parameters` enforces the declared configuration and callback contracts.
`_cv_bin_count` clones the model and uses `CoverageSearchCV` when `n_bins="cv"`.
`_learn_bin_params` learns feature bounds, counts, quantiles/widths and merged
boundaries. `_predict_bin_expr` maps queries to those fitted groups with each
strategy's equality/edge convention. `_prediction_frame` validates queries,
joins group summaries without reordering, and applies range/support policies.

`fit` snapshots the selected columns and computes native aggregations in Polars.
Numeric adapters run directly on group arrays in Python, so their exceptions
remain normal Python exceptions rather than crossing a Rust callback boundary.


### KernelSmoother

`_positive` and `_validate_parameters` validate configuration.
`_check_fitted` uses the sklearn fitted-state contract. `_normalized_kernel`
calculates each shape, includes observation weights, preserves exact zeros, and
uses log-scale evaluation for infinite-support built-ins. `_resolve_algorithm`
chooses support windows only for already sorted training features and compact
kernels. `_kernel_rows` yields bounded dense blocks or exact local windows;
it never sorts a dataset or index.

`_prediction_weights` solves the local-linear moments, choosing the explicit
singular policy when needed. `_handle_unsupported` applies NaN/error policy.
`_evaluate` coordinates range policy, coefficients, predictions, support counts,
and optional dense inspection weights. `_KERNELS` and `_COMPACT_KERNELS` declare
supported names and the kernels eligible for windows. Public inspection uses
`bandwidth_`.

### CoverageSearchCV

`fit` validates explicit fold indices, clones each candidate in each fold,
accumulates pooled squared error and support, rejects ineligible candidates,
and refits the first best candidate. `__sklearn_is_fitted__` checks for a winner,
so `cv_results_` alone does not make a failed search fitted.

## Shared utilities

The private `BaseUtils` class supplies implementation helpers, not a standalone
user estimator. Validation accepts arrays or selected pandas/Polars columns, materializes
a single snapshot, rejects invalid rows, and preserves dtype/order. Public
methods inherited by estimators are documented on their API pages. The complete
utility reference below includes private helpers for contributors.

```{eval-rst}
.. autoclass:: rgram.base.BaseUtils
   :members:
   :private-members:
   :special-members: __sklearn_is_fitted__
   ```

## Aggregation and resampling internals

The named reduction tuple `AGGREGATIONS` contains mean, median, min, max, sum,
count, std, var, first, and last. `Aggregation` is the union of declared names,
native-expression functions, numeric adapters, and weighted-expression adapters.
Public adapter classes and their calls are covered in [the aggregation reference](api/aggregation.md).

```{eval-rst}
.. autofunction:: rgram.aggregation.aggregation_expr
   ```

```{eval-rst}
.. automodule:: rgram.uncertainty
   :members:
   :private-members:
   ```

`bootstrap_interval` fixes the selected smoother bandwidth (including its
adjustment), or the fitted bin count, before drawing replicas. Explicit bin width
stays explicit. Bin boundaries still refit. Draw storage is dense in
`n_resamples × n_queries`. Strict fit/prediction errors are preserved.

## Shared type annotations

| Name in `rgram._typing` | Meaning |
|---|---|
| `Array`, `FloatArray` | General NumPy array and float64 computation/result array. |
| `Input` | Column-name string, array-like, Polars Series or DataFrame. |
| `Frame` | Pandas/Polars DataFrame or Polars LazyFrame for named selection. |
| `Prediction` | One-dimensional float64 prediction array. |
| `NumericReducer` | One- or two-array callback returning a scalar. |
| `ExpressionReducer` | Polars expression-to-expression callback. |
| `WeightedReducer` | Two-expression callback returning a reduction expression. |
| `Splitter` | Protocol with `split(X, y=None, groups=None)` yielding train/test indices. |
| `CV` | Integer, splitter, or iterable of train/test index pairs. |

These annotations do not add support for arbitrary multivariate or sparse data.
Actual validation contracts are in the estimator reference.

## Examples and validation

`examples/kernel_explorer.py` builds interactive Matplotlib controls through
`main`, with `update` and `show_interval` callbacks. It generates ordered
synthetic observations; interval calculation is explicitly triggered.
`examples/benchmark_neighbors.py` times fit and query paths separately and verifies
agreement. Neither module runs its main routine on import. The
[examples page](examples.md) documents their execution and intended interpretation.

`tests/` contains regression, numerical, property, integration, and performance
checks. `scripts/check_docs.py` checks documentation coverage and rendered local
links; `scripts/check_examples.py` executes guide snippets and public doctests; it is a development tool, not an installed library API. See
[development instructions](development.md) for build/test commands and publishing.

`examples/plot_relationships.py` supplies nine independent gallery functions and
a CLI to show them one at a time or save separate large figures. Shared helpers
provide consistent axes, observations, legends, and smoother configuration.
The documentation example check draws every gallery figure headlessly.

### Pandas conversion

`BaseUtils._is_pandas` recognizes actual DataFrame/Series objects and subclasses
through the already loaded optional pandas module. Rgram's conversion helpers
do not import pandas; the dependency remains optional. `_as_array` converts complete numeric pandas columns to
NumPy, including nullable numeric dtypes, without index alignment or integer
rounding. `_pandas_frame` converts only selected, uniquely labeled columns into
the internal Polars snapshot. It does not call `pl.from_pandas` or require Arrow.
The same array normalization serves fitting, queries, targets, and weights.
