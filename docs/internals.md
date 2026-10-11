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

<!-- BEGIN REFERENCE:rgram.base.BaseUtils -->
<span id="rgram.base.BaseUtils"></span>

## rgram.base.BaseUtils

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/base.py#L16">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">rgram.base.</span><span class="api-name">BaseUtils</span>(

)</code></pre></div>


<div class="api-class-body" markdown="1">

Shared validation and inspection methods for univariate regressors.


### Methods

| Method | Purpose |
| --- | --- |
| [predict_interval](#rgram.base.BaseUtils.predict_interval) | Compute pointwise paired-bootstrap confidence intervals for the curve. |
| [regression_diagnostics](#rgram.base.BaseUtils.regression_diagnostics) | Inspect observed values, fitted values, and residuals in input order. |

<span id="rgram.base.BaseUtils.predict_interval"></span>

### predict_interval {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/base.py#L606">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.base.BaseUtils.</span><span class="api-name">predict_interval</span>(
    X,
    *,
    confidence_level=0.95,
    n_resamples=200,
    random_state=None,
    method=&#x27;percentile&#x27;,
    resampling=&#x27;iid&#x27;,
    groups=None,
    block_size=None,
    min_valid_fraction=1.0,
)</code></pre></div>


Compute pointwise paired-bootstrap confidence intervals for the curve.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_queries,) or (n_queries, 1)</span></dt><dd><p>Query locations; order and duplicates are preserved.</p></dd>
<dt><code class="api-field-name">confidence_level</code> <span class="api-field-type">float, default=0.95</span></dt><dd><p>Nominal interval level strictly between zero and one.</p></dd>
<dt><code class="api-field-name">n_resamples</code> <span class="api-field-type">int, default=200</span></dt><dd><p>Bootstrap refits, at least two. More draws improve Monte Carlo stability.</p></dd>
<dt><code class="api-field-name">random_state</code> <span class="api-field-type">int or None, default=None</span></dt><dd><p>Seed for resampling, independent of any estimator CV seed.</p></dd>
<dt><code class="api-field-name">method</code> <span class="api-field-type">{&#x27;percentile&#x27;, &#x27;basic&#x27;}, default=&#x27;percentile&#x27;</span></dt><dd><p>Bootstrap quantiles, or quantiles reflected around the original estimate.</p></dd>
<dt><code class="api-field-name">resampling</code> <span class="api-field-type">{&#x27;iid&#x27;, &#x27;groups&#x27;, &#x27;blocks&#x27;}, default=&#x27;iid&#x27;</span></dt><dd><p>Resample paired independent rows, whole independent groups, or contiguous moving blocks in original training order. No sorting occurs.</p></dd>
<dt><code class="api-field-name">groups</code> <span class="api-field-type">array-like of shape (<code>n_samples_in_</code>,) or None, default=None</span></dt><dd><p>Nonmissing group labels, required only for group resampling. At least two distinct groups are required.</p></dd>
<dt><code class="api-field-name">block_size</code> <span class="api-field-type">int or None, default=None</span></dt><dd><p>Block length from 2 to <code>n_samples_in_</code> - 1, required only for blocks.</p></dd>
<dt><code class="api-field-name">min_valid_fraction</code> <span class="api-field-type">float, default=1.0</span></dt><dd><p>Fraction of draws that must support each query, in (0, 1]. At least two valid draws are always required. Lower fractions condition on valid draws.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">interval</code> <span class="api-field-type">polars.DataFrame</span></dt><dd><p>Columns: x, prediction, lower, upper, n_valid, bootstrap_coverage. Insufficiently supported bounds are NaN. Queries are never dropped.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If no successful fit is available.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>For invalid options, fewer than two training rows, invalid group/block inputs, or errors from the estimator&#x27;s strict fit/prediction policies.</p></dd>
</dl>

#### Notes

These are curve intervals, not prediction intervals or simultaneous bands.
IID draws assume independent random-design rows. Group and block draws require
appropriate independent groups or ordered dependence. Smoothing settings are
held fixed; bin boundaries are relearned. Selection uncertainty and smoothing
bias are not corrected. Memory includes n_resamples * n_queries draw values.
A draw with no positive observation weight counts as unsupported. Other fit
errors propagate. Explicit neighbor-only algorithms can reject unsorted draws.


</div>

<span id="rgram.base.BaseUtils.regression_diagnostics"></span>

### regression_diagnostics {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/base.py#L568">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.base.BaseUtils.</span><span class="api-name">regression_diagnostics</span>(X, y)</code></pre></div>


Inspect observed values, fitted values, and residuals in input order.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, 1)</span></dt><dd><p>One evaluation feature. Named inputs must match the fitted feature.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, 1)</span></dt><dd><p>Observed response, aligned with x. May be training or held-out data.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">diagnostics</code> <span class="api-field-type">polars.DataFrame</span></dt><dd><p>Columns: x, observed, prediction, residual (observed minus prediction), supported (finite prediction). Unsupported rows remain present.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If no successful fit is available.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>For invalid, misaligned inputs or strict prediction policies.</p></dd>
</dl>


</div>

</div>
<!-- END REFERENCE:rgram.base.BaseUtils -->

## Aggregation and resampling internals

The named reduction tuple `AGGREGATIONS` contains mean, median, min, max, sum,
count, std, var, first, and last. `Aggregation` is the union of declared names,
native-expression functions, numeric adapters, and weighted-expression adapters.
Public adapter classes and their calls are covered in [the aggregation reference](api/aggregation.md).

<!-- BEGIN REFERENCE:rgram.aggregation.aggregation_expr -->
<span id="rgram.aggregation.aggregation_expr"></span>

## rgram.aggregation.aggregation_expr

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L276">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.</span><span class="api-name">aggregation_expr</span>(
    aggregation,
    values,
    weights=None,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Resolve a named/native reduction into a Polars expression.

This is an implementation helper; users should configure Regressogram.agg.
Numeric adapters run directly per bin and are rejected by this resolver.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">aggregation</code> <span class="api-field-type">str or callable</span></dt><dd><p>Declared native aggregation or expression adapter.</p></dd>
<dt><code class="api-field-name">values</code> <span class="api-field-type">polars.Expr</span></dt><dd><p>Bin-response expression.</p></dd>
<dt><code class="api-field-name">weights</code> <span class="api-field-type">polars.Expr or None, default=None</span></dt><dd><p>Observation-weight expression. Only mean/sum names or explicit weighted adapters accept weights.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">expression</code> <span class="api-field-type">polars.Expr</span></dt><dd><p>One declared scalar reduction. Invalid callback types/signatures raise.</p></dd>
</dl>


</div>
<!-- END REFERENCE:rgram.aggregation.aggregation_expr -->

[Resampling implementation](https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/uncertainty.py).

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
