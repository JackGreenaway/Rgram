# Rgram

Rgram provides univariate regressograms and kernel regression with a scikit-learn-style estimator API. It accepts numeric arrays or selected Polars columns, preserves observation and prediction order, and makes data handling, aggregation, support and uncertainty explicit.

This README describes the current working tree. The [advanced guide](docs/advanced.md) explains numerical and statistical details, and the [review and roadmap](docs/review.md) records completed improvements and remaining work.

## Installation

From a checkout, install the locked development environment and run the tests:

```bash
uv sync
uv run pytest -q
```

Matplotlib is optional for library users. Install the plotting extra with `pip install 'rgram[plot]'`, or use `uv sync --extra plot` from the checkout. NumPy, Polars and scikit-learn are core dependencies. Hypothesis and pytest are development dependencies.

## Quick start without cross-validation

Cross-validation is not required and does not run during either of these fits. A fixed bin count or automatic rule fits directly to every training observation. A kernel bandwidth rule also fits directly, without splitting the dataset.

```python
import numpy as np
from rgram import KernelSmoother, Regressogram

x = np.array([0., 1., 2., 3., 4.])
y = np.array([0., 2., 1., 4., 3.])

binned = Regressogram(n_bins=2, agg="mean").fit(x, y)
smoothed = KernelSmoother(
    kernel="gaussian", bandwidth="manual", bandwidth_value=0.8,
).fit(x, y)

print(binned.predict([0.5, 2.5]))
print(smoothed.predict_diagnostics([0.5, 2.5]))
print(binned.data_summary_)
```

Both estimators return `self` from `fit`, support `get_params`, `set_params`, cloning, pipelines and R² `score`, and expose fitted attributes with a trailing underscore. Refit after changing parameters that determine the fitted model. They support one feature and one target; arrays may have shape `(n_samples,)` or `(n_samples, 1)`.

For Polars data, select the columns explicitly. A LazyFrame is materialized once for the selected feature and target so later changes to its source cannot change a fitted model.

```python
import polars as pl

frame = pl.DataFrame({"temperature": x, "demand": y})
model = Regressogram(n_bins=2).fit("temperature", "demand", data=frame.lazy())
print(model.feature_names_in_)
print(model.predict(pl.DataFrame({"temperature": [1.5, 2.5]})))
```

Named prediction inputs must match the fitted feature name. An unnamed array is also accepted because the estimator has exactly one feature. A refit with unnamed data clears the previous feature name.

## What happens to the data

Rgram does not sort training rows, query rows or a training index. It does not silently remove duplicate observations, drop missing rows, impute values, shuffle a normal fit, trim outliers, or change caller-owned arrays. Null, NaN, infinite, complex and multivariate inputs are rejected rather than cleaned automatically. Predictions retain query order and duplicates, including unsupported rows represented by NaN when that policy is selected.

The following computations are part of the estimator, and are exposed rather than hidden:

| Operation | Behavior and inspection |
|---|---|
| Input snapshot | `X_`, `y_` and `sample_weight_` are independent copies in original row order. Sample weights retain their supplied values. |
| Numeric calculations | Regression calculations use float64 buffers. Integer or wider-float values that would lose precision are rejected with an instruction to convert or rescale explicitly. |
| Binning | The chosen strategy assigns observations to cells without modifying them. `bins_` contains occupied cells in first-observed order, their counts, weight totals and observed ranges. |
| Quantile calculations | Computing a median or quantile may order temporary values as part of that requested statistic. It never reorders the observation dataset or a prediction index. |
| Bin adjustments | Automatic caps, reductions caused by insufficient distinct boundaries, and explicit-count reductions emit `BinningWarning`. Inspect `requested_n_bins_`, `n_bins_` and `bin_edges_`. |
| Kernel weighting | Kernel weights are calculated from distances. The smoother exposes `weight_scale_` for numerical normalization and `get_weights()` for the resulting influences; its weight snapshot is unchanged. |
| Unsupported queries | The row remains present. `unsupported="nan"` warns and returns NaN; `unsupported="raise"` fails explicitly. |
| Outside-range queries | An explicit `extrapolation` policy applies and emits a warning. Original query coordinates are never clipped or overwritten. |
| CV or bootstrap | These operations only run when explicitly requested. They create subsets or resamples, leave the original arrays unchanged, and preserve paired feature, target and weight alignment. |

`data_summary_` records received and retained row counts, original dtypes, whether sample weights were supplied, the selected prediction policy and whether CV was used. Kernel summaries also record the fitted computation path and numerical weight scale. An invalid refit disables prediction access rather than continuing to use an earlier fitted model.

## Aggregation: simple names or any scalar function

A callable remains the flexible extension point, but users do not need to write a function for common statistics. Choose a name for the simplest and usually fastest path:

```python
Regressogram(agg="mean")
Regressogram(agg="median")
Regressogram(agg="sum")
```

Supported names are `mean`, `median`, `min`, `max`, `sum`, `count`, `std`, `var`, `first` and `last`. `std` and `var` use sample definitions with one degree of freedom. `first` and `last` refer to original relative row order within a bin, so they are intentionally order-sensitive.

There are two explicit custom-function styles. A bare callable receives a Polars expression and returns a scalar aggregation expression. An `array_aggregation` adapter receives a NumPy copy of the bin's values and returns one real numeric scalar. Keeping these styles explicit avoids guessing the function's signature or calling it on a different data type than the author intended.

```python
from rgram import array_aggregation, quantile

# Native Polars expressions stay in the fast aggregation engine.
robust = Regressogram(agg=lambda values: values.median())

# Ordinary NumPy/Python functions run once per bin.
custom = Regressogram(agg=array_aggregation(np.median))
custom_quantile = Regressogram(
    agg=array_aggregation(lambda values: np.quantile(values, 0.9)),
)

# A native, cloneable quantile helper avoids the Python callback overhead.
ninetieth_percentile = Regressogram(agg=quantile(0.9, interpolation="linear"))
```

Numeric callbacks receive copies in original relative order, including zero-weight rows, and can define their own statistic without mutating the caller's data or fitted snapshots. Their errors propagate as Python errors. They are generally slower than native expressions. An estimator needs one numeric prediction per bin, so arrays, dictionaries, strings and complex return values are rejected. NaN results remain unsupported estimates, rather than causing row removal. Use named top-level functions when standard pickle serialization is required; lambdas are not generally pickleable.

### Explicit observation weights

Both estimators accept `sample_weight` in `fit` and `fit_predict`. Weights must be finite and nonnegative, have the same number of rows, and contain positive total mass. Zero-weight rows remain in the snapshots and counts.

For the regressogram, `agg="mean"` computes `sum(weight * y) / sum(weight)` and `agg="sum"` computes `sum(weight * y)`. Other named reductions reject weights because weighted medians, variances and counts have multiple possible conventions. Bare one-argument callables also reject supplied weights rather than silently ignoring them. Define the convention explicitly with either adapter:

```python
from rgram import weighted_aggregation

weights = np.array([1., 2., 0., 3., 1.])
weighted_mean = Regressogram(n_bins=2, agg="mean").fit(x, y, sample_weight=weights)

numpy_weighted = Regressogram(
    n_bins=2,
    agg=array_aggregation(
        lambda values, weights: np.average(values, weights=weights),
        weighted=True,
    ),
).fit(x, y, sample_weight=weights)

polars_weighted = Regressogram(
    n_bins=2,
    agg=weighted_aggregation(lambda values, weights: (values * weights).sum() / weights.sum()),
).fit(x, y, sample_weight=weights)
```

A weighted adapter used without `sample_weight` receives unit weights. An all-zero-weight bin has no weighted mean and follows the unsupported policy. Its weighted sum is zero by definition. Bin selection rules and automatic kernel bandwidth rules use all feature rows without observation weighting; weighted feature quantiles are not implicitly substituted.

## Choosing bins

`binning="dist"` uses equal-frequency quantile cells. `binning="width"` uses equal-width cells. `binning="int"` groups by truncating the feature toward zero, and `binning="none"` groups identical feature values. Those are explicit grouping choices; none modifies the stored feature values.

| Setting | Meaning |
|---|---|
| `n_bins=None` or `"auto"` | A fast rule, with no CV. Quantile binning uses a bounded cube-root count; width binning combines FD and Sturges. |
| `n_bins=12` | An explicit count for either `dist` or `width`. |
| `n_bins="fd"`, `"scott"`, `"sturges"`, `"rice"` or `"sqrt"` | A named distributional count rule, with no CV. |
| `binning="width", bin_width=0.5` | A width in feature units; requires `n_bins=None`. The final cell can be shorter. |
| `n_bins="cv"` | Explicitly request regression-error-based bin-count selection. |

`max_bins=512` bounds automatic allocation and rejects larger explicit counts unless the limit is increased. `min_samples_bin=5` bounds the automatic quantile count and generated CV candidate grid; it is a target occupancy, not a guarantee for every cell. Duplicate boundaries can collapse cells, and constant features use one cell; such changes are reported. No observations are removed to force a desired number of cells.

Inspect `n_bins_`, `bin_edges_`, `bin_width_` and `bins_`. The `bin_left` and `bin_right` columns describe nominal cell bounds, while `x_min_observed` and `x_max_observed` describe the observations actually present. Quantile cells are right-closed; width cells are left-closed, with the training maximum included in the last cell.

The automatic width rule follows the FD/Sturges combination described by [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html), with a Sturges fallback for zero IQR. Histogram rules describe the distribution of `X`; they do not optimize predictions of `y`.

## Is CV necessary?

No. Cross-validation, or CV, is an optional way to choose parameters. It repeatedly fits candidate models on part of the observations and measures error on the held-out observations. It then selects a candidate and refits it on all training rows. This costs additional computation and requires a split scheme appropriate to the data.

Use a fixed count, automatic bin rule, manual bandwidth, or bandwidth rule if that meets your needs. `Regressogram()` does not run CV, even though its dormant `cv` parameter defaults to five folds. CV runs only when you select `n_bins="cv"` or explicitly use `CoverageSearchCV` or an external search object.

```python
from rgram import CoverageSearchCV

# No CV: fit directly to all observations.
plain = Regressogram(n_bins=2).fit(x, y)

# Explicit CV: generate candidates and evaluate held-out regression error.
tuned = Regressogram(n_bins="cv", cv=2, bin_candidates=[1, 2]).fit(x, y)

# Explicit search over smoothing parameters.
search = CoverageSearchCV(
    KernelSmoother(kernel="gaussian", bandwidth="manual"),
    {"bandwidth_value": [0.5, 1.0, 2.0]},
    cv=2,
).fit(x, y)
```

`CoverageSearchCV` reports error and the proportion of validation rows with finite predictions. By default, every fold must have full coverage, so a narrow kernel cannot win by failing to predict difficult rows. Inspect `cv_results_`, `cv_splits_`, `best_params_` and `best_estimator_`. Model error and support policies are respected; the search does not secretly change `unsupported="raise"` into a permissive policy. Select `unsupported="nan"` yourself if missing predictions should be recorded and scored through the coverage constraint.

Integer CV uses shuffled fold membership with a reproducible `random_state`; it does not shuffle stored training rows or the final fitted dataset. Use `TimeSeriesSplit` for appropriate ordered problems, or `GroupKFold` with `groups` supplied to `CoverageSearchCV.fit` for grouped observations. Lowering `min_coverage` explicitly allows error to be computed on supported rows only, which can bias comparisons. Use independent test data or nested CV to assess a tuned model's performance.

## Kernels and local-linear fitting

The built-in kernels are Epanechnikov, Gaussian, uniform, triangular, cosine, logistic, biweight and tricube. Custom kernels accept a Polars expression of normalized distances and return finite nonnegative weights. There is no universally best kernel; tune bandwidth separately for each shape rather than assuming the same numerical bandwidth gives the same effective neighborhood.

`regression="local_constant"` is the default weighted mean. `regression="local_linear"` fits a weighted line around each query and can reduce boundary bias, as described in [Statsmodels' kernel-regression documentation](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html). Local-linear prediction coefficients can be negative and predictions can exceed the observed response range. A singular local fit warns and falls back to a constant by default; choose `singular="raise"` to reject it.

```python
linear = KernelSmoother(
    kernel="tricube", regression="local_linear",
    bandwidth="manual", bandwidth_value=1.5,
).fit(x, y)

print(linear.bandwidth_)
print(linear.algorithm_)
print(linear.predict_diagnostics([0.5, 2.5]))
print(linear.get_weights([0.5], kind="prediction"))
print(linear.get_weights([0.5], kind="kernel"))
```

`get_weights(kind="prediction")` exposes coefficients whose product with the training responses reconstructs supported predictions. `kind="kernel"` exposes positive normalized kernel weights. `effective_n` describes the concentration of positive kernel weights, not an uncertainty estimate. The dense inspection matrix costs `n_query * n_train` storage, so use a small query selection.

### Performance without hidden sorting

`algorithm="auto"` uses neighbor windows only when training `X` is already nondecreasing and the kernel has compact support. Otherwise it uses bounded brute-force blocks. `algorithm="neighbors"` requires eligible input and raises instead of sorting it. `algorithm="brute"` always uses bounded blocks. Duplicate sorted feature values remain separate observations.

Gaussian, logistic and custom kernels use bounded blocks without tail truncation. Compact-window prediction can be much faster with small neighborhoods, but unsorted input deliberately keeps the brute-force path. `algorithm_` and `data_summary_` expose the fitted path. Run `uv run python examples/benchmark_neighbors.py` for a reproducible comparison using already-sorted generated data.

## Warning settings belong to Python, not the estimator

Both estimators warn on unsorted training or query features by default. Unsorted input is valid; the warning makes its order visible, especially before drawing connected line plots. Rgram does not sort it after warning. If you choose to sort outside the library, apply the same permutation to `x`, `y` and observation weights, and keep any time/group interpretation in mind.

Warnings have no `warn` or `warn_unsorted` constructor or mutable class parameters. Use Python's standard [warning filters](https://docs.python.org/3/library/warnings.html), which support a local context, category-specific suppression or treating warnings as errors:

```python
import warnings
from rgram import RgramWarning, UnsortedInputWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UnsortedInputWarning)
    predictions = model.predict([2.5, 0.5])

# To silence all rgram advisory warnings within a block:
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RgramWarning)
    predictions = model.predict([2.5, 0.5])
```

Other categories include `SupportWarning`, `ExtrapolationWarning`, `NumericalWarning`, `BinningWarning` and `DataHandlingWarning`. Filtering warnings does not disable validation, restore unsupported predictions or bypass explicit error policies. Python normally reports a warning once per source location; use `warnings.simplefilter("always", UnsortedInputWarning)` when every occurrence matters.

Parameters that change the model or its predictions, such as `agg`, `n_bins`, `bandwidth`, `extrapolation` and `unsupported`, remain constructor parameters so cloning and `set_params` preserve them. They should not become mutable class-wide settings. This follows the distinction between estimator configuration and external warning presentation in the [scikit-learn estimator conventions](https://scikit-learn.org/stable/developers/develop.html).

## Support, extrapolation and uncertainty

For regressograms, `extrapolation="clip"` retains the existing edge-bin prediction behavior and now warns explicitly on outside-range queries. It maps a prediction to an edge cell without overwriting the query value. For kernel smoothers, `extrapolation="allow"` evaluates at the actual outside-range query and warns; positive kernel support is still required. Both support `extrapolation="nan"` and `"raise"`. `binning="none"` cannot invent an estimate for an unseen feature value. Set `unsupported="raise"` to reject empty cells, zero-mass weighted means or unsupported kernel queries.

No regressogram intervals are computed by default: `ci=None`. If you want descriptive quantile endpoints, request `ci=(quantile(0.1), quantile(0.9))`. Mean ± standard deviation can still be supplied through explicit Polars callables, but it describes spread and is not a confidence interval for the mean. The legacy `return_ci=True` argument returns these configured summaries, or `(prediction, None, None)` if none are configured.

Both estimators provide `predict_interval()` for pointwise paired-bootstrap confidence intervals of the fitted regression curve. These are not prediction intervals for future observations or simultaneous confidence bands. They condition on the selected smoothing parameters and do not correct smoothing bias or include tuning uncertainty.

```python
interval = smoothed.predict_interval(
    [0.5, 2.5], confidence_level=0.95,
    n_resamples=500, random_state=42,
)
print(interval)
```

IID resampling assumes independent observations. Whole-group resampling and moving-block resampling are explicit alternatives for suitable dependent data. Resampling uses copies and preserves feature/target/weight pairing. By default, every replicate must support a query or its bounds remain NaN; `n_valid` and `bootstrap_coverage` make that visible. See the [advanced guide](docs/advanced.md) for assumptions, options and memory costs. User-selected strict error policies also apply during resampling.

## Diagnostics and interactive exploration

`regression_diagnostics(x, y)` returns observations, predictions, residuals and support in original row order. `predict_diagnostics(x)` exposes support and training information for each query. Regressogram `bins_` additionally reports row counts, positive-weight counts, weight totals, observed feature ranges and nominal cell bounds.

```python
from rgram import plot_diagnostics

print(binned.regression_diagnostics(x, y))
figure, axes = plot_diagnostics(binned, x, y)
```

Diagnostic plots use scatter points and do not sort or connect observation rows. Unsupported observations remain in the diagnostic table; the figure reports how many cannot be drawn because their predictions are non-finite.

Run `uv run python examples/kernel_explorer.py` for kernel, bandwidth, query, local-regression and binning controls, plus an explicitly requested bootstrap interval button. The demo generates ordered synthetic features; the library does not sort them. Explorer callbacks are tested headlessly; a desktop backend is required for interactive use.

## Compatibility and current limits

The latest working-tree changes remove `warn` and `warn_unsorted` constructor parameters, make unsorted warnings the default, remove automatic sorted-index construction, change `agg`'s default to the equivalent named `"mean"`, and change the default `ci` to `None`. They add numeric aggregation adapters, weighted regressograms, feature-name checks and explicit support/extrapolation inspection. Migrate warning suppression to Python filters and request descriptive intervals explicitly if desired.

CV remains optional. Sparse or multivariate regression, weighted built-in quantiles/variances, automatic imputation, full sklearn metadata routing, simultaneous confidence bands and prediction intervals are not implemented. Custom weighted conventions can be supplied through adapters. The library tests selected sklearn contracts and integrations but does not claim every generic estimator check passes its intentionally univariate API.

## Development

```bash
uv run pytest -q
uv run ruff check src/rgram examples tests/test_advanced_estimators.py tests/test_estimator_contract.py tests/test_aggregation_and_transparency.py tests/test_properties.py
uv build
```

Tests include independent numerical references, custom callback failures, cloning and serialization, CV coverage, bootstrap calculations, feature names, and property-based checks for row conservation and prediction permutation. CI tests Python 3.9 and 3.12. Performance examples report workload-specific results rather than universal speed claims.

## License

Rgram is licensed under the [MIT License](LICENSE).
