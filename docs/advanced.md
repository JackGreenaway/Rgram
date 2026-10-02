# Advanced estimator contracts

The [README](../README.md) contains complete starter examples, aggregation choices, warning filters and an explanation of optional cross-validation. The [statistical parameter reference](statistical_parameters.md) explains each setting, its default and its effect on estimates. This guide describes the numerical and statistical details. All ordinary fits preserve observation order. Neither estimator sorts a dataset or builds a sorted training index.

## Aggregation execution and weights

Named aggregations and bare Polars-expression callables run in the native aggregation engine. `array_aggregation(func)` runs a numeric function once per occupied bin in Python. Its arrays are independent copies in original relative row order. Running numeric callbacks outside a Rust UDF boundary keeps user exceptions as normal Python exceptions, including errors about non-scalar results.

`array_aggregation(func, weighted=True)` calls `func(values, weights)`. `weighted_aggregation(func)` calls a native expression function with the corresponding value and weight expressions. A weighted adapter without supplied weights receives ones. No adapter silently guesses whether a second function argument means a weight, an axis or something else. For example, use `lambda values, weights: np.average(values, weights=weights)` rather than passing `np.average` directly as a two-positional-argument weighted function.

Named weighted means and sums use the original weights. Other weighted statistical conventions must be defined by the caller, including weighted quantiles, sample variances and effective counts. Zero-weight rows remain available to custom callbacks and in `X_`, `y_`, `sample_weight_` and `n_samples`. The `n_positive_weight` and `weight_sum` summary columns distinguish them from contributing positive weights. For an all-zero-weight bin, a weighted mean is undefined and follows `unsupported`; a weighted sum is zero.

Every aggregation and configured interval endpoint must return one real numeric scalar per bin. NaN is an explicit unsupported result, not a request to remove a bin or prediction row. Infinite results raise. Functions that deliberately filter, sort or otherwise transform their received values are client-authored statistics; the library does not add such preprocessing around them.

The default `agg="mean"` is equivalent to the previous default mean callable. `ci=None` is the default: no descriptive endpoints are computed unless supplied. A tuple such as `ci=(quantile(0.1), quantile(0.9))` describes response quantiles within a bin. It is not a confidence interval for the regression mean. Standard pickle supports named reductions, quantile helpers and adapters around top-level functions; arbitrary lambdas need a serializer that supports them.

## Observation preservation and numerical buffers

Selected Polars columns are materialized once. Array and frame inputs are validated and copied into fitted snapshots, preserving their values, inferred numeric dtypes and original order. The computational representation uses float64 where regression arithmetic requires it. Conversions that would lose integer precision or wider-floating precision raise rather than silently round values. A prediction always corresponds to the query at the same position.

The smoother keeps original observation weights in `sample_weight_` and exposes `weight_scale_` for the common normalization used during numerical calculations. That scaling cancels in normalized kernel weights; it is not a change to the stored data. A positive observation weight that would underflow to zero during this scaling is rejected. Kernel tails can still become numerically zero when distances are extremely large; `n_neighbors` and `get_weights` describe the numerical weights actually used.

`data_summary_` reports retained rows, original dtypes, the float64 computation buffer, whether weights were supplied and the selected computation or binning policies. Snapshots are independent from caller arrays. The audit reports no dropped, sorted or imputed observations. Fitted attributes can still be deliberately modified by a caller, as with other Python estimators; do not mutate them if you want reproducible fitted behavior.

Feature names are recorded when available. Named prediction inputs are checked against `feature_names_in_`; unnamed one-feature arrays remain valid. Extra columns and multivariate arrays are rejected, not reduced to the first column. A refit with unnamed inputs clears the old feature name.

## Bin rules and boundaries

For `binning="dist"`, automatic selection uses `ceil(n**(1/3))` with limits from `min_samples_bin`, the number of observations and `max_bins`. Equal-frequency cells already adapt their widths to feature density, so their default count does not depend on the full feature range divided by IQR. This avoids a large count driven solely by a distant outlier. It is a heuristic, not regression-error optimization.

For `binning="width"`, automatic selection takes the larger FD or Sturges count. FD and Scott compute a width and convert it to a count using ceiling. A zero-IQR FD calculation falls back to Sturges for nonconstant features. Constant features use one cell. The formulas follow the [NumPy histogram rule definitions](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html), except that rgram's explicit zero-IQR fallback and continuous-feature treatment are documented choices.

Explicit `n_bins` applies to both width and quantile binning. For quantile bins, an explicit count greater than the number of observations is reduced with a warning; repeated boundaries also warn when they reduce the number of cells. Explicit counts beyond `max_bins` raise rather than silently exceed the allocation guard. A manual `bin_width` that requires too many cells also raises. Automatic caps are reported through `BinningWarning`. These operations change the fitted partition, not the observations.

`bin_edges_` contains interior boundaries. Quantile cells are right-closed; width cells are left-closed with the training maximum included in the final cell. Empty interior cells remain possible. `bins_` lists occupied cells in first-observed order, not sorted order. `bin_left` and `bin_right` describe nominal cell bounds for width/dist, while `x_min_observed` and `x_max_observed` describe the actual rows. The count and weight columns allow row and mass conservation checks.

`binning="int"` explicitly truncates feature values toward zero to form group identifiers; it does not replace the stored feature values. `binning="none"` groups equal values and cannot supply an estimate for an unseen value. Sorting temporary boundaries, unique-value tables or values inside a requested order statistic is mathematical computation, not an automatic row permutation or a sorted training index.

## Kernel evaluation without automatic sorting

The built-in compact kernels are Epanechnikov, uniform, triangular, cosine, biweight and tricube. Gaussian and logistic have infinite support. Custom kernels receive a Polars expression of normalized distances and must return finite nonnegative weights.

With `algorithm="auto"`, already-nondecreasing training features and a compact kernel use binary-search support windows. Unsorted features use bounded dense blocks and emit the normal unsorted-input warning. `algorithm="neighbors"` requires eligible input and raises otherwise. The estimator does not sort an index to satisfy that request. Duplicate ordered feature values remain separate training observations, and query order does not need to be sorted for either computation path.

Eligible neighbor evaluation costs approximately `O(m log n + total neighbors)` after the ordinary fit, with no sorting stage. Dense evaluation takes `O(m*n)` time and processes at most `batch_size` query rows at once, also bounded to approximately one million query/training pairs per block where possible. A single very large training vector still requires `O(n)` temporary memory. Infinite-support and custom kernels use the dense path without approximate tail truncation.

`get_weights` intentionally allocates the full dense inspection result. Ordinary predictions do not require that allocation. `kind="kernel"` returns positive normalized kernel/observation weights, while `kind="prediction"` returns the coefficients used to combine the responses. These are identical for a local constant. `effective_n` is `1 / sum(kernel_weights**2)` and describes positive weight concentration, not an uncertainty interval.

Run `uv run python examples/benchmark_neighbors.py` to compare eligible neighbor search and bounded brute force on already-sorted generated observations. Broad support can erase the speed advantage. Previously reported timings from an automatically sorted-index implementation do not describe the present unsorted-input policy.

## Local-linear fits and numerical fallback

`regression="local_linear"` fits a weighted intercept and slope around each query. The implementation centers and scales local offsets, then calculates equivalent prediction coefficients. Independent weighted-least-squares tests verify the calculation. Local-linear fitting can reduce boundary bias, but negative coefficients and predictions outside the observed response range are possible. [Statsmodels documents the distinction between local-constant and local-linear regression](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html).

If the weighted local design is singular or numerically unresolved, `singular="constant"` uses a local-constant fallback and emits `NumericalWarning`. `local_linear_fallback` identifies those queries. Choose `singular="raise"` if the fallback is unacceptable. Warning suppression does not change the fallback policy.

Automatic bandwidth rules use all feature values without observation weighting. Silverman's zero-IQR fallback uses a positive standard deviation and emits `DataHandlingWarning`. Constant or singleton features require a manual positive bandwidth. Refit after changing bandwidth parameters. There is no automatic regression bandwidth search during an ordinary fit.

## Extrapolation and unsupported predictions

Regressograms use `extrapolation="clip"` by default for compatibility: an outside-range query can use an edge cell, with `ExtrapolationWarning`. The query itself is not clipped. Kernel smoothers use `extrapolation="allow"` by default and evaluate the kernel at the actual query, also with a warning. Both offer `"nan"` and `"raise"` policies. `in_training_range` exposes this distinction separately from numerical support.

`unsupported="nan"` preserves a missing-estimate row and warns. `unsupported="raise"` fails explicitly. An empty cell, an all-zero-weight mean, or an unsupported kernel neighborhood cannot silently disappear from the result. Infinite-support kernels can give finite predictions far outside the training range, so finite prediction coverage is not evidence of reliable extrapolation.

## Optional CV, fit weights and inspectable splits

Cross-validation only runs when requested through `n_bins="cv"`, `CoverageSearchCV`, or another explicit search object. It fits candidate models on training subsets and evaluates held-out rows. Ordinary fits use every supplied observation without splitting. Dormant `cv`, `bin_candidates` and `random_state` constructor options do not initiate tuning.

`CoverageSearchCV` uses pooled mean squared error over finite validation predictions and reports the coverage of every fold. Each fold must meet `min_coverage`; ineligible candidates receive infinite selection loss. By default, `min_coverage=1.0` prevents models from benefiting from missing difficult predictions. Lowering it explicitly changes the comparison to supported rows only. All-ineligible searches raise and retain results for inspection.

The search does not override support, extrapolation or warning policies. Candidate fit and prediction errors propagate. Select a permissive missing-prediction policy explicitly if coverage should be measured instead of raising. Supplied sample weights are sliced with the training rows and passed to fitting; scoring remains unweighted MSE. Nested pipeline weight routing is not implemented. Candidate-specific preprocessing can be placed in a sklearn pipeline when desired, making that preprocessing a client choice.

Integer CV uses shuffled KFold membership controlled by `random_state`; actual train/test indices are exposed through `cv_splits_`, and the final model is refit in the original full-data order. Use a supplied temporal splitter for ordered dependence. For groups, provide `GroupKFold` and aligned group labels to `CoverageSearchCV.fit`. The selected model's named feature is retained after refitting. Use an independent test set or nested CV for an unbiased assessment after tuning.

`n_bins="cv"` is a convenience for count selection and exposes `cv_results_`; use the separate search object when you need split indices, grouped fitting, or a broader parameter grid. The search is serial and univariate and does not aim to reproduce every GridSearchCV option.

## Bootstrap intervals and their assumptions

Both estimators provide `predict_interval` with `confidence_level`, `n_resamples`, `random_state`, `method`, `resampling` and `min_valid_fraction`. The result includes original query values, predictions, lower/upper bounds, `n_valid` and `bootstrap_coverage`. These are pointwise confidence intervals for the fitted regression functional, not prediction intervals or simultaneous bands. With mean aggregation they concern the fitted conditional-mean curve; a custom aggregation changes the target statistic.

`resampling="iid"` samples paired feature, target and observation-weight rows and assumes independent identically distributed observations from a random design. `resampling="groups"` samples whole independent groups and preserves within-group order; at least two groups are required, and very few groups do not support reliable nominal coverage. `resampling="blocks"` uses overlapping moving blocks in original training order and requires an explicit block size. Approximate stationarity and enough blocks matter; block sampling does not automatically protect against trends or regime changes.

`method="percentile"` uses empirical bootstrap quantiles. `method="basic"` reflects those quantiles around the fitted estimate. These are the standard definitions described by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). More resamples improve Monte Carlo stability, but do not fix a poor inferential design.

Bandwidth and the selected bin count remain fixed during resampling, while regressogram boundaries are re-estimated on each resample. Hyperparameter selection uncertainty and smoothing bias are not corrected. Nominal coverage is not guaranteed for arbitrary data or smoothing choices. Each fit is a clone; the original estimator's training arrays are unchanged. Draw storage costs `O(n_resamples*n_query)`, so use a modest evaluation grid.

By default, every replicate must support a query before bounds are reported. Otherwise the interval remains NaN and a warning points to `n_valid` and `bootstrap_coverage`. Lowering `min_valid_fraction` explicitly uses quantiles of valid draws, which changes the inferential meaning. Replicates containing only zero observation weights are counted as unsupported rather than silently omitted. Other errors, including strict support or singular-fit policies, propagate.

Bootstrap resampling can create unsorted temporary samples even when the original observations were sorted. Standard warning filters remain active, and the estimator does not sort those resamples. An explicit `algorithm="neighbors"` can therefore reject a resample; use `"auto"` or `"brute"` for bootstrap work that must accommodate arbitrary sample order. A local `warnings.catch_warnings` block can suppress expected `UnsortedInputWarning` messages at the caller's choice.

`KernelSmoother.predict(..., return_ci=True)` is a convenience for IID-bootstrap bounds. For regressograms, legacy `return_ci` returns only the explicitly configured `ci` aggregations, or `None` endpoints by default. Use `predict_interval` for bootstrap inference on either estimator.

## Warnings, diagnostics and documented limits

Warnings use Python categories, not constructor switches or mutable class settings. `UnsortedInputWarning`, `SupportWarning`, `ExtrapolationWarning`, `NumericalWarning`, `BinningWarning` and `DataHandlingWarning` all derive from `RgramWarning`. Standard filters can ignore, display repeatedly or escalate them to exceptions, including inside CV and bootstrap. Validation errors remain active independently of filters.

`regression_diagnostics` returns observed values, predictions, residuals and support without sorting or dropping rows. Optional `plot_diagnostics` uses scatter plots, reports unsupported rows that cannot be drawn, and does not connect observations after secretly sorting them. The explorer uses explicitly generated ordered synthetic data and makes interval calculation an explicit button action.

Tests cover numeric references, row/weight snapshots, custom callback errors, named feature checks, selected sklearn contracts, support-aware selection, bootstrap calculations and property-based row conservation. Some broader features remain deliberately absent: sparse/multivariate estimation, built-in weighted quantile/variance conventions, automatic imputation, full sklearn metadata routing, prediction intervals and simultaneous bands. The [review](review.md) describes concrete next steps without claiming these features are implemented.
