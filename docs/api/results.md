# Fitted attributes and result schemas

Unless noted, tables are Polars DataFrames and queries retain their input order
and duplicates. Null metadata means a bin was absent; NaN predictions/bounds mean
no finite estimate was available under the selected policy. Strict policies can
raise before a table is returned. Never treat numerical support as accuracy.

## Shared fitted state

`Regressogram` and `KernelSmoother` expose `X_`, `y_`, and `sample_weight_` as
independent one-dimensional training snapshots; omitted weights are ones.
`n_samples_in_` counts all rows, `n_features_in_` is one, and
`sample_weight_provided_` records whether weights were supplied.
`feature_names_in_` exists only for named input and is cleared by an unnamed refit.
The estimator class reference lists all model-specific fitted attributes.

`data_summary_` has these keys:

| Key | Meaning |
|---|---|
| `n_rows_received`, `n_rows_retained` | Training row counts; equal after successful validation. |
| `rows_sorted`, `rows_dropped`, `values_imputed` | `False`, `0`, `0` respectively. |
| `x_dtype`, `y_dtype` | Original snapshot dtypes as strings. |
| `computation_dtype` | `"float64"`. |
| `sample_weight_provided` | Whether observation weights were supplied. |
| `weights_modified_in_snapshot` | `False`; computational normalization does not alter stored weights. |
| Regressogram: `binning`, `n_bins_requested`, `n_bins_fitted`, `cv_used`, `extrapolation` | Requested bin strategy/option, fitted count, internal CV flag, and range policy. |
| Smoother: `algorithm`, `weight_scale`, `bandwidth`, `extrapolation` | Fit-time computation path, numerical weight scale, effective fitted bandwidth, and range policy. |

`requested_n_bins_` is available for width/dist binning only and is the nominal
count after rule/cap/sample/constant-feature adjustment, before boundary merging.
`selected_n_bins_` stores the internal CV winner or the original `n_bins` setting.
For integer/exact grouping, `n_bins_` counts unique groups; `bin_edges_` is empty
and `bin_width_` is None. `bins_` contains occupied groups, so its height need
not equal the nominal cell count.

## Regressogram bins

| Column in `bins_` | Meaning |
|---|---|
| `rgram_bin` | Integer cell index for dist/width, truncated integer for int, feature value for none. |
| `y_pred_rgram` | Configured response aggregation. |
| `y_pred_rgram_lci`, `y_pred_rgram_uci` | Present only when descriptive `ci` reductions are configured. |
| `n_samples` | All observations in the cell, including zero-weight rows. |
| `weight_sum` | Sum of supplied weights, or row count with omitted weights. |
| `n_positive_weight` | Number of rows with positive weight. |
| `x_min_observed`, `x_max_observed` | Actual observed feature range in the cell. |
| `bin_left`, `bin_right` | Nominal cell limits for dist/width only; equality conventions are in the parameter reference. |

Rows follow first-observed cell order, not sorted boundary order.

## Prediction diagnostics

Regressogram `predict_diagnostics` returns:

| Column | Meaning |
|---|---|
| `x` | Original query coordinate. |
| `prediction` | Bin aggregation at the query. |
| `rgram_bin` | Assigned group/cell, including edge assignment where applicable. |
| `n_samples`, `weight_sum`, `n_positive_weight` | Information from the assigned occupied bin, or null if absent. |
| `supported` | Whether the prediction is finite. |
| `in_training_range` | Whether the original query lies between observed feature extrema, inclusively. |

Kernel `predict_diagnostics` and `predict_grid` return:

| Column | Meaning |
|---|---|
| `x`, `prediction`, `supported`, `in_training_range` | Same meanings as above. |
| `n_neighbors` | Count of positive numerical kernel/observation weights; zero for unsupported queries. |
| `effective_n` | Inverse sum of squared normalized nonnegative weights; zero for unsupported queries. A concentration measure, not degrees of freedom. |
| `local_linear_fallback` | Whether this query used a local constant because its slope was unresolved. |

## Residual diagnostics

`regression_diagnostics(x, y)` returns `x`, `observed`, `prediction`, `residual`,
and `supported`. Residual is observed minus predicted, with NaN when unsupported.
It accepts training or held-out aligned pairs. No row is removed because its
prediction is unsupported.

## Bootstrap intervals

`predict_interval` returns:

| Column | Meaning |
|---|---|
| `x` | Original query. |
| `prediction` | Original fitted curve estimate. |
| `lower`, `upper` | Pointwise bootstrap bounds; NaN without adequate draw support. |
| `n_valid` | Number of finite bootstrap predictions for this query. |
| `bootstrap_coverage` | `n_valid / n_resamples`, not nominal confidence coverage. |

Intervals require at least two valid draws and the configured valid fraction.
These bounds are distinct from regressogram descriptive `ci` columns.

## Search results

`CoverageSearchCV.cv_results_` contains one entry per candidate:

| Key | Meaning |
|---|---|
| `params` | List of candidate parameter dictionaries. |
| `mean_test_mse` | Pooled squared error divided by the number of finite validation predictions, not an unweighted average of fold means. Infinity if none are finite. |
| `coverage` | Pooled supported validation fraction. |
| `min_fold_coverage` | Lowest supported fraction among folds. |
| `eligible` | Every fold meets `min_coverage` and pooled MSE is finite. |
| `selection_loss` | MSE for eligible candidates, infinity otherwise. |
| `n_validation`, `n_supported` | Total validation occurrences and supported occurrences; repeated validation membership counts repeatedly. |
| `split{k}_coverage` | Supported fraction in fold k. |
| `split{k}_test_mse` | MSE over supported rows in fold k; infinity when none are supported. |

All entries except `params` are NumPy arrays. `cv_splits_` stores copied train/test
indices, and `n_splits_` stores the fold count. `best_index_`, `best_params_`,
`best_score_` (negative selection loss), and `best_estimator_` describe the refitted
winner. An all-ineligible fit keeps results for inspection but does not produce
a fitted predictor. Internal `Regressogram(n_bins="cv")` exposes the same results
dictionary and its selected bin count, but not a complete search object.

## Compatibility state

`Regressogram.over_cols` and `KernelSmoother._bw_value` are legacy implementation
state. They are not additional tuning parameters or supported result schemas;
use the fitted attributes above. Private computational buffers and helper methods
are covered by [implementation responsibilities](../internals.md).
