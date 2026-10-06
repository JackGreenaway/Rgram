# Using Rgram with scikit-learn

Rgram is primarily an exploratory tool for relationships between **one numeric
feature and one numeric response** at a time. Its single-feature scope is
intentional: each fitted curve has a clear feature axis and response summary. Use a `Regressogram` for an interpretable, piecewise constant summary
of that relationship, or a `KernelSmoother` for a smooth local estimate without
choosing a global polynomial or linear form. These are useful for inspecting
nonlinearity, comparing subpopulations, and plotting response curves. The same
estimators can also serve as univariate predictive baselines. A regressogram is easy to audit through bin counts
and estimates; a smoother can describe gradual effects that a stepped curve
misses. They require `y`; they do not estimate a feature's
probability density. They also do not adjust for other features or establish
causality. Compare held-out error with other regressors when prediction matters.

## Familiar estimator operations

```python
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from rgram import KernelSmoother, Regressogram

rng = np.random.default_rng(42)
X = rng.uniform(0, 6, size=(120, 1))
y = np.sin(X[:, 0]) + rng.normal(0, 0.1, size=120)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = KernelSmoother(kernel="gaussian")
model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)  # ndarray of shape (n_samples,)
print(mean_squared_error(y_test, y_pred))
print(model.score(X_test, y_test))  # R²: higher is better, can be negative
fresh_model = clone(model)         # copies parameters, does not copy fitted state
fresh_model.set_params(bandwidth_adjust=0.5).fit(X_train, y_train)
```

`fit` returns the estimator. `fit_predict(X, y)` predicts at the original
training rows; its in-sample accuracy is not a held-out evaluation. Parameters
come from `get_params()` and fitted attributes have a trailing underscore.
Refit after changing any parameter that determines the learned model.
`predict` and diagnostics raise `sklearn.exceptions.NotFittedError` before a
successful fit, including after a failed refit.

Use `X=` consistently for fitting, prediction, and diagnostics.
`predict(X)` always returns a one-dimensional array.
For column-name exploration, use `fit(X="temperature", y="demand", data=frame)`
with a pandas DataFrame, Polars DataFrame, or Polars LazyFrame. Pass `data` and `sample_weight` by keyword.

## Prediction pipelines and tuning

The estimators occupy the **final regressor step** of a pipeline. They have no
`transform` or `fit_transform`; they are not intermediate preprocessing steps.
Keep all learned preprocessing inside the pipeline during cross-validation.

```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("impute", SimpleImputer()),
    ("scale", StandardScaler()),
    ("model", KernelSmoother(kernel="gaussian")),
])
cv = KFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    {"model__bandwidth_adjust": [0.5, 1.0, 2.0]},
    scoring="neg_mean_squared_error", cv=cv, error_score="raise",
)
search.fit(X_train, y_train)
print(search.best_params_)
print(search.predict(X_test))
print(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2"))

# With sklearn's default metadata routing configuration:
pipeline.fit(X_train, y_train, model__sample_weight=np.ones(len(y_train)))
```

Nested parameter names follow the usual `step__parameter` convention. A manual
bandwidth is measured in the units **after preprocessing**, so scaling changes
its interpretation. `sample_weight` affects model fitting; a pipeline does not
automatically send those weights to the scaler or validation scorer. Full
metadata routing is not supported; use the default routing configuration and
explicit step-prefixed fit arguments.

For a wider input table, select exactly one feature before the regressor:

```python
from sklearn.compose import ColumnTransformer

import polars as pl

frame = pl.DataFrame({
    "temperature": X[:, 0], "humidity": np.ones(len(X)), "demand": y,
})
select = ColumnTransformer([
    ("temperature", StandardScaler(), ["temperature"]),
], remainder="drop", sparse_threshold=0)
weather_model = Pipeline([
    ("select", select),
    ("model", Regressogram(n_bins=5)),
])
weather_model.fit(frame.select(["temperature", "humidity"]), frame["demand"])
print(weather_model.predict(frame.select(["temperature", "humidity"])))
```

The selection must retain a two-dimensional, one-column feature matrix. A pandas
DataFrame, Polars DataFrame, NumPy array, list, or Series containing numeric
values can be used directly. Pandas also supports named `data=` selection; see
[the pandas quick start](getting_started.md#using-pandas). Index labels are ignored,
so separately supplied features, targets, and weights must already have matching
row order. Named one-column inputs record `feature_names_in_`;
a different prediction column name raises. Switching between named and unnamed
inputs emits a standard `UserWarning`, as with scikit-learn.

## Exploratory analysis

The primary workflow is to fit on all observations for a descriptive view, then
inspect the curve, support, and residuals. For a wider table, fit each feature
against the response separately; for subgroup comparisons, select each group
explicitly and fit a separate curve. These pairwise summaries do not adjust for
other features or capture interactions. Change the bin count or bandwidth to
check how much the apparent pattern depends on smoothing. This answers a different question from estimating generalization
error. Shuffle is normal for prediction; row order and duplicates are preserved.
Rgram emits `UnsortedInputWarning` for unordered inputs. Suppress that advisory
with Python's warning filters if order is irrelevant to your workflow.

```python
import warnings
from rgram import UnsortedInputWarning, plot_diagnostics

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UnsortedInputWarning)
    descriptive = Regressogram(n_bins=6).fit(X, y)
    residuals = descriptive.regression_diagnostics(X, y)
    print(descriptive.bins_)  # occupied cells, estimates, counts, boundaries
    print(residuals)
    curve = descriptive.predict_diagnostics(np.linspace(0, 6, 100))
    figure, axes = plot_diagnostics(descriptive, X, y)  # requires rgram[plot]

smooth = KernelSmoother(kernel="gaussian").fit(X, y)
print(smooth.predict_grid())  # evenly spaced locations, support diagnostics
print(smooth.bandwidth_)
```

Diagnostic tables are Polars DataFrames. When using a fitted pipeline, Rgram's
custom diagnostics belong to its final estimator and expect **transformed**
features, for example `pipeline.named_steps["model"].regression_diagnostics(
pipeline[:-1].transform(X_train), y_train)`. Use `pipeline.predict(X)` with
original features for ordinary prediction. Rgram's standalone plotting helper
also expects the final estimator and its feature coordinates.

Scikit-learn's `partial_dependence` also works with the fitted estimators and
pipelines using its brute method. With one input feature it evaluates the fitted
response curve; it does not reveal an effect adjusted for other variables.

## Behavior to account for

| Topic | Contract |
|---|---|
| Shapes | One feature and one target; `X` can be `(n,)` or `(n, 1)`, with `(n, 1)` preferred in sklearn workflows. One feature at a time is intentional for pairwise exploration; multiple features/targets and sparse input are unsupported. |
| Missing values | Null, NaN, infinity, and complex values are rejected. Use an explicit imputer for missing features and clean targets before fitting. Numeric strings are not automatically converted. |
| Weights | Finite, nonnegative, with positive total mass. Zero-weight rows still participate in bin-boundary and automatic bandwidth calculations. Built-in weighted aggregations have restrictions; see the parameter reference. |
| Unsupported locations | Compact kernels and empty regressogram bins can return NaN with `SupportWarning`. Ordinary sklearn scorers reject non-finite predictions. Use `unsupported="raise"` to fail promptly, increase bandwidth/reduce bins, or inspect coverage with `CoverageSearchCV`. Never silently discard validation rows. |
| Extrapolation | Regressograms default to edge-cell clipping; smoothers evaluate the actual query. Both warn outside the training range and offer `extrapolation="raise"` or `"nan"`. Gaussian kernels usually avoid gaps in support, but do not guarantee useful extrapolation. |
| Automatic tuning | Default bin/bandwidth rules use training features, not held-out prediction error. CV runs only when requested; `n_bins="cv"` explicitly enables internal CV. |
| Uncertainty | `predict` returns a 1D array by default. `predict_interval` returns pointwise bootstrap confidence intervals for the curve, not future-observation prediction intervals. Configured regressogram descriptive endpoints are available in `predict_diagnostics(X)` as `summary_lower` and `summary_upper`. |
| Score | Inherited `score` is R² and accepts scoring weights. `CoverageSearchCV.best_score_` is negative pooled validation MSE; its `score` is R². |

`CoverageSearchCV` is an optional Rgram helper, not a drop-in `GridSearchCV`:
it uses pooled, unweighted validation MSE, integer CV means **shuffled** KFold,
it refits the winner, propagates fit errors, and requires finite predictions in
every fold by default. Its result keys differ from sklearn's. Use standard
`GridSearchCV` for normal pipelines and scoring options; see the
[advanced guide](advanced.md) for coverage-aware selection.

The package verifies these specific sklearn workflows rather than claiming all
generic estimator checks pass: many checks assume multivariate input. See the
[parameter reference](statistical_parameters.md) for choices and statistical
assumptions, and [sklearn's estimator conventions](https://scikit-learn.org/stable/developers/develop.html)
for the underlying API.

(metadata_routing)=
## Metadata routing compatibility

Inherited scikit-learn metadata-request methods are available and documented in
the API reference. Full metadata routing is not implemented throughout Rgram.
Use explicit `sample_weight=` for a direct fit and step-prefixed fitting weights
with sklearn's default routing configuration for a pipeline. `CoverageSearchCV`
does not forward weights through nested pipeline steps. See
[scikit-learn's metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html)
for the upstream request mechanism.

(r2_score)=
## R² scoring

The inherited `score(X, y, sample_weight=None)` evaluates the coefficient of
determination, R², using the estimator's predictions. With equal observation
weights, it is

$$
R^2 = 1 - \frac{\sum_i (y_i - \widehat y_i)^2}
                 {\sum_i (y_i - \overline y)^2}.
$$

The best possible score is 1; predicting the observed mean gives 0 for a
nonconstant response, and scores can be negative. Scoring weights apply to both
sums and to the observed mean. They do not refit the model. R² is undefined with
fewer than two evaluation observations. Scikit-learn's default finite-score
handling maps constant-response cases to 1 for perfect predictions and 0 otherwise.
See [the scikit-learn R² reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
for details. This score is distinct from `CoverageSearchCV.best_score_`, which is
negative pooled validation MSE.

## Glossary

```{eval-rst}
.. glossary::

   meta-estimator
      An estimator that wraps or combines other estimators. Examples include
      scikit-learn's Pipeline and GridSearchCV, and Rgram's CoverageSearchCV.
      Wrapping a model does not automatically route every fitting argument;
      use the explicit weight forwarding described above.
```
