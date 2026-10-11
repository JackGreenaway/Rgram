# CoverageSearchCV

This optional search object is deliberately narrower than `GridSearchCV`.
It uses pooled unweighted validation MSE and rejects candidates with inadequate
support in any fold. See [search result fields](results.md#search-results) and
[the search contracts](../advanced.md#optional-cv-fit-weights-and-inspectable-splits).

## What it solves

A candidate smoother can look accurate on the rows where it predicts while
leaving other validation rows unsupported. `CoverageSearchCV` makes the required
fraction of finite predictions explicit when comparing parameter settings.

## When to use it

Use it to select a single-feature regressor by pooled validation MSE while
requiring sufficient prediction coverage in every fold. The default requires
full coverage. Use ordinary `GridSearchCV` when you need its broader scoring and
search options; this helper has its own [search contracts](../advanced.md).

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

<!-- BEGIN REFERENCE:rgram.CoverageSearchCV -->
<span id="rgram.CoverageSearchCV"></span>

## rgram.CoverageSearchCV

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/model_selection.py#L18">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">rgram.</span><span class="api-name">CoverageSearchCV</span>(
    estimator,
    param_grid,
    *,
    cv=5,
    min_coverage=1.0,
    random_state=0,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Select parameters by validation MSE with explicit support requirements.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">estimator</code> <span class="api-field-type">sklearn estimator</span></dt><dd><p>Cloneable single-feature regressor with fit and predict. Fit errors and prediction policies are respected, not overridden.</p></dd>
<dt><code class="api-field-name">param_grid</code> <span class="api-field-type">dict or list of dicts</span></dt><dd><p>Parameter names mapped to sequences of candidates, as in ParameterGrid. Nested names use the usual step__parameter convention.</p></dd>
<dt><code class="api-field-name">cv</code> <span class="api-field-type">int, splitter or iterable, default=5</span></dt><dd><p>Integer uses shuffled KFold. Otherwise supply a split(X, y, groups) object or iterable of nonempty, disjoint train/test index pairs.</p></dd>
<dt><code class="api-field-name">min_coverage</code> <span class="api-field-type">float, default=1.0</span></dt><dd><p>Minimum fraction of finite predictions required in every fold, in (0, 1]. Lowering this compares only supported rows and can favor selective models.</p></dd>
<dt><code class="api-field-name">random_state</code> <span class="api-field-type">int or None, default=0</span></dt><dd><p>Seed for integer-CV splitting. Does not reseed a supplied splitter.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">best_estimator_</code> <span class="api-field-type">estimator</span></dt><dd><p>Winning model refitted on all rows. Available only after successful fit.</p></dd>
<dt><code class="api-field-name">best_params_</code> <span class="api-field-type">dict</span></dt><dd><p>Winning parameter combination.</p></dd>
<dt><code class="api-field-name">best_index_</code> <span class="api-field-type">int</span></dt><dd><p>Position of the winner in <code>cv_results_</code>. Ties use the first candidate.</p></dd>
<dt><code class="api-field-name">best_score_</code> <span class="api-field-type">float</span></dt><dd><p>Negative pooled unweighted validation MSE, not R-squared.</p></dd>
<dt><code class="api-field-name">cv_results_</code> <span class="api-field-type">dict</span></dt><dd><p>Candidate parameters, MSE, support fractions, eligibility, selection loss, validation counts, and per-fold MSE/coverage arrays. See result schemas.</p></dd>
<dt><code class="api-field-name">cv_splits_</code> <span class="api-field-type">tuple of (train_indices, test_indices)</span></dt><dd><p>Copies of the actual fold membership indices.</p></dd>
<dt><code class="api-field-name">n_splits_</code> <span class="api-field-type">int</span></dt><dd><p>Number of supplied validation folds.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Always one after successful fit.</p></dd>
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of shape (1,)</span></dt><dd><p>Original feature name, only when the training input is named.</p></dd>
</dl>

### Notes

The selection loss pools squared errors over finite validation predictions.
Each fold must independently meet min_coverage. Ineligible candidates have
infinite selection loss. If none qualifies, fit raises and `cv_results_` remains
inspectable, but the search is not fitted. There is no n_jobs, scoring, refit,
or error_score option. Inherited score reports R-squared.
Weights affect fitting only. Direct weight forwarding to nested pipelines is
not implemented. Unlike GridSearchCV's integer regression CV, integer CV here
is shuffled. Preprocessing must be inside the candidate pipeline to avoid leakage.


### Methods

| Method | Purpose |
| --- | --- |
| [fit](#rgram.CoverageSearchCV.fit) | Evaluate candidates and refit the eligible winner on all observations. |
| [get_metadata_routing](#rgram.CoverageSearchCV.get_metadata_routing) | Get metadata routing of this object. |
| [get_params](#rgram.CoverageSearchCV.get_params) | Get parameters for this estimator. |
| [predict](#rgram.CoverageSearchCV.predict) | Predict through the refitted winning estimator. |
| [score](#rgram.CoverageSearchCV.score) | Return :ref:`coefficient of determination <r2_score>` on test data. |
| [set_fit_request](#rgram.CoverageSearchCV.set_fit_request) | Configure whether metadata should be requested to be passed to the ``fit`` method. |
| [set_params](#rgram.CoverageSearchCV.set_params) | Set the parameters of this estimator. |
| [set_score_request](#rgram.CoverageSearchCV.set_score_request) | Configure whether metadata should be requested to be passed to the ``score`` method. |

<span id="rgram.CoverageSearchCV.fit"></span>

### fit {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/model_selection.py#L97">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(
    X,
    y,
    *,
    data=None,
    groups=None,
    sample_weight=None,
)</code></pre></div>


Evaluate candidates and refit the eligible winner on all observations.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, 1), or str</span></dt><dd><p>Single feature or its column name when data is supplied.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like or str</span></dt><dd><p>Aligned numeric response or its column name. Required.</p></dd>
<dt><code class="api-field-name">data</code> <span class="api-field-type">pandas.DataFrame, polars.DataFrame or polars.LazyFrame, default=None</span></dt><dd><p>Source for named feature and response columns.</p></dd>
<dt><code class="api-field-name">groups</code> <span class="api-field-type">array-like of shape (n_samples,) or None, default=None</span></dt><dd><p>Labels passed to an explicit group-aware splitter. Rejected with integer cv.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">array-like of shape (n_samples,) or None, default=None</span></dt><dd><p>Nonnegative fitting weights, sliced by training fold. Scoring is unweighted.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">CoverageSearchCV</span></dt><dd><p>Fitted search. The winner is refit on every supplied row.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>For invalid input, fold indices, policies, or when no candidate qualifies. Candidate fit/prediction errors propagate rather than becoming scores.</p></dd>
</dl>


</div>

<span id="rgram.CoverageSearchCV.get_metadata_routing"></span>

### get_metadata_routing {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_metadata_requests.py#L1743">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_metadata_routing</span>()</code></pre></div>


Get metadata routing of this object.

Please check [metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
mechanism works.

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">routing</code> <span class="api-field-type">MetadataRequest</span></dt><dd><p>A <code>sklearn.utils.metadata_routing.MetadataRequest</code> encapsulating routing information.</p></dd>
</dl>


</div>

<span id="rgram.CoverageSearchCV.get_params"></span>

### get_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L248">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_params</span>(deep=True)</code></pre></div>


Get parameters for this estimator.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">deep</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>If True, will return the parameters for this estimator and contained subobjects that are estimators.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">params</code> <span class="api-field-type">dict</span></dt><dd><p>Parameter names mapped to their values.</p></dd>
</dl>


</div>

<span id="rgram.CoverageSearchCV.predict"></span>

### predict {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/model_selection.py#L295">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict</span>(X)</code></pre></div>


Predict through the refitted winning estimator.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_queries,) or (n_queries, 1)</span></dt><dd><p>Query feature values, with the fitted feature name if named.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">prediction</code> <span class="api-field-type">ndarray of shape (n_queries,)</span></dt><dd><p>Winner&#x27;s predictions in query order, respecting its prediction policies.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If a successful search has not produced <code>best_estimator_</code>.</p></dd>
</dl>


</div>

<span id="rgram.CoverageSearchCV.score"></span>

### score {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L672">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">score</span>(
    X,
    y,
    sample_weight=None,
)</code></pre></div>


Return `coefficient of determination` on test data.

The coefficient of determination, $R^2$, is defined as
$(1 - \frac{u}{v})$, where $u$ is the residual
sum of squares `((y_true - y_pred)** 2).sum()` and $v$
is the total sum of squares `((y_true - y_true.mean()) ** 2).sum()`.
The best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always predicts
the expected value of `y`, disregarding the input features, would get
a $R^2$ score of 0.0.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_samples, n_features)</span></dt><dd><p>Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape <code>(n_samples, n_samples_fitted)</code>, where <code>n_samples_fitted</code> is the number of samples used in the fitting for the estimator.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, n_outputs)</span></dt><dd><p>True values for <code>X</code>.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">array-like of shape (n_samples,), default=None</span></dt><dd><p>Sample weights.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">score</code> <span class="api-field-type">float</span></dt><dd><p><span class="arithmatex">\(R^2\)</span> of <code>self.predict(X)</code> w.r.t. <code>y</code>.</p></dd>
</dl>

#### Notes

The $R^2$ score used when calling `score` on a regressor uses
`multioutput='uniform_average'` from version 0.23 to keep consistent
with default value of `sklearn.metrics.r2_score`.
This influences the `score` method of all the multioutput
regressors (except for
`sklearn.multioutput.MultiOutputRegressor`).


</div>

<span id="rgram.CoverageSearchCV.set_fit_request"></span>

### set_fit_request {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_metadata_requests.py#L1440">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_fit_request</span>(
    *,
    data=&#x27;$UNCHANGED$&#x27;,
    groups=&#x27;$UNCHANGED$&#x27;,
    sample_weight=&#x27;$UNCHANGED$&#x27;,
)</code></pre></div>


Configure whether metadata should be requested to be passed to the `fit` method.

Note that this method is only relevant when this estimator is used as a
sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
with `enable_metadata_routing=True` (see `sklearn.set_config`).
Please check the [metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `fit` if provided. The request is ignored if metadata is not provided.

- `False`: metadata is not requested and the meta-estimator will not pass it to `fit`.

- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.

- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the
existing request. This allows you to change the request for some
parameters and not others.
Added in version 1.3.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">data</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>data</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">groups</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>groups</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>fit</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>


</div>

<span id="rgram.CoverageSearchCV.set_params"></span>

### set_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L400">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_params</span>(**params)</code></pre></div>


Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as `sklearn.pipeline.Pipeline`). The latter have
parameters of the form `<component>__<parameter>` so that it's
possible to update each component of a nested object.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">**params</code> <span class="api-field-type">dict</span></dt><dd><p>Estimator parameters.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">estimator instance</span></dt><dd><p>Estimator instance.</p></dd>
</dl>


</div>

<span id="rgram.CoverageSearchCV.set_score_request"></span>

### set_score_request {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_metadata_requests.py#L1440">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_score_request</span>(*, sample_weight=&#x27;$UNCHANGED$&#x27;)</code></pre></div>


Configure whether metadata should be requested to be passed to the `score` method.

Note that this method is only relevant when this estimator is used as a
sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
with `enable_metadata_routing=True` (see `sklearn.set_config`).
Please check the [metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `score` if provided. The request is ignored if metadata is not provided.

- `False`: metadata is not requested and the meta-estimator will not pass it to `score`.

- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.

- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the
existing request. This allows you to change the request for some
parameters and not others.
Added in version 1.3.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>score</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>


</div>

</div>
<!-- END REFERENCE:rgram.CoverageSearchCV -->

Implementation source: [model_selection.py](https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/model_selection.py).
