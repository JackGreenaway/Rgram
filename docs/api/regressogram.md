# Regressogram

Use this estimator for compact, interpretable summaries of a feature-response
relationship. [Theory and interpretation](../theory.md#regressograms-fixed-cells-and-local-summaries)
and [bin/aggregation parameters](../statistical_parameters.md) explain its
statistical choices. See [result schemas](results.md) for `bins_` and diagnostics.

## What it solves

A scatter plot can contain too many individual responses to summarize clearly.
`Regressogram` groups a feature into intervals and computes one response statistic
per interval, giving you a compact table and a piecewise constant curve.

## When to use it

Use it when interval boundaries and counts help explain the relationship, or when
you need a median, percentile, or custom per-bin statistic. Increasing the bin
count retains more local detail while reducing the observations pooled in each
cell. Compare [bin counts](../gallery/bins.md) and
[response summaries](../gallery/summaries.md) before settling on a configuration.

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
large response. See [result schemas](results.md) for the table columns.

<!-- BEGIN REFERENCE:rgram.Regressogram -->
<span id="rgram.Regressogram"></span>

## rgram.Regressogram

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/rgram.py#L24">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">rgram.</span><span class="api-name">Regressogram</span>(
    *,
    binning=&#x27;dist&#x27;,
    agg=&#x27;mean&#x27;,
    ci=None,
    n_bins=None,
    bin_width=None,
    max_bins=512,
    min_samples_bin=5,
    cv=5,
    bin_candidates=None,
    random_state=0,
    extrapolation=&#x27;clip&#x27;,
    unsupported=&#x27;nan&#x27;,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Regressogram

Binned regression estimator for one feature and target.
Predicts using binned aggregation with optional confidence intervals.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">binning</code> <span class="api-field-type">{&#x27;dist&#x27;, &#x27;width&#x27;, &#x27;none&#x27;, &#x27;int&#x27;}, default=&#x27;dist&#x27;</span></dt><dd><p>Binning strategy.</p></dd>
<dt><code class="api-field-name">agg</code> <span class="api-field-type">str or callable, default=&#x27;mean&#x27;</span></dt><dd><p>Named reduction, Polars expression callable, or array_aggregation adapter. Every bin must produce one real numeric scalar; weights are never ignored.</p></dd>
<dt><code class="api-field-name">ci</code> <span class="api-field-type">tuple of aggregations or None, default=None</span></dt><dd><p>Explicit lower/upper descriptive summaries. No intervals are computed by default. Use predict_interval for pointwise bootstrap inference.</p></dd>
<dt><code class="api-field-name">n_bins</code> <span class="api-field-type">int, str or None, default=None</span></dt><dd><p>Count or selection rule for both dist and width. None means &#x27;auto&#x27;. Rules: auto, cv, fd, scott, sturges, sqrt, rice. Dist auto uses a bounded cube-root count; width auto combines FD and Sturges. CV minimizes held-out regression MSE and requires full prediction support in every fold.</p></dd>
<dt><code class="api-field-name">bin_width</code> <span class="api-field-type">float or None, default=None</span></dt><dd><p>Explicit width in feature units; requires width binning and n_bins=None. The last bin may be shorter. Counts that exceed max_bins raise.</p></dd>
<dt><code class="api-field-name">max_bins</code> <span class="api-field-type">int, default=512</span></dt><dd><p>Allocation guard. Automatic rules are capped; larger explicit counts raise.</p></dd>
<dt><code class="api-field-name">min_samples_bin</code> <span class="api-field-type">int, default=5</span></dt><dd><p>Bounds the default dist-auto count and generated CV candidates by n // value. This is a target occupancy, not a guarantee with ties or explicit counts.</p></dd>
<dt><code class="api-field-name">cv</code> <span class="api-field-type">int, splitter or iterable, default=5</span></dt><dd><p>Used only for n_bins=&#x27;cv&#x27;. Integers use shuffled KFold.</p></dd>
<dt><code class="api-field-name">bin_candidates</code> <span class="api-field-type">sequence of int or None, default=None</span></dt><dd><p>Candidate counts for CV. Defaults to a bounded geometric grid.</p></dd>
<dt><code class="api-field-name">random_state</code> <span class="api-field-type">int or None, default=0</span></dt><dd><p>Seed for default CV splitting.</p></dd>
<dt><code class="api-field-name">extrapolation</code> <span class="api-field-type">{&#x27;clip&#x27;, &#x27;nan&#x27;, &#x27;raise&#x27;}, default=&#x27;clip&#x27;</span></dt><dd><p>Outside the training range, map to edge bins (with a warning), leave unsupported, or raise. Query values are never modified.</p></dd>
<dt><code class="api-field-name">unsupported</code> <span class="api-field-type">{&#x27;nan&#x27;, &#x27;raise&#x27;}, default=&#x27;nan&#x27;</span></dt><dd><p>Preserve missing estimates as NaN with a warning, or raise.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">X_</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Independent snapshot of original training features in row order.</p></dd>
<dt><code class="api-field-name">y_</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Independent snapshot of original training responses in row order.</p></dd>
<dt><code class="api-field-name">sample_weight_</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Original supplied weights, or ones if absent. Independent snapshot.</p></dd>
<dt><code class="api-field-name">sample_weight_provided_</code> <span class="api-field-type">bool</span></dt><dd><p>Whether observation weights were supplied during fit.</p></dd>
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of shape (1,)</span></dt><dd><p>Training feature name, present only for named input.</p></dd>
<dt><code class="api-field-name">data_summary_</code> <span class="api-field-type">dict</span></dt><dd><p>Data integrity, original dtype, fitted policy and computation summary.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Always one.</p></dd>
<dt><code class="api-field-name">n_samples_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of rows, including zero-weight rows.</p></dd>
<dt><code class="api-field-name">selected_n_bins_</code> <span class="api-field-type">int, str or None</span></dt><dd><p>Count selected by internal CV, or original n_bins option without CV.</p></dd>
<dt><code class="api-field-name">requested_n_bins_</code> <span class="api-field-type">int</span></dt><dd><p>Nominal cell count after rule/cap selection but before tied boundaries are merged. Present for dist/width only. Sample/constant-x constraints can reduce an explicit request before this attribute is assigned.</p></dd>
<dt><code class="api-field-name">n_bins_</code> <span class="api-field-type">int</span></dt><dd><p>Number of fitted cells (empty cells may exist).</p></dd>
<dt><code class="api-field-name">bin_edges_</code> <span class="api-field-type">ndarray</span></dt><dd><p>Interior fitted boundaries for dist/width binning.</p></dd>
<dt><code class="api-field-name">bin_width_</code> <span class="api-field-type">float or None</span></dt><dd><p>Width for width binning; None for other strategies.</p></dd>
<dt><code class="api-field-name">bins_</code> <span class="api-field-type">polars.DataFrame</span></dt><dd><p>Occupied bin estimates and original observation counts.</p></dd>
<dt><code class="api-field-name">cv_results_</code> <span class="api-field-type">dict</span></dt><dd><p>Error and coverage by candidate, available when n_bins=&#x27;cv&#x27;.</p></dd>
</dl>


### Methods

| Method | Purpose |
| --- | --- |
| [fit](#rgram.Regressogram.fit) | Fit one numeric feature and one numeric response. |
| [fit_predict](#rgram.Regressogram.fit_predict) | Fit and evaluate at the original training feature values. |
| [get_metadata_routing](#rgram.Regressogram.get_metadata_routing) | Get metadata routing of this object. |
| [get_params](#rgram.Regressogram.get_params) | Get parameters for this estimator. |
| [predict](#rgram.Regressogram.predict) | Evaluate the fitted bin summary at each query. |
| [predict_diagnostics](#rgram.Regressogram.predict_diagnostics) | Return bin assignment and support information for each query. |
| [predict_interval](#rgram.Regressogram.predict_interval) | Compute pointwise paired-bootstrap confidence intervals for the curve. |
| [regression_diagnostics](#rgram.Regressogram.regression_diagnostics) | Inspect observed values, fitted values, and residuals in input order. |
| [score](#rgram.Regressogram.score) | Return :ref:`coefficient of determination <r2_score>` on test data. |
| [set_fit_request](#rgram.Regressogram.set_fit_request) | Configure whether metadata should be requested to be passed to the ``fit`` method. |
| [set_params](#rgram.Regressogram.set_params) | Set the parameters of this estimator. |
| [set_score_request](#rgram.Regressogram.set_score_request) | Configure whether metadata should be requested to be passed to the ``score`` method. |

<span id="rgram.Regressogram.fit"></span>

### fit {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/rgram.py#L359">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(
    X,
    y,
    *,
    data=None,
    sample_weight=None,
)</code></pre></div>


Fit one numeric feature and one numeric response.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, 1), or str</span></dt><dd><p>Numeric feature, or its column name when data is supplied.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, 1), or str</span></dt><dd><p>Numeric response, or its column name when data is supplied.</p></dd>
<dt><code class="api-field-name">data</code> <span class="api-field-type">pandas.DataFrame, polars.DataFrame or polars.LazyFrame, default=None</span></dt><dd><p>Source for named columns. Pandas columns are converted internally to Polars. Selected columns are snapshotted once; indices are ignored.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">array-like of shape (n_samples,), default=None</span></dt><dd><p>Finite nonnegative influence weights with positive total mass.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">estimator</span></dt><dd><p>Fitted estimator, with independent training snapshots.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>For incompatible shapes, unequal lengths, missing/non-finite values, invalid settings, or unsupported weighting conventions.</p></dd>
<dt><code class="api-field-name">TypeError</code></dt><dd><p>For nonnumeric/complex input or invalid callback/data types.</p></dd>
</dl>

#### Notes

Rows and duplicates are preserved. Unsorted input emits an advisory warning.
A failed refit invalidates prediction access. Zero-weight rows still participate
in feature-based bin or bandwidth selection. Normal fitting does not run CV;
Regressogram(n_bins='cv') explicitly requests it.


</div>

<span id="rgram.Regressogram.fit_predict"></span>

### fit_predict {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/rgram.py#L637">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_predict</span>(
    X,
    y,
    *,
    data=None,
    sample_weight=None,
)</code></pre></div>


Fit and evaluate at the original training feature values.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like or str</span></dt><dd><p>One training feature, or its column name when data is supplied.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like or str</span></dt><dd><p>Aligned numeric response, or its column name when data is supplied.</p></dd>
<dt><code class="api-field-name">data</code> <span class="api-field-type">pandas.DataFrame, polars.DataFrame or polars.LazyFrame, default=None</span></dt><dd><p>Source for named columns.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">array-like of shape (n_samples,), default=None</span></dt><dd><p>Observation influence weights; see fit.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">prediction</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Predictions at original training rows, preserving their order.</p></dd>
</dl>

#### Notes

Training predictions describe the fit, not held-out performance.
Fit followed by predict evaluates new queries; predict_interval returns
bootstrap bounds, and predict_diagnostics exposes descriptive endpoints.


</div>

<span id="rgram.Regressogram.get_metadata_routing"></span>

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

<span id="rgram.Regressogram.get_params"></span>

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

<span id="rgram.Regressogram.predict"></span>

### predict {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/rgram.py#L573">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict</span>(X)</code></pre></div>


Evaluate the fitted bin summary at each query.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_queries,) or (n_queries, 1)</span></dt><dd><p>Query feature values. Named inputs must match the fitted feature.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">prediction</code> <span class="api-field-type">ndarray of shape (n_queries,)</span></dt><dd><p>One estimate per query, in input order. Unsupported estimates are NaN unless the configured support policy raises.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If no successful fit is available.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>For invalid queries, feature-name mismatch, or strict policies.</p></dd>
</dl>


</div>

<span id="rgram.Regressogram.predict_diagnostics"></span>

### predict_diagnostics {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/rgram.py#L601">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_diagnostics</span>(X)</code></pre></div>


Return bin assignment and support information for each query.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_queries,) or (n_queries, 1)</span></dt><dd><p>One query feature, with fitted feature name if named.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">diagnostics</code> <span class="api-field-type">polars.DataFrame</span></dt><dd><p>Columns: x, prediction, rgram_bin, n_samples, weight_sum, n_positive_weight, supported, in_training_range. With ci configured, summary_lower and summary_upper expose descriptive bin endpoints, not bootstrap confidence bounds. Each row corresponds to the query at the same position. Missing bin information is null; unsupported predictions are NaN unless a strict policy raises.</p></dd>
</dl>


</div>

<span id="rgram.Regressogram.predict_interval"></span>

### predict_interval {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/base.py#L606">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_interval</span>(
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

<span id="rgram.Regressogram.regression_diagnostics"></span>

### regression_diagnostics {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/base.py#L568">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">regression_diagnostics</span>(X, y)</code></pre></div>


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

<span id="rgram.Regressogram.score"></span>

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

<span id="rgram.Regressogram.set_fit_request"></span>

### set_fit_request {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_metadata_requests.py#L1440">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_fit_request</span>(
    *,
    data=&#x27;$UNCHANGED$&#x27;,
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
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>fit</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>


</div>

<span id="rgram.Regressogram.set_params"></span>

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

<span id="rgram.Regressogram.set_score_request"></span>

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
<!-- END REFERENCE:rgram.Regressogram -->

Implementation source: [rgram.py](https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/rgram.py).
