# KernelSmoother

Use this estimator to inspect a local response curve. [Kernel theory](../theory.md#kernel-regression-a-moving-neighborhood)
and [bandwidth/kernel parameters](../statistical_parameters.md) explain the
smoothing controls. See [result schemas](results.md) for diagnostics and snapshots.

## What it solves

You want to see a changing local response without imposing a global straight line
or polynomial. `KernelSmoother` pools observations with distance-based weights,
producing a smooth curve you can evaluate at chosen feature locations.

## When to use it

Use it when a continuous local mean is more useful than fixed interval summaries.
Bandwidth sets the smoothing scale in feature units. Compare bandwidths and
inspect support; choose local-linear fitting when a local slope is useful, with
the [boundary example](../gallery/boundary.md) explaining its behavior.

## Example: compare smoothing scales

```python
import numpy as np
from rgram import KernelSmoother

X = np.linspace(0.0, 6.0, 40).reshape(-1, 1)
y = np.sin(X[:, 0])
query = np.linspace(0.0, 6.0, 12).reshape(-1, 1)
for bandwidth in (0.3, 1.0):
    model = KernelSmoother(
        kernel="gaussian", bandwidth="manual", bandwidth_value=bandwidth,
    ).fit(X, y)
    print(bandwidth, model.predict(query))
    assert np.isfinite(model.predict(query)).all()
```

The narrower bandwidth follows more local shape; the broader one averages across
more of the feature range. This noiseless illustration makes that difference easy
to inspect. With real data, compare residuals and observation influence as well.
The [bandwidth gallery](../gallery/bandwidth.md) repeats the comparison with noise.

<!-- BEGIN REFERENCE:rgram.KernelSmoother -->
<span id="rgram.KernelSmoother"></span>

## rgram.KernelSmoother

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py#L23">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">rgram.</span><span class="api-name">KernelSmoother</span>(
    bandwidth=&#x27;silverman&#x27;,
    bandwidth_value=None,
    bandwidth_adjust=1.0,
    kernel=&#x27;epanechnikov&#x27;,
    n_eval_samples=100,
    *,
    batch_size=128,
    unsupported=&#x27;nan&#x27;,
    regression=&#x27;local_constant&#x27;,
    singular=&#x27;constant&#x27;,
    algorithm=&#x27;auto&#x27;,
    extrapolation=&#x27;allow&#x27;,
)</code></pre></div>


<div class="api-class-body" markdown="1">

One-dimensional local-constant or local-linear kernel regression.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">bandwidth</code> <span class="api-field-type">{&#x27;silverman&#x27;, &#x27;scott&#x27;, &#x27;manual&#x27;}, default=&#x27;silverman&#x27;</span></dt><dd><p>Rule of thumb or manual bandwidth selection. Rules are based on x, not regression error; use cross-validation to tune predictive accuracy.</p></dd>
<dt><code class="api-field-name">bandwidth_value</code> <span class="api-field-type">float or None, default=None</span></dt><dd><p>Positive bandwidth in feature units, required for &#x27;manual&#x27;.</p></dd>
<dt><code class="api-field-name">bandwidth_adjust</code> <span class="api-field-type">float, default=1.0</span></dt><dd><p>Positive multiplier applied to the selected bandwidth.</p></dd>
<dt><code class="api-field-name">kernel</code> <span class="api-field-type">str or callable, default=&#x27;epanechnikov&#x27;</span></dt><dd><p>One of epanechnikov, gaussian, uniform, triangular, cosine, logistic, biweight, tricube. A callable accepts a Polars expression and returns nonnegative weights.</p></dd>
<dt><code class="api-field-name">n_eval_samples</code> <span class="api-field-type">int, default=100</span></dt><dd><p>Number of points returned by predict_grid().</p></dd>
<dt><code class="api-field-name">batch_size</code> <span class="api-field-type">int, default=128</span></dt><dd><p>Maximum query rows processed together. Blocks are also capped at approximately one million query/training pairs (at least one query).</p></dd>
<dt><code class="api-field-name">unsupported</code> <span class="api-field-type">{&#x27;nan&#x27;, &#x27;raise&#x27;}, default=&#x27;nan&#x27;</span></dt><dd><p>Policy when a query has no positive kernel weight. &#x27;nan&#x27; preserves the row and warns; &#x27;raise&#x27; fails explicitly. Extrapolation has a separate policy.</p></dd>
<dt><code class="api-field-name">regression</code> <span class="api-field-type">{&#x27;local_constant&#x27;, &#x27;local_linear&#x27;}, default=&#x27;local_constant&#x27;</span></dt><dd><p>Local constant (Nadaraya-Watson) or weighted local linear fit. Local linear reduces boundary bias but can extrapolate and use negative coefficients.</p></dd>
<dt><code class="api-field-name">singular</code> <span class="api-field-type">{&#x27;constant&#x27;, &#x27;raise&#x27;}, default=&#x27;constant&#x27;</span></dt><dd><p>Fall back with a warning to a local constant for singular local-linear fits, or raise. Diagnostics always identify fallback rows.</p></dd>
<dt><code class="api-field-name">algorithm</code> <span class="api-field-type">{&#x27;auto&#x27;, &#x27;brute&#x27;, &#x27;neighbors&#x27;}, default=&#x27;auto&#x27;</span></dt><dd><p>Auto uses support windows only for already-sorted x and compact kernels; otherwise bounded dense blocks. Neighbors requires eligible input and raises otherwise. This estimator never sorts observations or an index.</p></dd>
<dt><code class="api-field-name">extrapolation</code> <span class="api-field-type">{&#x27;allow&#x27;, &#x27;nan&#x27;, &#x27;raise&#x27;}, default=&#x27;allow&#x27;</span></dt><dd><p>Outside-range queries warn and use the stated policy. Query values are never clipped. Allow still requires positive kernel support.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">X_</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Independent snapshot of original training features in row order.</p></dd>
<dt><code class="api-field-name">y_</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Independent snapshot of original training responses in row order.</p></dd>
<dt><code class="api-field-name">sample_weight_</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Original supplied weights, or ones if absent. Independent snapshot.</p></dd>
<dt><code class="api-field-name">sample_weight_provided_</code> <span class="api-field-type">bool</span></dt><dd><p>Whether observation weights were supplied during fit.</p></dd>
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray of shape (1,)</span></dt><dd><p>Training feature name, present only for named input.</p></dd>
<dt><code class="api-field-name">data_summary_</code> <span class="api-field-type">dict</span></dt><dd><p>Data integrity, original dtype, fitted policy and computation summary.</p></dd>
<dt><code class="api-field-name">training_sorted_</code> <span class="api-field-type">bool</span></dt><dd><p>Whether training features were already nondecreasing.</p></dd>
<dt><code class="api-field-name">algorithm_</code> <span class="api-field-type">str</span></dt><dd><p>Computational path selected during fitting. Prediction resolves the current algorithm option again, so changing it can alter the path.</p></dd>
<dt><code class="api-field-name">weight_scale_</code> <span class="api-field-type">float</span></dt><dd><p>Maximum training observation weight used for computational normalization.</p></dd>
<dt><code class="api-field-name">bandwidth_</code> <span class="api-field-type">float</span></dt><dd><p>Effective fitted bandwidth, including bandwidth_adjust.</p></dd>
<dt><code class="api-field-name">n_samples_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of training rows retained, including zero-weight rows.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Always one. Both 1D inputs and (n_samples, 1) arrays are accepted.</p></dd>
</dl>

### Notes

Training inputs are copied. Null, non-finite and multivariate inputs are
rejected, never silently dropped. Kernel inspection columns follow the
original training row order. Compact-kernel prediction visits only supported
neighbors when training x is already sorted. No sorting is performed. Infinite-support
and custom kernels use O(n_train * n_query) time with bounded blocks. get_weights() explicitly
materializes a dense matrix and should be used on small query selections.


### Methods

| Method | Purpose |
| --- | --- |
| [fit](#rgram.KernelSmoother.fit) | Fit one numeric feature and one numeric response. |
| [fit_predict](#rgram.KernelSmoother.fit_predict) | Fit and evaluate at the original training feature values. |
| [get_metadata_routing](#rgram.KernelSmoother.get_metadata_routing) | Get metadata routing of this object. |
| [get_params](#rgram.KernelSmoother.get_params) | Get parameters for this estimator. |
| [get_weights](#rgram.KernelSmoother.get_weights) | Inspect the dense matrix of local influences in training-row order. |
| [predict](#rgram.KernelSmoother.predict) | Evaluate the local-constant or local-linear response curve. |
| [predict_diagnostics](#rgram.KernelSmoother.predict_diagnostics) | Return local support and numerical information for each query. |
| [predict_grid](#rgram.KernelSmoother.predict_grid) | Evaluate diagnostics on an evenly spaced training-range grid. |
| [predict_interval](#rgram.KernelSmoother.predict_interval) | Compute pointwise paired-bootstrap confidence intervals for the curve. |
| [regression_diagnostics](#rgram.KernelSmoother.regression_diagnostics) | Inspect observed values, fitted values, and residuals in input order. |
| [score](#rgram.KernelSmoother.score) | Return :ref:`coefficient of determination <r2_score>` on test data. |
| [set_fit_request](#rgram.KernelSmoother.set_fit_request) | Configure whether metadata should be requested to be passed to the ``fit`` method. |
| [set_params](#rgram.KernelSmoother.set_params) | Set the parameters of this estimator. |
| [set_score_request](#rgram.KernelSmoother.set_score_request) | Configure whether metadata should be requested to be passed to the ``score`` method. |

<span id="rgram.KernelSmoother.fit"></span>

### fit {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py#L185">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(
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

<span id="rgram.KernelSmoother.fit_predict"></span>

### fit_predict {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py#L596">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit_predict</span>(
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
Fit followed by predict evaluates new queries; use predict_interval
for bootstrap confidence bounds.


</div>

<span id="rgram.KernelSmoother.get_metadata_routing"></span>

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

<span id="rgram.KernelSmoother.get_params"></span>

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

<span id="rgram.KernelSmoother.get_weights"></span>

### get_weights {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py#L499">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_weights</span>(
    X,
    *,
    kind=&#x27;prediction&#x27;,
)</code></pre></div>


Inspect the dense matrix of local influences in training-row order.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_queries,) or (n_queries, 1)</span></dt><dd><p>Query feature values.</p></dd>
<dt><code class="api-field-name">kind</code> <span class="api-field-type">{&#x27;prediction&#x27;, &#x27;kernel&#x27;}, default=&#x27;prediction&#x27;</span></dt><dd><p>Prediction coefficients or normalized nonnegative kernel/observation weights. Local-linear prediction coefficients can be negative.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">weights</code> <span class="api-field-type">ndarray of shape (n_queries, <code>n_samples_in_</code>)</span></dt><dd><p>Columns follow original training order. Unsupported rows contain zeros. Supported prediction rows combine <code>y_</code> into predicted values. For local constants, prediction and kernel weights coincide.</p></dd>
</dl>

#### Notes

This intentionally allocates a dense matrix. Use small query selections;
ordinary prediction does not require storing this full matrix.


</div>

<span id="rgram.KernelSmoother.predict"></span>

### predict {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py#L549">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict</span>(X)</code></pre></div>


Evaluate the local-constant or local-linear response curve.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_queries,) or (n_queries, 1)</span></dt><dd><p>Query feature values. Named inputs must match the fitted feature.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">prediction</code> <span class="api-field-type">ndarray of shape (n_queries,)</span></dt><dd><p>One value per query in input order. Unsupported estimates are NaN unless the configured support policy raises.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If no successful fit is available.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>For invalid queries, feature-name mismatch, or strict policies.</p></dd>
</dl>


</div>

<span id="rgram.KernelSmoother.predict_diagnostics"></span>

### predict_diagnostics {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py#L526">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_diagnostics</span>(X)</code></pre></div>


Return local support and numerical information for each query.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_queries,) or (n_queries, 1)</span></dt><dd><p>One query feature, with fitted feature name if named.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">diagnostics</code> <span class="api-field-type">polars.DataFrame</span></dt><dd><p>Columns: x, in_training_range, prediction, supported, n_neighbors, effective_n, local_linear_fallback. One row per query in original order. Unsupported rows have NaN prediction, zero neighbors and effective_n, and false supported, unless a strict policy raises.</p></dd>
</dl>

#### Notes

effective_n = 1 / sum(normalized kernel/observation weights ** 2). This
measures weight concentration, not confidence or signed-coefficient variance.


</div>

<span id="rgram.KernelSmoother.predict_grid"></span>

### predict_grid {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py#L576">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_grid</span>()</code></pre></div>


Evaluate diagnostics on an evenly spaced training-range grid.

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">diagnostics</code> <span class="api-field-type">polars.DataFrame</span></dt><dd><p>The predict_diagnostics schema at n_eval_samples locations spanning the observed feature minimum and maximum, endpoints included.</p></dd>
</dl>

#### Notes

This changes evaluation locations, not smoothing. With a constant feature,
all grid coordinates are equal. Fit must have succeeded first.


</div>

<span id="rgram.KernelSmoother.predict_interval"></span>

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

<span id="rgram.KernelSmoother.regression_diagnostics"></span>

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

<span id="rgram.KernelSmoother.score"></span>

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

<span id="rgram.KernelSmoother.set_fit_request"></span>

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

<span id="rgram.KernelSmoother.set_params"></span>

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

<span id="rgram.KernelSmoother.set_score_request"></span>

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
<!-- END REFERENCE:rgram.KernelSmoother -->

Implementation source: [smoothing.py](https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/smoothing.py).
