# Aggregations and custom reducers

`Regressogram.agg` and each optional descriptive `ci` endpoint accept named
statistics, native Polars expression functions, or the adapters below.

## Named reductions

| Name | Unweighted meaning | With observation weights |
|---|---|---|
| `mean` | Arithmetic mean | Weighted mean, undefined at zero total mass. |
| `sum` | Response total | Sum of weight × response; zero at zero mass. |
| `median` | Median | Requires an explicit weighted callback. |
| `min`, `max` | Extremal observed response | Requires an explicit weighted callback. |
| `count` | Number of non-null responses (all fitted responses are validated) | Requires an explicit weighted callback. |
| `std`, `var` | Polars sample standard deviation/variance with `ddof=1`; singleton gives null/unsupported | Requires an explicit weighted callback. |
| `first`, `last` | Response in original within-bin row order | Requires an explicit weighted callback. |

Callbacks must return one real numeric scalar per bin, or one Polars scalar
reduction expression for native callbacks. NaN is an explicit missing estimate;
infinity and nonscalar outputs are rejected. A bare callable means a native
Polars-expression callback, not an automatically detected NumPy function.

## Factory functions

<!-- BEGIN REFERENCE:rgram.array_aggregation -->
<span id="rgram.array_aggregation"></span>

## rgram.array_aggregation

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L184">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.</span><span class="api-name">array_aggregation</span>(
    func,
    *,
    weighted=False,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Wrap a NumPy/Python scalar function for agg or a ci endpoint.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">func</code> <span class="api-field-type">callable</span></dt><dd><p>Callback func(values), or func(values, weights) with weighted=True.</p></dd>
<dt><code class="api-field-name">weighted</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Whether the callback explicitly accepts paired observation weights.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">adapter</code> <span class="api-field-type">ArrayAggregation</span></dt><dd><p>Cloneable reducer, retaining the callable and weighting option.</p></dd>
</dl>

### Examples

```pycon
>>> import numpy as np
>>> from rgram import Regressogram, array_aggregation
>>> model = Regressogram(n_bins=1, agg=array_aggregation(np.median))
>>> model.fit([0., 1., 2.], [1., 3., 9.]).predict([1.])
array([3.])
```


</div>
<!-- END REFERENCE:rgram.array_aggregation -->

<!-- BEGIN REFERENCE:rgram.weighted_aggregation -->
<span id="rgram.weighted_aggregation"></span>

## rgram.weighted_aggregation

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L212">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.</span><span class="api-name">weighted_aggregation</span>(func)</code></pre></div>


<div class="api-class-body" markdown="1">

Wrap a native weighted Polars scalar reduction.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">func</code> <span class="api-field-type">callable</span></dt><dd><p>Callback func(values_expr, weights_expr) returning a Polars expression.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">adapter</code> <span class="api-field-type">WeightedAggregation</span></dt><dd><p>Cloneable native reducer. With no fitting weights, the adapter uses ones.</p></dd>
</dl>

### Examples

```pycon
>>> from rgram import Regressogram, weighted_aggregation
>>> reducer = weighted_aggregation(lambda values, weights:
...     (values * weights).sum() / weights.sum())
>>> model = Regressogram(n_bins=1, agg=reducer)
>>> model.fit([0., 1.], [2., 8.], sample_weight=[1., 2.]).predict([0.5])
array([6.])
```


</div>
<!-- END REFERENCE:rgram.weighted_aggregation -->

<!-- BEGIN REFERENCE:rgram.quantile -->
<span id="rgram.quantile"></span>

## rgram.quantile

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L237">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.</span><span class="api-name">quantile</span>(
    q,
    *,
    interpolation=&#x27;linear&#x27;,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Create an unweighted native Polars quantile reducer.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">q</code> <span class="api-field-type">float</span></dt><dd><p>Probability in [0, 1]. Validated when the reducer is evaluated.</p></dd>
<dt><code class="api-field-name">interpolation</code> <span class="api-field-type">str, default=&#x27;linear&#x27;</span></dt><dd><p>nearest, higher, lower, midpoint, linear, or equiprobable, passed to Polars.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">reducer</code> <span class="api-field-type">Quantile</span></dt><dd><p>Cloneable quantile callable for agg or either descriptive ci endpoint.</p></dd>
</dl>

### Examples

```pycon
>>> from rgram import Regressogram, quantile
>>> model = Regressogram(n_bins=1, agg=quantile(0.5))
>>> model.fit([0., 1., 2.], [1., 3., 9.]).predict([1.])
array([3.])
```


</div>
<!-- END REFERENCE:rgram.quantile -->

## Returned adapter objects

These classes are importable from `rgram.aggregation`. The factories are the
recommended way to create them. Their parameters participate in cloning and
nested tuning, for example `model.set_params(agg__q=0.9)` followed by refitting.
A custom callback must implement any desired zero-mass or weighted-quantile
convention explicitly; the library does not infer it.

<!-- BEGIN REFERENCE:rgram.aggregation.ArrayAggregation -->
<span id="rgram.aggregation.ArrayAggregation"></span>

## rgram.aggregation.ArrayAggregation

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L15">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">rgram.aggregation.</span><span class="api-name">ArrayAggregation</span>(
    func,
    *,
    weighted=False,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Adapt a numeric callback to a cloneable per-bin scalar reduction.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">func</code> <span class="api-field-type">callable</span></dt><dd><p>Function returning one real numeric scalar from values, or values and weights when weighted=True. Receives independent arrays in bin-row order.</p></dd>
<dt><code class="api-field-name">weighted</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Explicitly require a two-argument callback. If fitting weights are absent, the callback receives ones. Unweighted callbacks reject supplied weights.</p></dd>
</dl>

### Notes

Usually constructed with array_aggregation. No values are sorted, filtered, or
imputed. Callback exceptions propagate. NaN marks unsupported results; infinity
is rejected by Regressogram. This adapter inherits sklearn get_params/set_params.


### Methods

| Method | Purpose |
| --- | --- |
| [__call__](#rgram.aggregation.ArrayAggregation.__call__) | Apply the numeric callback to independent input copies. |
| [get_metadata_routing](#rgram.aggregation.ArrayAggregation.get_metadata_routing) | Get metadata routing of this object. |
| [get_params](#rgram.aggregation.ArrayAggregation.get_params) | Get parameters for this estimator. |
| [set_params](#rgram.aggregation.ArrayAggregation.set_params) | Set the parameters of this estimator. |

<span id="rgram.aggregation.ArrayAggregation.__call__"></span>

### __call__ {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L38">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.ArrayAggregation.</span><span class="api-name">__call__</span>(values, weights=None)</code></pre></div>


Apply the numeric callback to independent input copies.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">values</code> <span class="api-field-type">array-like</span></dt><dd><p>Responses in one bin.</p></dd>
<dt><code class="api-field-name">weights</code> <span class="api-field-type">array-like or None, default=None</span></dt><dd><p>Paired influence weights. A weighted adapter uses ones when absent.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">value</code> <span class="api-field-type">float</span></dt><dd><p>One real scalar. Nonscalar, complex, and imprecisely converted integer callback results raise ValueError.</p></dd>
</dl>


</div>

<span id="rgram.aggregation.ArrayAggregation.get_metadata_routing"></span>

### get_metadata_routing {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_metadata_requests.py#L1743">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.ArrayAggregation.</span><span class="api-name">get_metadata_routing</span>()</code></pre></div>


Get metadata routing of this object.

Please check [metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
mechanism works.

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">routing</code> <span class="api-field-type">MetadataRequest</span></dt><dd><p>A <code>sklearn.utils.metadata_routing.MetadataRequest</code> encapsulating routing information.</p></dd>
</dl>


</div>

<span id="rgram.aggregation.ArrayAggregation.get_params"></span>

### get_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L248">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.ArrayAggregation.</span><span class="api-name">get_params</span>(deep=True)</code></pre></div>


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

<span id="rgram.aggregation.ArrayAggregation.set_params"></span>

### set_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L400">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.ArrayAggregation.</span><span class="api-name">set_params</span>(**params)</code></pre></div>


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

</div>
<!-- END REFERENCE:rgram.aggregation.ArrayAggregation -->

<!-- BEGIN REFERENCE:rgram.aggregation.WeightedAggregation -->
<span id="rgram.aggregation.WeightedAggregation"></span>

## rgram.aggregation.WeightedAggregation

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L86">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">rgram.aggregation.</span><span class="api-name">WeightedAggregation</span>(
    func,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Adapt a native Polars weighted reduction.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">func</code> <span class="api-field-type">callable</span></dt><dd><p>Function accepting (values_expr, weights_expr) and returning one scalar reduction as a Polars expression. Usually use weighted_aggregation.</p></dd>
</dl>

### Notes

Without supplied weights, nonmissing values receive unit weights. The adapter
inherits sklearn get_params/set_params and is cloneable.


### Methods

| Method | Purpose |
| --- | --- |
| [__call__](#rgram.aggregation.WeightedAggregation.__call__) | Build the weighted Polars reduction expression. |
| [get_metadata_routing](#rgram.aggregation.WeightedAggregation.get_metadata_routing) | Get metadata routing of this object. |
| [get_params](#rgram.aggregation.WeightedAggregation.get_params) | Get parameters for this estimator. |
| [set_params](#rgram.aggregation.WeightedAggregation.set_params) | Set the parameters of this estimator. |

<span id="rgram.aggregation.WeightedAggregation.__call__"></span>

### __call__ {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L104">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.WeightedAggregation.</span><span class="api-name">__call__</span>(values, weights=None)</code></pre></div>


Build the weighted Polars reduction expression.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">values</code> <span class="api-field-type">polars.Expr</span></dt><dd><p>Expression selecting bin responses.</p></dd>
<dt><code class="api-field-name">weights</code> <span class="api-field-type">polars.Expr or None, default=None</span></dt><dd><p>Weight expression. None gives unit weights for nonmissing values.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">expression</code> <span class="api-field-type">polars.Expr</span></dt><dd><p>The expression returned by func; Regressogram checks the result type.</p></dd>
</dl>


</div>

<span id="rgram.aggregation.WeightedAggregation.get_metadata_routing"></span>

### get_metadata_routing {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_metadata_requests.py#L1743">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.WeightedAggregation.</span><span class="api-name">get_metadata_routing</span>()</code></pre></div>


Get metadata routing of this object.

Please check [metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
mechanism works.

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">routing</code> <span class="api-field-type">MetadataRequest</span></dt><dd><p>A <code>sklearn.utils.metadata_routing.MetadataRequest</code> encapsulating routing information.</p></dd>
</dl>


</div>

<span id="rgram.aggregation.WeightedAggregation.get_params"></span>

### get_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L248">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.WeightedAggregation.</span><span class="api-name">get_params</span>(deep=True)</code></pre></div>


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

<span id="rgram.aggregation.WeightedAggregation.set_params"></span>

### set_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L400">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.WeightedAggregation.</span><span class="api-name">set_params</span>(**params)</code></pre></div>


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

</div>
<!-- END REFERENCE:rgram.aggregation.WeightedAggregation -->

<!-- BEGIN REFERENCE:rgram.aggregation.Quantile -->
<span id="rgram.aggregation.Quantile"></span>

## rgram.aggregation.Quantile

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L126">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">rgram.aggregation.</span><span class="api-name">Quantile</span>(
    q,
    *,
    interpolation=&#x27;linear&#x27;,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Cloneable Polars quantile reduction with explicit interpolation.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">q</code> <span class="api-field-type">float</span></dt><dd><p>Quantile probability in [0, 1], excluding booleans.</p></dd>
<dt><code class="api-field-name">interpolation</code> <span class="api-field-type">str, default=&#x27;linear&#x27;</span></dt><dd><p>One of nearest, higher, lower, midpoint, linear, equiprobable. Passed directly to Polars quantile.</p></dd>
</dl>

### Notes

Usually constructed with quantile. Observation weights are unsupported;
choose an explicit weighted adapter if you require weighted quantiles.
Inherits sklearn get_params/set_params.


### Methods

| Method | Purpose |
| --- | --- |
| [__call__](#rgram.aggregation.Quantile.__call__) | Build an unweighted quantile expression. |
| [get_metadata_routing](#rgram.aggregation.Quantile.get_metadata_routing) | Get metadata routing of this object. |
| [get_params](#rgram.aggregation.Quantile.get_params) | Get parameters for this estimator. |
| [set_params](#rgram.aggregation.Quantile.set_params) | Set the parameters of this estimator. |

<span id="rgram.aggregation.Quantile.__call__"></span>

### __call__ {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py#L148">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.Quantile.</span><span class="api-name">__call__</span>(values)</code></pre></div>


Build an unweighted quantile expression.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">values</code> <span class="api-field-type">polars.Expr</span></dt><dd><p>Expression selecting bin responses.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">expression</code> <span class="api-field-type">polars.Expr</span></dt><dd><p>Quantile reduction with the configured probability and interpolation.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If q or interpolation is invalid.</p></dd>
</dl>


</div>

<span id="rgram.aggregation.Quantile.get_metadata_routing"></span>

### get_metadata_routing {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_metadata_requests.py#L1743">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.Quantile.</span><span class="api-name">get_metadata_routing</span>()</code></pre></div>


Get metadata routing of this object.

Please check [metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
mechanism works.

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">routing</code> <span class="api-field-type">MetadataRequest</span></dt><dd><p>A <code>sklearn.utils.metadata_routing.MetadataRequest</code> encapsulating routing information.</p></dd>
</dl>


</div>

<span id="rgram.aggregation.Quantile.get_params"></span>

### get_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L248">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.Quantile.</span><span class="api-name">get_params</span>(deep=True)</code></pre></div>


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

<span id="rgram.aggregation.Quantile.set_params"></span>

### set_params {.api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L400">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.aggregation.Quantile.</span><span class="api-name">set_params</span>(**params)</code></pre></div>


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

</div>
<!-- END REFERENCE:rgram.aggregation.Quantile -->

A callback may transform its independent input copies, but Rgram adds no sorting
or filtering around it. Named reducers and adapters with importable top-level
callbacks can be pickled. Lambdas and closures may not work with standard pickle.

Implementation source: [aggregation.py](https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/aggregation.py).
