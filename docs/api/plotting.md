# Plotting

<!-- BEGIN REFERENCE:rgram.plot_diagnostics -->
<span id="rgram.plot_diagnostics"></span>

## rgram.plot_diagnostics

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/plotting.py#L19">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">rgram.</span><span class="api-name">plot_diagnostics</span>(
    estimator,
    X,
    y,
)</code></pre></div>


<div class="api-class-body" markdown="1">

Plot residuals and observed-versus-predicted values without sorting rows.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">estimator</code> <span class="api-field-type">Regressogram or KernelSmoother</span></dt><dd><p>Successfully fitted model providing regression_diagnostics. For a pipeline, use the final estimator with transformed feature coordinates.</p></dd>
<dt><code class="api-field-name">X</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, 1)</span></dt><dd><p>Evaluation feature values, aligned with y.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,) or (n_samples, 1)</span></dt><dd><p>Observed responses, for training or held-out observations.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">figure</code> <span class="api-field-type">matplotlib.figure.Figure</span></dt><dd><p>Figure containing both diagnostic panels and a support-count title.</p></dd>
<dt><code class="api-field-name">axes</code> <span class="api-field-type">ndarray of shape (2,)</span></dt><dd><p>Residual-versus-feature axis and observed-versus-predicted axis.</p></dd>
</dl>

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ImportError</code></dt><dd><p>If Matplotlib is absent; install rgram[plot].</p></dd>
<dt><code class="api-field-name">sklearn.exceptions.NotFittedError</code></dt><dd><p>If the estimator has no successful fit.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If evaluation pairs are invalid or strict prediction policies raise.</p></dd>
</dl>

### Notes

Uses scatter points, never connecting or sorting observation rows.
Unsupported rows remain in the diagnostic table and are counted in the
title, though non-finite points cannot be drawn. No data is modified.
The caller chooses whether to display, save, or close the returned figure.


</div>
<!-- END REFERENCE:rgram.plot_diagnostics -->

The helper expects a Rgram estimator and feature values in that estimator's
coordinate system. For a pipeline, transform the features through
`pipeline[:-1]` and pass its final estimator. It returns a Matplotlib figure and
a two-element array of axes; calling `plt.show()` or saving the figure is up to
the caller. See the [worked examples](../examples.md).

Implementation source: [plotting.py](https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/plotting.py).
