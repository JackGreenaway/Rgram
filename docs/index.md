<div class="docs-hero" markdown="1">

# Rgram

<p class="lead">See how a response changes across a feature.</p>

**A scatter plot shows observations. Rgram helps you summarize the relationship.**
Explore whether demand rises with temperature, a response levels off with age,
or two selected groups follow different curves. Fit interpretable bin summaries
or smooth local estimates, then inspect the observations supporting each result.

Rgram combines **Polars diagnostic tables** with **scikit-learn-style estimators**.
Choose your response statistic, control the neighborhood size, and reuse the fit
for curves, diagnostics, or held-out prediction.

- **Describe nonlinear shape:** inspect local response changes without choosing a global polynomial.
- **Make the summary inspectable:** review bin counts, local influence, residuals, and prediction support.
- **Use a familiar workflow:** fit, predict, clone, and tune the estimator inside scikit-learn pipelines.

<div class="docs-actions" markdown="1">

[Getting started](getting_started.md)
[API reference](api/index.md)
[Examples](examples.md)

</div>
</div>

## The problems it helps solve

| Analysis question | What you use | What you can inspect |
| --- | --- | --- |
| A noisy scatter plot makes the broad relationship hard to see. | `KernelSmoother` with an explicit bandwidth. | A smooth local curve and its sensitivity to the neighborhood scale. |
| You need a compact summary that can be explained as intervals. | `Regressogram` with a chosen bin count and aggregation. | Per-bin response estimates, counts, and boundaries in `bins_`. |
| A few large responses make the mean a poor answer to your question. | A regressogram median or explicit quantile. | How the typical response or a chosen percentile varies across the feature. |
| Selected groups may have different response patterns. | A separate fit for each group on common query coordinates. | Descriptive curve differences together with each group's local support. |
| A fitted curve gives a value where few observations contribute. | Prediction diagnostics and kernel influence weights. | Support, concentration of weights, and whether the query is outside the observed range. |
| Parameter choices must also work on unseen rows. | A preprocessing pipeline and held-out evaluation; `CoverageSearchCV` when finite-prediction coverage matters. | Validation error, supported fractions, and the selected parameters. |

Start with the [bandwidth comparison](gallery/bandwidth.md),
[response summaries](gallery/summaries.md), or [subgroup example](gallery/subgroups.md)
according to the question you want to answer.

## Why use it?

### Choose the statistic that answers your question

A mean, median, and percentile describe different aspects of a response.
`Regressogram` makes that choice explicit and returns a table you can review or
share. Use named reductions for ordinary summaries or an
[aggregation adapter](api/aggregation.md) for your own scalar statistic.

### Explore shape without fixing a global equation

`KernelSmoother` estimates the response using nearby observations. Its bandwidth
controls how local the estimate is; local-linear fitting also estimates a local
slope. Compare several bandwidths to see which patterns persist. The
[theory guide](theory.md) explains the assumptions behind each curve.

### Review the evidence behind a fitted value

Inspect counts, support flags, residuals, and normalized observation influence
rather than reading the curve alone. Diagnostics remain in the original row
order and return Polars DataFrames. See the [influence example](gallery/influence.md)
and [result schemas](api/results.md).

### Move from exploration to evaluated prediction

The same estimators support `fit`, `predict`, cloning, parameter search, and
scikit-learn pipelines. Keep learned preprocessing inside the pipeline and
measure held-out error when prediction is your goal. The
[pipeline guide](sklearn.md) provides runnable workflows.

## Example: summarize demand across temperature

Here is a small illustrative dataset with demand increasing and then leveling
off. Fit a binned summary and a smooth curve to the same observations:

```python
import numpy as np
import polars as pl

from rgram import KernelSmoother, Regressogram

observations = pl.DataFrame({
    "temperature": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
    "demand": [20.0, 24.0, 35.0, 48.0, 60.0, 65.0, 66.0, 67.0],
})
binned = Regressogram(n_bins=4).fit(
    "temperature", "demand", data=observations,
)
smooth = KernelSmoother(
    kernel="gaussian", bandwidth="manual", bandwidth_value=2.0,
).fit("temperature", "demand", data=observations)

query = pl.DataFrame({"temperature": np.linspace(10.0, 24.0, 8)})
print(binned.bins_)
print(smooth.predict_diagnostics(query))
assert np.isfinite(smooth.predict(query)).all()
```

`bins_` provides four interval summaries; `predict_diagnostics` describes the
smooth estimates at the requested temperatures. The bandwidth is in temperature
units. These fits describe the supplied observations. To assess the leveling-off
pattern, collect more observations and compare bandwidths; to assess predictive
accuracy, evaluate on held-out rows. The [quick start](getting_started.md) shows
how to plot the result, and the [gallery](examples.md) explores the choices.

## When to use it

Use Rgram for a numeric response and **one numeric feature at a time**, especially
when your goal is pairwise exploration, an interpretable response summary, or an
inspectable local predictor. Arrays, numeric pandas columns, and Polars columns
are accepted; optional pandas and plotting support have separate installation extras.

Fit separate curves for other features or explicitly selected subgroups.
Pairwise curves do not adjust for other variables or identify causal effects.
A multivariate prediction task needs an appropriate multivariate estimator.
Bootstrap curve intervals also differ from future-observation prediction
intervals. See [theory and suitability](theory.md) for interpretation.

## Explore the documentation

<div class="docs-topics" markdown="1">
<div class="docs-topic" markdown="1">

### Binned response summaries

Choose intervals and the response statistic you want to describe.

**Component:** [Regressogram](api/regressogram.md).

[Bin-count example](gallery/bins.md) · [Mean and median](gallery/summaries.md)

</div>
<div class="docs-topic" markdown="1">

### Smooth local curves

Compare neighborhood scales, local slopes, and observation influence.

**Component:** [KernelSmoother](api/kernel_smoother.md).

[Bandwidth example](gallery/bandwidth.md) · [Boundary behavior](gallery/boundary.md)

</div>
<div class="docs-topic" markdown="1">

### Evaluation and selection

Fit preprocessing within each training fold and inspect prediction coverage.

**Component:** [CoverageSearchCV](api/coverage_search.md).

[Pipeline guide](sklearn.md) · [Support example](gallery/support.md)

</div>
<div class="docs-topic" markdown="1">

### Diagnostics and interpretation

Review residuals, local support, and pointwise curve uncertainty.

[Residual example](gallery/residuals.md) · [Bootstrap intervals](gallery/intervals.md)

[Result schemas](api/results.md)

</div>
</div>

## API stability

Rgram has an **unstable API**. Parameters, methods, and result schemas may change
between releases without backward compatibility or a deprecation period.
Pin the package version for reproducible analyses. This site's reference describes
the source revision used to build it.
