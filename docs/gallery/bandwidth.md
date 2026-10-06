# Choosing a bandwidth

## What this example reviews

Kernel regression pools nearby responses using distance-dependent weights. This
example reviews the neighborhood scale: the same observations, Gaussian kernel,
local-constant fit, and query grid are used throughout, while the manual bandwidth
changes. The comparison isolates smoothing rather than a change in data or model
family.

```{image} ../_static/gallery/bandwidth.png
:alt: Choosing a bandwidth: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## Choices and their impact

| Item | What is being reviewed | Impact and interpretation |
|---|---|---|
| Gray observations | Original response variation | Shows the noise that the curves are averaging, rather than eliminating from the data. |
| Dashed generating mean | Known synthetic signal | Helps distinguish recovered structure from extra wiggles or blurred peaks. |
| Bandwidth 0.12 | A narrow distance scale | Concentrates influence locally. The curve follows more detail, which can include sample noise. |
| Bandwidth 0.45 | An intermediate distance scale | Pools more neighboring responses, trading some detail for a steadier curve. It is illustrative, not a recommended universal value. |
| Bandwidth 1.2 | A broad distance scale | Averages across larger parts of the relationship and can flatten peaks, valleys, and transitions. |
| Gaussian kernel and fixed grid | What is deliberately held constant | Bandwidth is a scale in feature units, not a hard radius: the Gaussian has infinite mathematical support. Adding query points changes display resolution only. |

## Interpreting the plot

Compare which broad features survive all three fits, then identify what appears
only at the narrowest scale or disappears at the broadest. More smoothness is not
automatically better, and greater detail is not automatically more faithful. A
small bandwidth may fit random variation; a large bandwidth may average over a
real change in the local mean.

The same numeric bandwidth has a different meaning after changing feature units.
For example, scaling a feature before fitting changes the units in which a manual
bandwidth is interpreted. Compare settings in the coordinate system supplied to
the estimator, and refit after changing them.

## When this is useful

Use this comparison when checking the sensitivity of an exploratory curve,
looking for nonlinear shape, or choosing a readable initial smoothing scale.
Patterns that vary strongly across plausible scales deserve closer inspection
alongside the raw observations and local support.

The plot does not certify which bandwidth predicts new responses best. If
prediction error is the goal, use cross-validation with preprocessing inside the
pipeline, and ensure validation predictions have adequate support.

## Explore further

Change `bandwidth_value`, inspect `predict_diagnostics(query)["effective_n"]`,
and compare the curve with how concentrated its local weights become. Then try a
compact kernel: changing the kernel also changes the meaning of support and can
require retuning bandwidth.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example bandwidth
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example bandwidth --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below is the implementation used to generate this plot. Inline
comments explain the modeling decisions and interpretation of diagnostic values.
It returns a figure; the CLI handles displaying or saving it and then closes it.
The [shared helpers](../examples.md#code-structure-and-shared-helpers) supply only
common styling and explicit Gaussian/manual-bandwidth configuration. Run the
complete downloaded script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: bandwidth
```
