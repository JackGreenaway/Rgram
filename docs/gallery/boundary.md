# Local-linear boundary behavior

## What this example reviews

This example compares a local constant with a local linear regression near the
ends of an observed feature range. The generating mean is a straight line, so
edge behavior is easier to distinguish from unknown curvature. Both fits use
the same data, Gaussian kernel, bandwidth 0.22, and evaluation locations.

```{image} ../_static/gallery/boundary.png
:alt: Local-linear boundary behavior: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## Choices and their impact

| Item | What is being reviewed | Impact and interpretation |
|---|---|---|
| Gray observations | Noisy samples around a linear mean | Supply the same local information to both estimators. |
| Dashed generating line | The reference relationship | Allows visible edge deviations to be interpreted in this simulation. |
| Local-constant curve | A distance-weighted average of nearby responses | Near an edge, observations mostly lie on one side, pulling the average toward that side's responses. |
| Local-linear curve | A local weighted intercept and slope | Uses a slope to describe the one-sided neighborhood and can reduce the local constant's boundary bias. |
| Shaded edge regions | Locations where neighborhoods are asymmetric | Guide attention to boundary behavior. These are neither confidence bounds nor unsupported regions. |

## Interpreting the plot

Compare each curve's distance from the dashed reference near the left and right
edges, then compare their behavior in the interior. Local linear fitting can be
helpful where averaging over a one-sided neighborhood shifts the estimate away
from the relationship at the query itself.

A local linear curve does not impose one global straight line: a separate local
fit is evaluated at every query. Its equivalent response coefficients can be
negative, so values can overshoot the observed response range. When a slope cannot
be resolved, inspect `local_linear_fallback`; the configured singular policy
controls whether the estimator falls back or raises.

## When this is useful

Use this comparison when edge behavior matters in a feature–response description
or when a local-constant curve seems to flatten toward the training boundaries.
It separates the choice of local fitting order from the choice of bandwidth.

The example evaluates inside the observed range. Improved behavior at its edges
does not validate predictions beyond the range, and local linear fitting does
not guarantee a monotone or bounded response curve.

## Explore further

Reduce the sample size or create repeated feature values to inspect fallback
flags. Compare `get_weights([query], kind="prediction")` with `kind="kernel"`
near an edge to see how signed local-linear coefficients differ from positive
kernel weights.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example boundary
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example boundary --output-dir /tmp/rgram-figures
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
:pyobject: boundary
```
