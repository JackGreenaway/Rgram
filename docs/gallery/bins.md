# Choosing a bin count

## What this example reviews

A regressogram turns a feature-response relationship into a table of local
summaries. This example asks how much detail that table should retain. Both fits
receive the same 150 observations and use the same mean aggregation; only the
number of quantile bins changes. The feature coordinates are evenly spaced here,
so quantile cells are approximately equal in width as well as occupancy.

```{image} ../_static/gallery/bins.png
:alt: Choosing a bin count: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## Choices and their impact

| Item                   | What is being reviewed                   | Impact and interpretation                                                                                                                     |
| ---------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Gray observations      | The responses supplied to both fits      | Their scatter shows individual variation that a bin average hides. All rows are retained.                                                     |
| Dashed generating mean | The sine function used in the simulation | Provides a reference for evaluating smoothing here; it would be unknown in ordinary data.                                                     |
| Six-bin curve          | A coarse mean summary                    | Roughly 25 observations contribute per cell in this dataset. Broad patterns are easier to describe, but changes inside a cell are unresolved. |
| Eighteen-bin curve     | A finer mean summary                     | Roughly 8-9 observations contribute per cell here. More local detail appears, with less averaging of response noise.                          |
| Dense query grid       | Evaluation at 350 feature locations      | Makes the display clear; it does not add observations, change boundaries, or improve statistical resolution.                                  |

## Interpreting the plot

Follow the broad rise and fall before focusing on individual steps. If both bin
counts show the same broad pattern, that feature of the summary is less dependent
on this particular resolution choice. A sharper step in the finer curve may be
noise, curvature resolved more finely, or the placement of a boundary. It is not
by itself evidence of a physical threshold.

Each flat segment estimates an average across its cell, not a point-specific
response at every coordinate along the segment. Inspect `bins_` for counts,
observed ranges, and nominal boundaries before interpreting a small cell's value.
With uneven feature density, equal-count cells can have very different widths;
with ties, their counts need not be equal.

## When this is useful

Use this comparison to choose an understandable summary table, inspect broad
nonlinearity, or check whether an apparent feature-response pattern persists
across several bin counts. It is especially useful when counts and ranges must
be reported alongside response summaries.

This in-sample comparison does not identify an optimal predictive bin count or
adjust for other variables. For predictive selection, compare held-out error and
coverage rather than choosing whichever curve looks most detailed.

## Explore further

Try `n_bins=3` and `n_bins=30`, then inspect `bins_["n_samples"]`. Repeat with
`binning="width"` on an unevenly distributed feature to see how a fixed-width
partition differs from a quantile partition. Refit whenever bin settings change.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example bins
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example bins --output-dir /tmp/rgram-figures
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
:pyobject: bins
```
