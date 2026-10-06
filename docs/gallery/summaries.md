# Means, medians, and response spread

Choose a summary that answers your question. The data contain deliberately added extreme responses. The mean and median use the same nine quantile bins. The shaded region shows the 10th–90th response percentiles inside each bin, using explicitly configured descriptive endpoints.

```{image} ../_static/gallery/summaries.png
:alt: Means, medians, and response spread: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

The shaded region describes response spread, not uncertainty about the median and not future-observation prediction coverage. A median changes the statistical target; it does not clean observations or make kernel least squares robust.

The main controls in this example are **Regressogram.agg and ci**. See the
[related guide](../api/aggregation.md) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example summaries
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example summaries --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: summaries
```
