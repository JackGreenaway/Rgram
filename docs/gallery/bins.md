# Choosing a bin count

Compare coarse and fine regressograms. A small bin count gives a compact response summary. More bins resolve finer local variation, but also give each cell fewer observations. The generating mean is shown only because this example uses synthetic data.

```{image} ../_static/gallery/bins.png
:alt: Choosing a bin count: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

Compare the two curves on the same observations. A jump at a bin edge is a consequence of the partition, not evidence of a real threshold. Inspect `model.bins_` for cell counts and limits.

The main controls in this example are **Regressogram.n_bins**. See the
[related guide](../theory.md#regressograms-fixed-cells-and-local-summaries) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example bins
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example bins --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: bins
```
