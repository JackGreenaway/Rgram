# Pointwise bootstrap intervals

Display uncertainty about a fitted curve. A Gaussian smoother is fitted to independent synthetic pairs with a randomly sampled feature and a noisy sine response. The features are explicitly sorted when constructing this plotting example. Paired IID bootstrap refits produce pointwise 95% bounds at 100 feature locations, using 100 resamples and a fixed random seed. The generating mean helps make smoothing bias visible.

```{image} ../_static/gallery/intervals.png
:alt: Pointwise bootstrap intervals: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

A filled region is a convenient display of pointwise intervals; it is not a simultaneous band and does not describe future response noise. This example assumes independent random-design sampling. Increasing resamples cannot correct smoothing bias.

The main controls in this example are **predict_interval**. See the
[related guide](../theory.md#uncertainty-dependence-and-evaluation) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example intervals
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example intervals --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: intervals
```
