# Choosing a bandwidth

Compare smoothing at three feature scales. The same observations are fitted using three manual bandwidths. Each bandwidth is measured in feature units; the query grid stays fixed so changes in the curve come from smoothing rather than plotting resolution.

```{image} ../_static/gallery/bandwidth.png
:alt: Choosing a bandwidth: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

The smallest bandwidth follows more local variation. The largest blurs peaks and valleys. Compare plausible settings rather than assuming the most detailed curve is the most reliable.

The main controls in this example are **KernelSmoother.bandwidth_value**. See the
[related guide](../theory.md#smoothing-trades-resolution-for-stability) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example bandwidth
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example bandwidth --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: bandwidth
```
