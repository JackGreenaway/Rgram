# Recognizing unsupported queries

Leave a visible gap where there is no estimate. The observations occupy two separated feature ranges. A compact Epanechnikov kernel cannot reach any observations near the middle of the gap. Unsupported predictions remain NaN, breaking the plotted line instead of silently connecting the two regions.

```{image} ../_static/gallery/support.png
:alt: Recognizing unsupported queries: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

The orange region marks unsupported queries. A Gaussian kernel could supply finite values there, but those values would not establish that the relationship is known inside the gap. The script filters the expected support advisory only because the missing support is explicitly plotted.

The main controls in this example are **predict_diagnostics.supported**. See the
[related guide](../user_guide.md#unsupported-locations-and-extrapolation) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example support
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example support --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: support
```
