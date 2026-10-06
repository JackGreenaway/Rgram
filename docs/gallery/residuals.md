# Inspecting residual structure

Check what heavy smoothing leaves behind. This single-panel residual plot uses the observations and predictions returned by `regression_diagnostics`. A deliberately large bandwidth leaves some systematic feature-dependent pattern in the residuals.

```{image} ../_static/gallery/residuals.png
:alt: Inspecting residual structure: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

Residuals are observed minus predicted. Look for structure alongside support and raw observations. These in-sample residuals are exploratory diagnostics, not held-out prediction error or a formal lack-of-fit test.

The main controls in this example are **regression_diagnostics**. See the
[related guide](../api/results.md#residual-diagnostics) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example residuals
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example residuals --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: residuals
```
