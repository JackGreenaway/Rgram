# Local-linear boundary behavior

Compare local constants and local slopes. The synthetic generating relationship is linear, while the training feature range has two boundaries. The shaded edge regions make it easy to compare the local-constant and local-linear curves where neighborhoods are asymmetric.

```{image} ../_static/gallery/boundary.png
:alt: Local-linear boundary behavior: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

A local slope can reduce boundary bias, but it can also produce values outside the observed response range. Inspect `local_linear_fallback` if a neighborhood cannot resolve a slope. The example evaluates inside the training range.

The main controls in this example are **KernelSmoother.regression**. See the
[related guide](../theory.md#local-linear-regression-and-boundaries) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example boundary
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example boundary --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: boundary
```
