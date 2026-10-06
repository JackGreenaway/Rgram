# Inspecting observation influence

See which training rows contribute to a query. The query is fixed at feature value 3. Equal-weight and weighted smoothers use the same observations and bandwidth. The second fit gives observations above the query three times as much influence.

```{image} ../_static/gallery/influence.png
:alt: Inspecting observation influence: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

The plotted weights sum to one and follow original training-row order. `effective_n` measures concentration, not confidence or regression degrees of freedom. Only one query is inspected, avoiding a large dense matrix.

The main controls in this example are **KernelSmoother.get_weights and sample_weight**. See the
[related guide](../api/kernel_smoother.md) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example influence
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example influence --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: influence
```
