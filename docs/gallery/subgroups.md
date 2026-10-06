# Comparing subgroups

Fit one curve per explicitly selected group. Both groups have observations on the same feature range. They use identical kernels, bandwidths, and query locations. Each group is fitted separately, so the plot describes the relationship within each observed subgroup.

```{image} ../_static/gallery/subgroups.png
:alt: Comparing subgroups: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## What to look for

Check overlap and local sample sizes before comparing real groups. These curves do not adjust for other variables, identify a causal group effect, or provide a formal test of curve differences.

The main controls in this example are **Separate fits with a shared bandwidth**. See the
[related guide](../theory.md#where-rgram-is-useful) for the statistical assumptions and API details.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example subgroups
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example subgroups --output-dir /tmp/rgram-figures
```

Download {download}`the complete gallery script <../../examples/plot_relationships.py>`.
The function below comes from that script. Its shared helpers create the large
figure, apply consistent styling, and configure the Gaussian smoother; run the
complete script using the command above.

```{literalinclude} ../../examples/plot_relationships.py
:language: python
:pyobject: subgroups
```
