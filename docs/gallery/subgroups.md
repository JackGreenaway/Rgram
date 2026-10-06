# Comparing subgroups

## What this example reviews

This example reviews whether an observed feature-response relationship differs
between two explicitly defined subgroups. Both groups cover the same feature
range and are fitted with identical smoothing settings. Group B's generating
mean has a known 0.8 response-unit offset; that construction makes a comparison
easy to interpret without claiming a real-world group effect.

```{image} ../_static/gallery/subgroups.png
:alt: Comparing subgroups: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## Choices and their impact

| Item                          | What is being reviewed                                           | Impact and interpretation                                                                                                     |
| ----------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Blue and orange points        | Responses within each subgroup                                   | Show within-group variation and which feature locations have observations.                                                    |
| Separate fitted curves        | Local averages conditional on feature within each selected group | Each curve uses only its own group's rows; the groups are not pooled into a single fit.                                       |
| Shared bandwidth 0.4          | A consistent smoothing scale                                     | Makes differences in this display easier to compare without confounding the visual comparison with different bandwidth rules. |
| Common feature range and grid | Comparable evaluation coordinates                                | Curves can be compared at the same locations here. Real groups may have little or no overlapping support.                     |
| Vertical separation           | Difference between fitted local summaries                        | Describes an observed association between subgroup and response at a feature location; it is not a causal or adjusted effect. |

## Interpreting the plot

Compare the groups at the same feature values, rather than comparing their overall
response means. Ask whether the difference resembles a roughly constant offset,
a change in shape, or a difference visible only in sparsely sampled regions.
Keep the observations visible so a smooth curve does not conceal a small or
unevenly sampled subgroup.

Identical smoothing settings make the comparison easier to audit, but they do not
guarantee equal estimation quality when group sizes or feature densities differ.
For real data, select each subgroup explicitly, inspect support and sample sizes,
and limit comparisons to ranges where both fits have relevant observations.

## When this is useful

Use this workflow to explore relationships across cohorts, segments, experiments,
or measurement conditions and to generate questions for a later joint analysis.
It helps distinguish a pooled trend from subgroup-specific descriptions.

Separate curves do not control for other differences between groups, establish
causality, or supply a formal test of curve equality. Apparent separation can
reflect differing compositions or sampling, not an intervention on group membership.

## Explore further

Reduce one group's sample size or restrict its feature range. Inspect each fit's
`predict_diagnostics` before comparing curves. Try a common narrower and wider
bandwidth to check whether the apparent shape difference is smoothing-sensitive.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example subgroups
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example subgroups --output-dir /tmp/rgram-figures
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
:pyobject: subgroups
```
