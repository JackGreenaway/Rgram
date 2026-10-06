# Inspecting observation influence

## What this example reviews

This example examines which original training observations influence a single
query at feature value 3. Both smoothers use bandwidth 0.6 and the same feature
locations and responses. One gives all rows equal observation weight; the other
gives rows above the query three times as much observation weight.

```{image} ../_static/gallery/influence.png
:alt: Inspecting observation influence: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## Choices and their impact

| Item | What is being reviewed | Impact and interpretation |
|---|---|---|
| Horizontal axis | Original training feature locations | Each plotted value corresponds to a training row, not to a new prediction query. |
| Vertical axis | Normalized local kernel/observation weights | Describes relative influence at this one query. Each supported curve's weights sum to one. |
| Blue weight curve | Distance weighting with equal observation weights | Nearby rows receive more influence, with symmetric geometry in this example. |
| Orange weight curve | Distance weighting combined with unequal observation weights | Influence shifts toward the higher-weight side; other normalized weights also change because the total is renormalized. |
| Dashed line at 3 | The location being predicted | Makes it clear that both weight curves describe the same query. |
| Effective n in the legend | Concentration of normalized positive weights | Gives an equivalent equal-weight count for concentration, not confidence, sample-size assurance, or regression degrees of freedom. |

## Interpreting the plot

Compare where influence is concentrated and how it shifts after reweighting.
Observation weights multiply the kernel's distance weights before normalization;
they do not move observations or change the fitted bandwidth. Multiplying all
observation weights by the same positive constant would leave these normalized
weights unchanged.

For local-constant fitting, these plotted weights are also the coefficients that
combine responses into a prediction. This equality does not hold for local-linear
fitting: `kind="prediction"` can then return signed coefficients, while
`kind="kernel"` remains nonnegative. The effective-n diagnostic uses the latter.

## When this is useful

Inspect local influence when a fitted value seems surprising, a few observations
appear to dominate, or you need to explain what supplied weights do. The full
matrix follows original training-row order, so weights can be matched to the
stored responses.

Weights express a chosen influence convention. They do not automatically produce
survey-design inference or a causal estimate. A finite estimate dominated by one
row can still have very little local information. `get_weights` allocates a dense
query-by-training matrix; inspect a small query selection rather than an enormous grid.

## Explore further

Move the query or reduce bandwidth, then compare concentration and `effective_n`.
Set selected observation weights to zero and inspect which rows cease contributing.
Try a local-linear fit and compare the two `kind` options explicitly.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example influence
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example influence --output-dir /tmp/rgram-figures
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
:pyobject: influence
```
