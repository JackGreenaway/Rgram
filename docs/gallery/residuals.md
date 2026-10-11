# Inspecting residual structure

## What this example reviews

This example reviews what a fitted curve leaves unexplained. A deliberately broad
Gaussian bandwidth of 1.4 is applied to a noisy sine relationship. Residuals are
evaluated at the original training features and plotted individually, with no
extra smoothing of the residuals.

![Inspecting residual structure: a single large plot of the example's observations or diagnostics.](../_static/gallery/residuals.png)

## Choices and their impact

| Item                          | What is being reviewed                    | Impact and interpretation                                                                         |
| ----------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Horizontal feature coordinate | Where each original observation occurred  | Retains the positional link between a feature value and its residual.                             |
| Blue residual points          | Observed response minus fitted response   | Positive values mean the fit underpredicts the observation; negative values mean it overpredicts. |
| Dashed zero line              | Reference for an exactly fitted response  | Helps reveal runs of residuals with the same sign and deviations from zero.                       |
| Bandwidth 1.4                 | Deliberately heavy smoothing              | Pools broad regions, making it easier to see mean structure the fit has smoothed away.            |
| In-sample evaluation          | A diagnostic of the supplied observations | Describes the fitted training relationship, not generalization error on independent data.         |

## Interpreting the plot

Look for feature-dependent runs of positive or negative residuals rather than
requiring each point to lie close to zero. A curved mean pattern can indicate
that the smoothing scale is too broad for the relationship. Individual large
residuals can arise from noise, unusual responses, or inadequate fit and deserve
context rather than automatic removal.

Also examine changes in vertical spread, local support, and the underlying response
plot. Mean structure and changing variability are different phenomena. A visually
patternless residual plot does not establish independence, constant variance, or
correct model specification; a visible pattern is a diagnostic prompt, not a p-value.

## When this is useful

Use residuals to check whether an exploratory curve misses systematic shape and
to identify regions needing closer examination. The diagnostic table records
observed, predicted, residual, and support values together, making unexpected
points easier to inspect.

Unsupported residuals remain NaN in the table and cannot be drawn. For predictive
performance, pass aligned held-out pairs to `regression_diagnostics` and evaluate
appropriate scores. An in-sample curve chosen for very small residuals may simply
fit noise.

## Explore further

Compare bandwidths 0.4 and 1.4 on the same data. Fit a local-linear curve and
inspect residuals near the feature boundaries. Finally, evaluate a separate
held-out sample to distinguish training diagnostics from prediction error.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example residuals
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example residuals --output-dir /tmp/rgram-figures
```

Download [the complete gallery script](https://github.com/JackGreenaway/Rgram/blob/main/examples/plot_relationships.py).
The function below is the implementation used to generate this plot. Inline
comments explain the modeling decisions and interpretation of diagnostic values.
It returns a figure; the CLI handles displaying or saving it and then closes it.
The [shared helpers](../examples.md#code-structure-and-shared-helpers) supply only
common styling and explicit Gaussian/manual-bandwidth configuration. Run the
complete downloaded script using the command above.

```python
def residuals() -> Figure:
    """Use residuals to inspect structure remaining after heavy smoothing."""
    rng = np.random.default_rng(22)
    X = np.linspace(0, 6, 150)
    y = np.sin(X) + rng.normal(0, 0.15, len(X))
    # Intentionally smooth heavily to leave a visible mean pattern in residuals.
    model = _gaussian_smoother(1.4).fit(X, y)
    # Residuals are observed minus predicted, evaluated at the original rows.
    # These are in-sample diagnostics, not a held-out accuracy estimate.
    diagnostic = model.regression_diagnostics(X, y)
    figure, ax = _create_axes(
        "Residuals reveal structure left by heavy smoothing",
        ylabel="Observed − predicted",
    )
    ax.scatter(
        X,
        diagnostic["residual"].to_numpy(),
        color=COLORS[0],
        alpha=0.65,
        s=32,
        edgecolors="none",
        label="Residuals at original rows",
    )
    # The zero line helps distinguish systematic underprediction and overprediction.
    ax.axhline(0, color="#334155", linestyle="--", lw=2, label="Zero residual")
    _add_legend(ax)
    return figure
```
