# Means, medians, and response spread

## What this example reviews

This example reviews the statistical summary of the response, rather than the
smoothing resolution. The same observations and nine quantile bins are used for
the mean and median. Every twelfth response is deliberately increased by 2.8
units; no observation is removed. The median fit also reports descriptive 10th
and 90th response percentiles within each bin.

```{image} ../_static/gallery/summaries.png
:alt: Means, medians, and response spread: a single large plot of the example's observations or diagnostics.
:width: 100%
```

## Choices and their impact

| Item | What is being reviewed | Impact and interpretation |
|---|---|---|
| Gray observations, including high responses | The full response distribution supplied to both fits | The extreme responses remain visible and participate in aggregation. |
| Blue mean curve | Arithmetic average within each bin | Every response value contributes to the total; unusually large responses can raise the average. |
| Orange median curve | Middle response within each bin | Responds differently to the deliberately added high values because it describes the center by rank, not the arithmetic average. |
| Shaded percentile envelope | Within-bin 10th–90th response percentiles | Describes response spread among observed rows in a cell, not uncertainty about the fitted mean or median. |
| Shared binning | A controlled comparison of summary statistics | Differences between these curves come from aggregation rather than different cell boundaries. |

## Interpreting the plot

Look at how high responses pull the mean relative to the median, then relate the
percentile envelope to the observations in each feature interval. The better
summary depends on the question: a mean is relevant to an average or total-based
quantity, while a median describes a typical middle response.

The envelope is returned through `summary_lower` and `summary_upper` in diagnostics
because it is a descriptive bin summary. It is not a bootstrap interval, a
simultaneous band, or a guaranteed coverage interval for future responses.
A wider envelope can reflect more heterogeneous responses without necessarily
meaning the central curve is estimated less precisely.

## When this is useful

Use this comparison when exploring skewed responses, sensitivity to extreme
observations, or differences between an average and a typical response. Quantile
summaries can also reveal spread that a single central curve conceals.

Choosing a median changes the target statistic; it is not an automatic cleaning
step and does not make the kernel smoother a robust-regression estimator. Small
bin samples can make quantiles unstable, so inspect counts before treating the
envelope as a detailed description of the response distribution.

## Explore further

Change the size or frequency of the deliberately high responses. Try other
`quantile(q)` values or a different bin count, then distinguish changes in the
central summary from changes in the descriptive spread. For uncertainty about a
chosen fitted statistic, use `predict_interval` and its resampling assumptions.

## Run this example

From the checkout, with `rgram[plot]` installed:

```bash
python examples/plot_relationships.py --example summaries
# Save just this figure without opening a desktop window:
python examples/plot_relationships.py --example summaries --output-dir /tmp/rgram-figures
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
:pyobject: summaries
```
