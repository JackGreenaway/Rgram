# Examples

Explore one relationship or diagnostic at a time. Each example has its own page
and a large **10 × 6.6 inch figure**, saved at **180 DPI** with readable labels and
legends. Click a plot to inspect the original image at full resolution.

The examples use reproducible synthetic data, so some plots can show the known
generating relationship. Real datasets usually do not supply that ground truth.
See [theory and suitability](theory.md) before interpreting a fitted curve.

```{toctree}
:maxdepth: 1

gallery/bins
gallery/bandwidth
gallery/subgroups
gallery/boundary
gallery/summaries
gallery/influence
gallery/intervals
gallery/support
gallery/residuals
```

## Run one plot at a time

```bash
python examples/plot_relationships.py --example bandwidth
```

Each page includes its own command and the relevant plotting function. With no
`--example` argument, the script displays each figure separately; close the
current window to continue to the next one. Install Matplotlib through
`python -m pip install 'rgram[plot]'` first.

To reproduce all documentation figures without a desktop:

```bash
python examples/plot_relationships.py --output-dir docs/_static/gallery
```

Download {download}`the complete gallery script <../examples/plot_relationships.py>`.
Figures are generated from the same functions shown on the example pages.

## Interactive kernel explorer

```bash
uv run python examples/kernel_explorer.py
```

The explorer provides controls for kernel, bandwidth, query location, local
constant/linear fitting, binning, and aggregation. It displays local influence
weights, support counts, effective sample size, and fallback flags. Bootstrap
intervals are calculated only after pressing the interval button. It requires a
desktop backend; GitHub Pages cannot run Python widgets. Download
{download}`the explorer script <../examples/kernel_explorer.py>` and run it locally.

## Exact neighbor and brute-force paths

```bash
uv run python examples/benchmark_neighbors.py
```

This script supplies already sorted features and a compact tricube kernel, then
times the automatic support-window path against bounded brute-force prediction.
It verifies numerical agreement. Results depend on hardware, sample size,
query count, and bandwidth; they are not a universal speed claim. Download
{download}`the benchmark script <../examples/benchmark_neighbors.py>`.

## Pipelines, selection, and weighted fitting

The [scikit-learn guide](sklearn.md) contains executable examples for fitting,
cloning, preprocessing pipelines, `GridSearchCV`, selecting one feature from a
wider table, and step-prefixed fitting weights. Custom callback examples are in
[the aggregation API](api/aggregation.md), and coverage-aware selection is shown
in [the search reference](api/coverage_search.md).
