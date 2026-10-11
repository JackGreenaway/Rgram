# Getting started

Start with a concrete question: **how does the typical response change across
one numeric feature?** This walkthrough fits the same observations with bins and
a smooth curve, shows how to inspect and plot the results, and explains which
choice to revisit next. No cross-validation runs unless you request it.

## API stability

Rgram has an **unstable API**. Parameters, methods, and result schemas may change
between releases without backward compatibility or a deprecation period.
Pin the package version for reproducible analyses.

## Installation

Install the library and optional plotting support:

```bash
python -m pip install rgram
python -m pip install 'rgram[plot]'
```

Python 3.9 or newer is required. NumPy, Polars, and scikit-learn are runtime
dependencies; Matplotlib is optional. Minimum declared versions are NumPy 2.0.2,
Polars 1.28.1, and scikit-learn 1.6. The pandas extra requires pandas 2.2.3 or
newer; plotting requires Matplotlib 3.9.4 or newer. This site's reference describes the source
revision used to build it, which may contain changes newer than a published wheel.
For the current checkout:

```bash
python -m pip install -e '.[plot]'
```

## Your first feature-response curve

The synthetic response below has a curved signal plus noise. A regressogram
summarizes six intervals; a Gaussian smoother pools nearby observations using a
bandwidth of 0.4 feature units. Keeping the observations fixed makes the two
summaries easier to compare.

```python
import numpy as np
from rgram import KernelSmoother, Regressogram

X = np.linspace(0, 6, 80).reshape(-1, 1)
y = np.sin(X[:, 0]) + np.random.default_rng(42).normal(0, 0.15, len(X))

binned = Regressogram(n_bins=6).fit(X, y)
smooth = KernelSmoother(
    kernel="gaussian", bandwidth="manual", bandwidth_value=0.4,
).fit(X, y)
query = np.linspace(0, 6, 100).reshape(-1, 1)

print(binned.bins_)
print(smooth.predict_diagnostics(query))
print(smooth.regression_diagnostics(X, y))
```

Both models describe one feature at a time. `fit` returns the model, and
`predict(query)` returns a NumPy array with one value for each query, in input
order. Diagnostics return Polars DataFrames. Cross-validation does not run in
this example. Unsorted inputs are accepted with an advisory warning and retain
their original order.

### Read the result

- `binned.bins_` contains the fitted intervals, counts, and response summaries.
- `smooth.predict_diagnostics(query)` describes predictions and support at the query locations.
- `smooth.regression_diagnostics(X, y)` compares observations with fitted values.

A query grid controls where you evaluate the curve; it does not add training
observations. Compare a smaller and a larger bandwidth before interpreting small
wiggles as a persistent pattern. See the [bandwidth example](gallery/bandwidth.md).

### Plot the comparison

With the plotting extra installed, reuse the fitted models above:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X[:, 0], y, alpha=0.5, label="Observed responses")
ax.step(query[:, 0], binned.predict(query), where="mid", label="Bin summaries")
ax.plot(query[:, 0], smooth.predict(query), label="Smooth local mean")
ax.set(xlabel="Feature", ylabel="Response")
ax.legend()
plt.show()
plt.close(fig)
```

The dots show response variation; the lines summarize that variation at different
scales. This is a descriptive fit on all rows. Use held-out data to measure
prediction performance rather than treating training residuals as that measure.

### Start from named columns

For named Polars columns:

```python
import polars as pl

frame = pl.DataFrame({"temperature": X[:, 0], "demand": y})
model = Regressogram(n_bins=6).fit(
    X="temperature", y="demand", data=frame,
)
print(model.predict(pl.DataFrame({"temperature": [1.0, 2.0, 3.0]})))
```

## Using pandas

Pandas is optional. If it is not already installed, use
`python -m pip install 'rgram[pandas]'`. Conversion does not require PyArrow.

```python
import pandas as pd
from rgram import Regressogram

frame = pd.DataFrame({
    "temperature": [0.0, 1.0, 2.0, 3.0],
    "demand": [1.0, 2.0, 2.5, 4.0],
})
model = Regressogram(n_bins=2).fit(
    X="temperature", y="demand", data=frame,
)
print(model.predict(frame[["temperature"]]))

# Standard scikit-learn form is also supported:
model.fit(frame[["temperature"]], frame["demand"])
print(model.predict_diagnostics(frame[["temperature"]]))
```

Only selected numeric columns are converted internally to Polars. Unselected
columns and the pandas index are ignored. Rows, duplicates, feature names, and
caller-owned data are preserved. Separate feature, response, and weight inputs
are paired **by row position**, never aligned by index labels. Predictions remain
NumPy arrays and diagnostic tables remain Polars DataFrames.

Nullable numeric pandas dtypes such as `Int64`, `Float64`, and `Float32` are
accepted when complete. Missing, non-finite, complex, and nonnumeric selected
values are rejected. A duplicated selected column label is ambiguous and raises.

## Next steps

- [Choose a workflow](user_guide.md) for exploration, summaries, or prediction.
- [Understand the theory and limits](theory.md) before interpreting a curve.
- [Use scikit-learn pipelines](sklearn.md) for preprocessing and evaluation.
- [Look up an API](api/index.md) for exact parameters and results.
- [Run a worked example](examples.md) to plot smoothing choices and subgroup curves.
