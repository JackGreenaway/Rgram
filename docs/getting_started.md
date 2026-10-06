# Getting started

## Installation

Install the library and optional plotting support:

```bash
python -m pip install rgram
python -m pip install 'rgram[plot]'
```

Python 3.9 or newer is required. NumPy, Polars, and scikit-learn are runtime
dependencies; Matplotlib is optional. This site's reference describes the source
revision used to build it, which may contain changes newer than a published wheel.
For the current checkout:

```bash
python -m pip install -e '.[plot]'
```

## Your first feature–response curve

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

For named Polars columns:

```python
import polars as pl

frame = pl.DataFrame({"temperature": X[:, 0], "demand": y})
model = Regressogram(n_bins=6).fit(
    x="temperature", y="demand", data=frame,
)
print(model.predict(pl.DataFrame({"temperature": [1.0, 2.0, 3.0]})))
```

## Next steps

- [Choose a workflow](user_guide.md) for exploration, summaries, or prediction.
- [Understand the theory and limits](theory.md) before interpreting a curve.
- [Use scikit-learn pipelines](sklearn.md) for preprocessing and evaluation.
- [Look up an API](api/index.md) for exact parameters and results.
- [Run a worked example](examples.md) to plot smoothing choices and subgroup curves.
