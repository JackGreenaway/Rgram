# Rgram

Rgram helps analysts explore how one numeric response varies with one numeric
feature. It provides interpretable binned summaries (`Regressogram`), kernel
smoothing (`KernelSmoother`), support diagnostics, bootstrap intervals, and
coverage-aware parameter selection. Estimators work with scikit-learn pipelines
and selected NumPy, pandas, or Polars columns.

## Installation

Python 3.9 or newer is required; NumPy, Polars, and scikit-learn are dependencies.

```bash
python -m pip install rgram
# Optional plotting and pandas support:
python -m pip install 'rgram[plot,pandas]'
```

For the current source revision, use `python -m pip install -e '.[plot,pandas]'`
from this checkout. Published packages may lag this documentation.

## Quick start

Summarize a response in two feature intervals:

```python
from rgram import Regressogram

model = Regressogram(n_bins=2).fit([0., 1., 2., 3.], [1., 3., 5., 7.])
print(model.predict([0.5, 2.5]))
```

```text
[2. 6.]
```

Each model accepts one feature and one target. Curves describe associations;
they do not adjust for other features or establish causality. Missing and
non-finite training values are rejected. Inspect support before interpreting
predictions and use held-out evaluation for predictive claims. The API is
unstable: pin a version for reproducible analyses.

Read the [full documentation](https://jackgreenaway.github.io/Rgram/):
[installation](docs/getting_started.md), [user guide](docs/user_guide.md),
[examples](docs/examples.md), [API reference](docs/api/index.md), and
[development and local preview](docs/development.md).
