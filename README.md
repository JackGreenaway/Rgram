# Rgram

Rgram is a Python library for performing regression analysis and visualisation. It provides tools for creating regressograms (rgrams) and performing kernel smoothing using Polars, a high-performance DataFrame library. The library is designed to simplify data analysis workflows and is compatible with `uv` for dependency management.

Notes on regressograms can be found in section 4.4 of `García-Portugués, E. (2023). Notes for nonparametric statistics. Carlos III University of Madrid: Madrid, Spain.`

## Features

- **Regressogram (`rgram`)**: Analyse relationships between variables with support for binning by index or distribution and optional Ordinary Least Squares (OLS) regression calculations.
- **Kernel Smoothing (`kernel_smoothing`)**: Perform kernel smoothing using the Epanechnikov kernel for regression analysis.
- **Flexible API**: Designed for ease of use and high performance thanks to Polars DataFrames and LazyFrames.

## Requirements

- Python >= 3.11
- `uv` for dependency management

## Installation

To get started with Rgram, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/JackGreenaway/Rgram.git
   cd Rgram
   ```

2. Install dependencies using `uv`:
   ```bash
   uv install
   ```

3. Verify the installation:
   ```bash
   uv sync
   ```

## Usage

---

### Example: Regressograms and Kernel Smoothing with Polars

> Imports and setup
```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from rgram.rgram import Regressogram, KernelSmoother

plt.style.use("ggplot")
```

> Generate sample data
```
n = 50
x = np.sort(np.random.normal(0, 1, n))
y = 1 + x
y_noise = y + np.random.normal(0, 2, n)
```

<div align="center">
  <img src="examples/base_function.png" alt="base function">
</div>

> Fit regressogram to noisy data
```python
df = pl.DataFrame({"x": x, "y": y, "y_noise": y_noise})

rgramer = Regressogram(data=df, x="x", y="y_noise")
rgram = rgramer.fit_transform().collect()

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, y, lw=0.5, label="true function")
ax.scatter(x, y_noise, s=15, alpha=0.3, marker="o", color="black")
ax.step(rgram["x_val"], rgram["y_pred_rgram"], lw=0.5, label="rgram")
ax.fill_between(
    rgram["x_val"],
    rgram["y_pred_rgram_uci"],
    rgram["y_pred_rgram_lci"],
    alpha=0.2,
    label="ci",
)

ax.set_xlabel("x variable"), ax.set_ylabel("y variable")
ax.legend()
fig.tight_layout()
plt.show()
```
<div align="center">
  <img src="examples/rgram.png" alt="rgram">
</div>

> Kernel smoothing on regressogram output
```python
smoother = KernelSmoother(data=rgram, x="x_val", y="y_pred_rgram")
ks_rgram = smoother.fit_transform().collect()

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, y, lw=0.5, label="true function")
ax.scatter(x, y_noise, s=15, alpha=0.3, marker="o", color="black")
ax.plot(ks_rgram["x_eval"], ks_rgram["y_kernel"], lw=0.5, label="smoothed rgram")

cis = []
for col in ["y_pred_rgram_lci", "y_pred_rgram_uci"]:
    smoother = KernelSmoother(data=rgram, x="x_val", y=col)
    cis += [smoother.fit_transform().collect()]

ax.fill_between(
    cis[0]["x_eval"],
    cis[0]["y_kernel"],
    cis[1]["y_kernel"],
    alpha=0.2,
    label="smoothed ci",
)

ax.set_xlabel("x variable"), ax.set_ylabel("y variable")
ax.legend()
fig.tight_layout()
plt.show()
```

<div align="center">
  <img src="examples/smoothed_rgram.png" alt="smoothed rgram">
</div>

This example demonstrates how to use the `Regressogram` and `KernelSmoother` classes to create a regressogram and apply kernel smoothing for visualisation.  
Both classes follow a scikit-learn-like API with `fit()`, `transform()`, and `fit_transform()` methods.  
For most use cases, `fit_transform()` is the recommended entry point.

---

