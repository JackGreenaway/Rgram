# Rgram

Rgram is a Python library for performing regression analysis and visualisation. It provides tools for creating regression grams (rgrams) and performing kernel smoothing using Polars, a high-performance DataFrame library. The library is designed to simplify data analysis workflows and is compatible with `uv` for dependency management.

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
   uv check
   ```

## Usage

### Example: Kernel Smoothing with Polars

```python
import polars as pl
import numpy as np
from rgram.smoothing import kernel_smoothing

# Generate sample data
n = 100
x = np.sort(np.random.normal(0, 1, n))
y = 1 + x
y_noise = y + np.random.normal(0, np.sqrt(2), n)

df = pl.DataFrame({"x": x, "y": y, "y_noise": y_noise})

# Perform kernel smoothing
kernel_smoothed = kernel_smoothing(df=df, x="x", y="y_noise", n_eval_samples=500)

print(kernel_smoothed)
```

```
shape: (474, 2)
┌───────────┬───────────┐
│ x_eval    ┆ kernel    │
│ ---       ┆ ---       │
│ f64       ┆ f64       │
╞═══════════╪═══════════╡
│ -2.951255 ┆ -2.412448 │
│ -2.940726 ┆ -2.412448 │
│ -2.930197 ┆ -2.412448 │
│ -2.919668 ┆ -2.412448 │
│ -2.90914  ┆ -2.412448 │
│ …         ┆ …         │
│ 2.260485  ┆ 2.101231  │
│ 2.271013  ┆ 2.103863  │
│ 2.281542  ┆ 2.106533  │
│ 2.292071  ┆ 2.109262  │
│ 2.3026    ┆ 2.112069  │
└───────────┴───────────┘
```

This example demonstrates how to use the `kernel_smoothing` function to smooth data using the Epanechnikov kernel.

---

### Example: Regressograms with Polars

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from rgram.rgram import rgram
from rgram.smoothing import kernel_smoothing

# Generate sample data
n = 100
x = np.sort(np.random.normal(0, 1, n))
y = 1 + x
y_noise = y + np.random.normal(0, np.sqrt(2), n)

df = pl.DataFrame({"x": x, "y": y, "y_noise": y_noise})

# Apply regression histogram with quantile binning
regressogram = rgram(
    df=df, x=["x"], y=["y_noise"], bin_style="dist", allow_negative_y="auto"
)

fig, ax = plt.subplots(figsize=(5, 5))

# Plot the regressogram
ax.plot(x, y, label="true function", color="black", lw=0.5)
ax.scatter(x, y_noise, s=15, alpha=0.3, marker="o", color="black")

ax.step(
    regressogram["x_val"],
    regressogram["y_pred_rgram"],
    label="regressogram",
    where="mid",
    lw=0.5,
)

# Perform kernel smoothing on the regressogram
kernel_smoothed = kernel_smoothing(
    df=regressogram, x="x_val", y="y_pred_rgram", hue=["x_var", "y_var"]
)

ax.plot(
    kernel_smoothed["x_eval"],
    kernel_smoothed["kernel"],
    label="kernel smoothing",
    lw=0.5,
)

# Add confidence intervals
kernel_smoothed_ci = kernel_smoothing(
    df=regressogram.unpivot(
        on=["y_pred_rgram_lci", "y_pred_rgram_uci"],
        index=["x_var", "y_var", "x_val"],
        variable_name="ci",
        value_name="y_pred_rgram_ci",
    ),
    x="x_val",
    y="y_pred_rgram_ci",
    hue=["x_var", "y_var", "ci"],
)

ax.fill_between(
    x=kernel_smoothed_ci.filter(pl.col("ci") == "y_pred_rgram_uci")["x_eval"],
    y1=kernel_smoothed_ci.filter(pl.col("ci") == "y_pred_rgram_lci")["kernel"],
    y2=kernel_smoothed_ci.filter(pl.col("ci") == "y_pred_rgram_uci")["kernel"],
    alpha=0.2,
)

plt.legend()
plt.tight_layout()
plt.show()
```

<p align="center">
  <img src="example.png">
</p>

This example demonstrates how to use the `rgram` function to create a regressogram and apply kernel smoothing for visualisation.

---

