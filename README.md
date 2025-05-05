# Rgram

Rgram is a Python library for performing regression analysis and visualisation. It provides tools for creating regression grams (rgrams) and performing kernel smoothing using Polars, a high-performance DataFrame library. The library is designed to simplify data analysis workflows and is compatible with `uv` for dependency management.

## Features

- **Regression Gram (`pl_rgram`)**: Analyse relationships between variables with support for binning by index or distribution and optional Ordinary Least Squares (OLS) regression calculations.
- **Kernel Smoothing (`pl_kernel_smoothing`)**: Perform kernel smoothing using the Epanechnikov kernel for regression analysis.
- **Flexible API**: Designed for ease of use with Polars DataFrames and LazyFrames.

## Requirements

- Python >= 3.11
- `uv` for dependency management
- Polars, NumPy, and other dependencies (managed via `uv`)

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
import numpy as np
from rgram.smoothing import kernel_smoothing

# Generate sample data
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])
x_eval = np.linspace(1, 5, 100)

# Perform kernel smoothing
smoothed_values = kernel_smoothing(x_train, y_train, x_eval)

print(smoothed_values)
```

```
shape: (100, 2)
┌──────────┬──────────┐
│ x_eval   ┆ kernel   │
│ ---      ┆ ---      │
│ f64      ┆ f64      │
╞══════════╪══════════╡
│ 1.0      ┆ 2.0      │
│ 1.040404 ┆ 2.0      │
│ 1.080808 ┆ 2.0143   │
│ 1.121212 ┆ 2.172075 │
│ 1.161616 ┆ 2.30444  │
│ …        ┆ …        │
│ 4.838384 ┆ 9.69556  │
│ 4.878788 ┆ 9.827925 │
│ 4.919192 ┆ 9.9857   │
│ 4.959596 ┆ 10.0     │
│ 5.0      ┆ 10.0     │
└──────────┴──────────┘
```

This example demonstrates how to use the `kernel_smoothing` function to smooth data using the Epanechnikov kernel.

---

### Example: Regressograms with Polars

```python
import polars as pl
from rgram.rgram import rgram

# Create a sample Polars DataFrame
data = {
    "x1": [1, 2, 3, 4, 5],
    "x2": [5, 4, 3, 2, 1],
    "y": [2, 3, 4, 5, 6],
}
df = pl.LazyFrame(data)

# Generate a regression gram
rgram_result = rgram(
    df=df,
    x=["x1", "x2"],
    y="y",
    metric=lambda x: x.mean(),
    hue=None,
    add_ols=True,
    bin_style="width",
)

# Collect the results
result = rgram_result.collect()
print(result)
```

```
shape: (10, 8)
┌─────┬───────┬───────┬───────────┬──────────────┬────────────┬──────┬───────┐
│ y   ┆ x_var ┆ x_val ┆ rgram_bin ┆ y_pred_rgram ┆ y_pred_ols ┆ coef ┆ const │
│ --- ┆ ---   ┆ ---   ┆ ---       ┆ ---          ┆ ---        ┆ ---  ┆ ---   │
│ f64 ┆ str   ┆ i64   ┆ f64       ┆ f64          ┆ f64        ┆ f64  ┆ f64   │
╞═════╪═══════╪═══════╪═══════════╪══════════════╪════════════╪══════╪═══════╡
│ 2.0 ┆ x1    ┆ 1     ┆ 0.0       ┆ 2.5          ┆ 2.0        ┆ 1.0  ┆ 1.0   │
│ 6.0 ┆ x2    ┆ 1     ┆ 0.0       ┆ 5.5          ┆ 6.0        ┆ -1.0 ┆ 7.0   │
│ 3.0 ┆ x1    ┆ 2     ┆ 0.0       ┆ 2.5          ┆ 3.0        ┆ 1.0  ┆ 1.0   │
│ 5.0 ┆ x2    ┆ 2     ┆ 0.0       ┆ 5.5          ┆ 5.0        ┆ -1.0 ┆ 7.0   │
│ 4.0 ┆ x1    ┆ 3     ┆ 1.0       ┆ 4.5          ┆ 4.0        ┆ 1.0  ┆ 1.0   │
│ 4.0 ┆ x2    ┆ 3     ┆ 1.0       ┆ 3.5          ┆ 4.0        ┆ -1.0 ┆ 7.0   │
│ 5.0 ┆ x1    ┆ 4     ┆ 1.0       ┆ 4.5          ┆ 5.0        ┆ 1.0  ┆ 1.0   │
│ 3.0 ┆ x2    ┆ 4     ┆ 1.0       ┆ 3.5          ┆ 3.0        ┆ -1.0 ┆ 7.0   │
│ 6.0 ┆ x1    ┆ 5     ┆ 2.0       ┆ 6.0          ┆ 6.0        ┆ 1.0  ┆ 1.0   │
│ 2.0 ┆ x2    ┆ 5     ┆ 2.0       ┆ 2.0          ┆ 2.0        ┆ -1.0 ┆ 7.0   │
└─────┴───────┴───────┴───────────┴──────────────┴────────────┴──────┴───────┘
```

This example demonstrates how to use the `rgram` function to analyse relationships between variables in a Polars DataFrame or LazyFrame.

---

