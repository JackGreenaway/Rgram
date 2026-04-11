# Rgram

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Rgram** is a high-performance Python library for nonparametric regression analysis and visualisation. It provides tools for creating **regressograms** (binned regression estimators) and performing **kernel smoothing** with the Epanechnikov kernel. Built on top of [Polars](https://pola-rs.github.io/) for rapid data processing, Rgram is designed for exploratory data analysis and statistical visualisation.

> **Theoretical foundation**: Regressograms are discussed in Section 4.4 of García-Portugués, E. (2023). _Notes for nonparametric statistics_. Carlos III University of Madrid.

## Table of Contents

- [Features](#features)
- [When to Use Rgram](#when-to-use-rgram)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Concepts & Architecture](#concepts--architecture)
- [API Reference](#api-reference)
  - [Regressogram](#regressogram)
  - [KernelSmoother](#kernelsmoother)
- [Examples](#examples)
- [Guides](#guides)
  - [Choosing a Binning Strategy](#choosing-a-binning-strategy)
  - [Custom Aggregation Functions](#custom-aggregation-functions)
  - [Data Input Formats](#data-input-formats)
- [Benefits vs Limitations](#benefits-vs-limitations)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Features

- **Regressogram Analysis**: Multiple binning strategies (`dist`, `width`, `none`, `int`) for flexible bin assignment
- **Confidence Intervals**: Customisable confidence interval computation via user-defined aggregation functions
- **Kernel Smoothing**: Epanechnikov kernel smoother with flexible bandwidth selection (`silverman`, `scott`, `manual`)
- **Predictions**: Apply fitted models to new data points via `predict()` method
- **Polars Backend**: High-performance DataFrame operations using lazy evaluation
- **Scikit-learn API**: Familiar `fit()`, `predict()`, and `fit_predict()` methods
- **Array-like or DataFrame Input**: Works seamlessly with Polars DataFrames or NumPy/Python arrays
- **Composable Design**: Clean, focused API allows users to easily compose additional statistical methods

## When to Use Rgram

Rgram is ideal for:

1. **Exploratory Data Analysis (EDA)**: Quickly visualise relationships between variables without assuming a specific functional form
2. **Non-parametric Regression**: When you don't want to assume the underlying relationship is linear or polynomial
3. **Binned Estimation**: When you need interpretable, step-wise predictions (e.g., age-based analysis, price ranges)
4. **Grouped Analysis**: When comparing multiple strata or groups simultaneously
5. **Robust Estimation**: When outliers exist and robust statistics (median, quantiles) are preferred
6. **Semi-parametric workflows**: As a first step before fitting parametric models or validating assumptions

Rgram is **NOT** the best choice for:

- High-dimensional feature spaces (use dimensionality reduction + Rgram, or scikit-learn alternatives)
- Time series with temporal dependencies (use specialised time series libraries)
- Classification tasks (Rgram is for regression only)
- When you need real-time predictions on streaming data (requires refitting)
- When extremely fast inference on massive datasets is critical (though Polars is reasonably fast)

## Requirements

- Python >= 3.9
- [Polars](https://pola-rs.github.io/) >= 1.28.1
- [typing-extensions](https://github.com/python/typing_extensions) >= 4.15.0

### Optional Dependencies (for development)

- `pytest` >= 8.4.2 - Testing framework
- `matplotlib` >= 3.9.4 - Visualisation
- `seaborn` >= 0.13.2 - Statistical graphics
- `scipy` >= 1.13.1 - Statistical functions
- `ruff` >= 0.14.7 - Code linting
- `ipykernel` >= 6.31.0 - Jupyter support

## Installation

### From PyPI (Recommended for Users)

The easiest way to install Rgram is from [PyPI](https://pypi.org/project/rgram/):

**Using pip**:
```bash
pip install rgram
```

**Using UV**:
```bash
uv pip install rgram
```

**Using conda**:
```bash
conda install -c conda-forge rgram
```

Then verify installation:
```bash
python -c "from rgram import Regressogram, KernelSmoother; print('Installation successful!')"
```

### From Source (For Development)

To build and install from source for development:

#### Using UV (Recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package installer and resolver written in Rust. It's the recommended way to work with this project for development.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JackGreenaway/Rgram.git
   cd Rgram
   ```

2. **Install dependencies and the package in development mode**:

   ```bash
   uv sync
   ```

3. **Verify installation**:
   ```bash
   python -c "from rgram import Regressogram, KernelSmoother; print('Installation successful!')"
   ```

#### Using pip

If you prefer standard pip:

```bash
git clone https://github.com/JackGreenaway/Rgram.git
cd Rgram
pip install -e .
```

#### Building the Package

To build a distribution package:

```bash
# Using UV
uv build

# Or using pip/setuptools
python -m build
```

## Quick Start

### Basic Regressogram Example

```python
import polars as pl
import numpy as np
from rgram import Regressogram

# Generate sample data
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)
y = np.sin(x) + np.random.normal(0, 0.5, n)

# Create and fit regressogram
df = pl.DataFrame({"x": x, "y": y})
rgram = Regressogram(binning="dist")
result = rgram.fit_predict(data=df, x="x", y="y")

print(result)
```

### Kernel Smoothing Example

```python
from rgram import KernelSmoother

# Apply kernel smoothing to data
smoother = KernelSmoother(bandwidth="silverman")
smoothed = smoother.fit_predict(data=df, x="x", y="y")

print(smoothed)  # Returns array of predictions
```

## Concepts & Architecture

### Regressogram Overview

A **regressogram** is a non-parametric regression estimator that:

1. Divides the feature space (x-axis) into bins
2. Aggregates target values (y) within each bin using a function (default: mean)
3. Returns the aggregated value for all points in that bin

**Key advantages**: Simple, interpretable, and computationally efficient
**Trade-off**: Creates step-wise predictions (discontinuous at bin boundaries)

### Binning Strategies

Rgram supports four binning methods:

| Strategy           | Method                                    | Use Case                             | Output                       |
| ------------------ | ----------------------------------------- | ------------------------------------ | ---------------------------- |
| `"dist"` (Default) | Distribution-based with Scott's bandwidth | General purpose, data-driven         | Fewer bins in sparse regions |
| `"width"`          | Fixed bin width from data range           | When consistent bin sizes matter     | Equal-width bins             |
| `"int"`            | Integer bin assignment                    | When x values are naturally discrete | Integer-indexed bins         |
| `"none"`           | Uses x values as unique bins              | Per-unique-value statistics          | One bin per unique x value   |

### Kernel Smoother Overview

A **kernel smoother** applies the Epanechnikov kernel to smooth predictions:

1. Defines evaluation points across the x-axis
2. For each point, computes a weighted average of nearby y values
3. Weights decay with distance from the evaluation point

**Key advantages**: Smooth predictions, continuous derivatives
**Trade-off**: More computationally expensive than regressograms

### Bandwidth Selection

The `KernelSmoother` supports three bandwidth selection methods:

1. **Silverman's Rule** (default) - Robust and data-adaptive:
   $$h = 0.9 \min(\sigma, IQR/1.34) \cdot n^{-1/5}$$
   Best for most use cases; automatically adapts to data spread

2. **Scott's Rule** - Simpler and less sensitive to outliers:
   $$h = 1.06 \cdot \sigma \cdot n^{-1/5}$$
   Good for normally distributed data

3. **Manual Specification** - Full control for expert users
   Specify exact bandwidth value for fine-tuned smoothness control

Each method balances bias and variance differently. Experiment with `bandwidth` parameter to find optimal smoothing for your data.

### Data Flow Architecture

```
Input Data (arrays/DataFrame)
    ↓
[_prepare_data] - Normalize to LazyFrame with named columns
    ↓
[fit] - Learn parameters (bins, bandwidth, etc.)
    ↓
[transform] - Apply learned mapping to all rows
    ↓
Output LazyFrame (collect() to materialise results)
```

Key design patterns:

- **Lazy evaluation**: Polars LazyFrames enable optimisation and memory efficiency
- **Fit-predict separation**: Learn on one dataset, apply to another (e.g., train/test split)
- **Composable**: Chain output of one tool into input of another

### Regressogram

The `Regressogram` class performs binned regression on one or more features and targets with customisable aggregation and optional confidence intervals.

#### Parameters

```python
Regressogram(
    binning: Literal["dist", "width", "none", "int"] = "dist",
    agg: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
    ci: Optional[tuple[Callable, Callable]] = (lambda x: x.mean() - x.std(), lambda x: x.mean() + x.std()),
    n_bins: Optional[int] = None,
)
```

| Parameter | Type               | Default                | Description                                                                                                                           |
| --------- | ------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `binning` | str                | `"dist"`               | Binning strategy. Options: `"dist"` (distribution-based), `"width"` (fixed width), `"none"` (unique x values), `"int"` (integer bins) |
| `agg`     | callable           | `lambda x: x.mean()`   | Aggregation function to apply to y values within each bin. Must accept and return a Polars expression                                 |
| `ci`      | tuple of callables | `(mean-std, mean+std)` | Tuple of functions for lower and upper confidence limit calculations. Set to `None` to disable                                        |
| `n_bins`  | int or None        | `None`                 | Number of bins for `"dist"` binning. If None, automatically calculated using Freedman-Diaconis rule. Ignored for other strategies     |

#### Methods

**`fit(x, y, data=None) -> Regressogram`**

Fit the regressogram to data.

- **x**: Column name(s) if `data` provided, else array-like
- **y**: Column name(s) if `data` provided, else array-like
- **data**: `pl.DataFrame`, `pl.LazyFrame`, or `None`. If `None`, x/y treated as arrays

Returns: self (fitted estimator)

**`fit_predict(x, y, data=None, return_ci=False) -> np.ndarray or tuple`**

Fit and return predictions in one call.

- Returns `np.ndarray` of predictions by default
- Returns `(y_pred, y_ci_low, y_ci_high)` tuple when `return_ci=True`
- When using DataFrame mode, predictions are at the training x values
- When using array mode, predictions are at the provided x values

**`predict(x: Union[Sequence[float], pl.Series]) -> np.ndarray`**

Make predictions on new x values using the fitted binning scheme.

- **x**: Array-like or `pl.Series` of new x points to predict
- Must call `fit()` before `predict()`

Returns: NumPy array of predicted values based on learned bins

---

### KernelSmoother

The `KernelSmoother` class performs kernel smoothing using the Epanechnikov kernel with flexible bandwidth selection.

#### Parameters

```python
KernelSmoother(
    bandwidth: Literal['silverman', 'scott', 'manual'] = 'silverman',
    bandwidth_value: Optional[float] = None,
    bandwidth_adjust: float = 1.0,
    n_eval_samples: int = 100
)
```

| Parameter          | Type  | Default       | Description                                                         |
| ------------------ | ----- | ------------- | ------------------------------------------------------------------- |
| `bandwidth`        | str   | `'silverman'` | Bandwidth selection method: `'silverman'`, `'scott'`, or `'manual'` |
| `bandwidth_value`  | float | `None`        | Manual bandwidth value. Required if `bandwidth='manual'`            |
| `bandwidth_adjust` | float | `1.0`         | Multiplicative adjustment factor for the calculated bandwidth       |

**Bandwidth Methods:**

- **`'silverman'`** (default): 0.9 × min(std, IQR/1.34) × n^(-1/5) - Robust, adapts to data spread
- **`'scott'`**: 1.06 × std × n^(-1/5) - Simpler, less sensitive to outliers
- **`'manual'`**: User specifies exact bandwidth value for fine-tuned control

#### Methods

**`fit(x, y, data=None) -> KernelSmoother`**

Fit the kernel smoother to data using the selected bandwidth method.

- **x**: Column name if `data` provided, else array-like (must be univariate)
- **y**: Column name if `data` provided, else array-like (must be univariate)
- **data**: `pl.DataFrame`, `pl.LazyFrame`, or `None`

Returns: self (fitted estimator)

**`fit_predict(x, y, data=None, x_eval=None, return_ci=False) -> np.ndarray or tuple`**

Fit and return predictions in one step.

- Returns `np.ndarray` of predictions by default
- **x_eval**: Optional array-like of x values for predictions. If `None`, predictions are at training x values
- Predictions use the fitted bandwidth determined during `fit()`
- `return_ci=True` currently returns `(y_pred, None, None)` (CIs not yet implemented for kernel smoother)

**`predict(x_eval: Union[Sequence[float], pl.Series], return_ci=False) -> np.ndarray or tuple`**

Apply the fitted smoother to new x values without refitting. Uses the bandwidth value determined during `fit()`.

- **x_eval**: Array-like or `pl.Series` of x points for predictions
- Must call `fit()` before `predict()`

Returns: NumPy array of smoothed predictions or tuple with optional confidence intervals

---

## Examples

### Example 1: Complete Regressogram Workflow with Visualisation

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from rgram import Regressogram

# Generate synthetic data
np.random.seed(42)
n = 150
x = np.linspace(0, 10, n)
y_true = np.sin(x)
y_noisy = y_true + np.random.normal(0, 0.6, n)

# Create DataFrame
df = pl.DataFrame({"x": x, "y_true": y_true, "y_noisy": y_noisy})

# Fit regressogram with different binning strategies
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, binning in zip(axes.flat, ["dist", "width", "int", "none"]):
    rgram = Regressogram(
        binning=binning,
        ci=(lambda x: x.mean() - 1.96 * x.std(), lambda x: x.mean() + 1.96 * x.std())
    )

    # Compute predictions at training points
    y_pred, y_ci_low, y_ci_high = rgram.fit_predict(data=df, x="x", y="y_noisy", return_ci=True)

    # Sort by x for proper plotting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    ax.scatter(x, y_noisy, alpha=0.4, s=20, label="observations")
    ax.plot(x, y_true, "g-", linewidth=2, label="true function")
    ax.step(x_sorted, y_pred_sorted, "r-", linewidth=2, where="post", label="rgram")

    if y_ci_low is not None and y_ci_high is not None:
        y_ci_low_sorted = y_ci_low[sort_idx]
        y_ci_high_sorted = y_ci_high[sort_idx]
        ax.fill_between(
            x_sorted,
            y_ci_low_sorted,
            y_ci_high_sorted,
            alpha=0.2,
            color="red",
            label="95% CI"
        )

    ax.set_title(f"Binning: {binning}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Example 2: Combining Regressogram with Kernel Smoothing

```python
from rgram import Regressogram, KernelSmoother
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x) * np.exp(-x / 5) + np.random.normal(0, 0.3, 200)

df = pl.DataFrame({"x": x, "y": y})

# Step 1: Fit regressogram and get predictions
rgram = Regressogram(binning="dist", ci=None)  # No CI for clarity
rgram_preds = rgram.fit_predict(data=df, x="x", y="y")

# Step 2: Smooth the regressogram predictions
# Get the binned predictions via fit_predict with array inputs
rgram_x = np.array([25, 30, 35, 40])  # Example bin centers
rgram_y = np.array([30, 35, 40, 35])  # Example bin predictions

smoother = KernelSmoother(bandwidth="silverman")
smoothed_y = smoother.fit_predict(x=rgram_x, y=rgram_y)

# Visualisation
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x, y, alpha=0.3, s=20, label="Raw observations")
ax.step(rgram_x, rgram_y,
        where="post", linewidth=2, label="Regressogram predictions")
ax.plot(np.sort(rgram_x), smoothed_y[np.argsort(rgram_x)],
        linewidth=2.5, color="green", label="Kernel smoothed")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 3: Custom Aggregation Functions

```python
from rgram import Regressogram
import polars as pl
import numpy as np

# Generate data
np.random.seed(42)
x = np.linspace(0, 10, 150)
y = np.sin(x) + np.random.normal(0, 0.5, 150)

df = pl.DataFrame({"x": x, "y": y})

# Use median instead of mean
rgram_median = Regressogram(
    binning="dist",
    agg=lambda x: x.median(),
    ci=(
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    )
)
result = rgram_median.fit_predict(data=df, x="x", y="y", return_ci=True)

print("Regressogram with median aggregation:")
if isinstance(result, tuple):
    y_pred, y_ci_low, y_ci_high = result
    print(f"Predictions shape: {y_pred.shape}")
else:
    print(f"Predictions: {result}")
```

## Guides

### Choosing a Binning Strategy

**Distribution-based binning (`"dist"`)** ← Default choice for most cases

- Uses Scott's bandwidth rule to adapt bin width to data density
- Fewer, wider bins where data is sparse; more bins where data is dense
- **Handles duplicates**: Robust to duplicate x values; uses qcut with `allow_duplicates=True` internally
- **Best for**: Normal or near-normal distributions, data exploration, datasets with duplicate values
- **Example**: Customer age analysis with uneven age distribution

```python
rgram = Regressogram(binning="dist")
# Automatically creates wider bins for underrepresented ages
# Handles duplicate ages gracefully
result = rgram.fit_predict(data=df, x="age", y="purchase_amount")
print(result)  # Array of predictions
```

**Fixed-width binning (`"width"`)**

- Creates equal-sized bins across the entire range
- Consistent interpretation across bins
- **Best for**: When bin boundaries have business meaning (e.g., income brackets: $0-50K, $50-100K, etc.)
- **Drawback**: May have very sparse bins at data extremes

```python
rgram = Regressogram(binning="width")
# Creates consistent income brackets, though some may be nearly empty
result = rgram.fit_predict(data=df, x="annual_income", y="credit_score")
print(result)  # Array of predictions
```

**Integer binning (`"int"`)**

- Casts x values to integers and groups by integer value
- **Best for**: Truly discrete integer data (number of items, years of experience)
- **Example**: Product rating (1-5 stars) vs review count

```python
rgram = Regressogram(binning="int")
result = rgram.fit_predict(data=df, x="product_rating", y="num_reviews")
print(result)  # Array of predictions
```

**No binning / Unique values (`"none"`)**

- Treats each unique x value as its own independent bin
- **Best for**: Computing statistics at each unique x value without binning across x values
- **Use case**: When x is already categorical or when you want predictions for each exact x value

```python
rgram = Regressogram(binning="none", agg=lambda x: x.mean())
result = rgram.fit_predict(data=df, x="x_col", y="target")
print(result)  # Array with one prediction per unique x value
```

### Controlling Bin Count for Distribution-Based Binning

By default, `"dist"` binning uses the **Freedman-Diaconis rule** to automatically determine the number of bins based on data distribution. You can override this with the `n_bins` parameter:

```python
from rgram import Regressogram

# Automatic (default) - Freedman-Diaconis rule
rgram_auto = Regressogram(binning="dist")

# Manual control - specify exact number of bins
rgram_5_bins = Regressogram(binning="dist", n_bins=5)
rgram_20_bins = Regressogram(binning="dist", n_bins=20)

# Fit and compare
result_auto = rgram_auto.fit_predict(data=df, x="x", y="y")
result_5 = rgram_5_bins.fit_predict(data=df, x="x", y="y")
result_20 = rgram_20_bins.fit_predict(data=df, x="x", y="y")

# Fewer bins (5) = smoother, coarser estimate
# More bins (20) = more detailed, but noisier estimate
```

**When to set custom `n_bins`**:

- You know optimal bin count from domain knowledge
- You want coarser or finer granularity than automatic selection provides
- The `n_bins` parameter is ignored for `"width"`, `"int"`, and `"none"` binning strategies

### Custom Aggregation Functions

By default, Regressogram uses the mean, but you can specify any Polars aggregation:

```python
import polars as pl
from rgram import Regressogram

# Median (robust to outliers)
rgram_median = Regressogram(agg=lambda x: x.median())

# Count of observations per bin
rgram_count = Regressogram(
    agg=lambda x: pl.len(),
    ci=None  # Disable confidence intervals for count
)

# Standard deviation
rgram_std = Regressogram(agg=lambda x: x.std())

result = rgram_count.fit_predict(data=df, x="x", y="dummy_col")
print(result)  # Array of bin counts
```

### Data Input Formats

Rgram follows a **seaborn-like API** where you can use either:

**Pattern 1: DataFrame + Column Names** (Best for large data and reusable workflows)

```python
import polars as pl
from rgram import Regressogram

df = pl.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "salary": [50000, 55000, 60000, 70000, 80000]
})

# Reference columns by name (like seaborn.kdeplot)
rgram = Regressogram()
result = rgram.fit_predict(data=df, x="age", y="salary")
print(result)  # Array of predictions
```

**Pattern 2: Raw Arrays/Series** (Best for quick analysis, interactive work)

```python
import numpy as np
from rgram import Regressogram

x = np.array([25, 30, 35, 40, 45])
y = np.array([50000, 55000, 60000, 70000, 80000])

# Pass arrays directly without a DataFrame (like seaborn.kdeplot with just x=)
rgram = Regressogram()
result = rgram.fit_predict(x=x, y=y)
print(result)  # Array of predictions
```

**Pattern 3: Mixed with Polars Series**

```python
import polars as pl
from rgram import Regressogram

df = pl.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "salary": [50000, 55000, 60000, 70000, 80000]
})

# Use Series directly without wrapping in DataFrame
result = rgram.fit_predict(x=df["age"], y=df["salary"])
print(result)  # Array of predictions
```

**Pattern 4: Multiple features/targets**

```python
# Multiple x columns (analysed as separate x-y pair combinations)
result = rgram.fit_predict(
    data=df,
    x=["age", "experience"],
    y="salary"
)
print(result)  # Array of predictions across x-y feature pairs
```

**When to use which pattern:**

- **DataFrame + names**: Production code, complex pipelines
- **Raw arrays**: Quick exploration, notebooks, when data is already in memory
- **Series**: Intermediate between the two; good for simple scripts

## Benefits vs Limitations

### Advantages of Regressogram

| Benefit              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **Interpretability** | Step-wise predictions are easy to explain to stakeholders ("if age 25-35, avg salary is X") |
| **Robustness**       | Can use median or quantiles instead of mean for outlier-resistant estimates                 |
| **Flexibility**      | Custom aggregation functions support domain-specific logic (e.g., weighted means)           |
| **Speed**            | Binning is computationally efficient; results scale well with data size                     |
| **No assumptions**   | Non-parametric; doesn't assume linearity, polynomials, or other functional forms            |

### Advantages of Kernel Smoother

| Benefit                  | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| **Smoothness**           | Produces continuous predictions without step discontinuities |
| **Local relationships**  | Captures local patterns via adaptive weighting               |
| **Derivative existence** | Smooth function enables gradient-based analysis              |
| **Visual appeal**        | Creates professional-looking curves for plots                |
| **Flexible composition** | Can be chained after regressogram for two-stage smoothing    |

### Limitations Summary

| Limitation                    | Impact                                           | Workaround                                                                    |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------- |
| **1D only**                   | Cannot handle high dimensions directly           | Use dimensionality reduction or analyse features independently                |
| **No feature selection**      | All x variables are used                         | Pre-select relevant features based on domain knowledge                        |
| **Binning creates artifacts** | Regressogram has artificial step discontinuities | Use KernelSmoother after Regressogram, or use Regressogram only for EDA       |
| **Bandwidth sensitivity**     | Kernel results vary with bandwidth choice        | Silverman's rule is automatic; use cross-validation for critical applications |
| **Memory for large data**     | Lazy evaluation has limits during `.collect()`   | Process data in batches; use Polars partitioning                              |
| **No missing value handling** | NaN values cause errors                          | Impute or remove missing values before fitting                                |
| **No real-time predictions**  | Must refit to add new data                       | Refitting is fast enough for small->medium datasets                           |
| **Categorical inputs**        | X and Y must be numeric                          | Encode categorical variables (ordinal encoding or one-hot + aggregation)      |

## Limitations

- **Univariate Kernel Smoothing**: `KernelSmoother` currently only supports single-variable smoothing. Multivariate kernel smoothing is not yet implemented.

- **Bandwidth Selection**: Kernel smoothing offers three methods (Silverman, Scott, Manual) but selection remains user-driven; automatic cross-validation is not yet implemented.

- **Binning Strategy Selection**: The choice of binning strategy can significantly impact results. The library provides multiple strategies but does not automatically select the optimal one. Users should experiment or use cross-validation.

- **Memory Efficiency**: For very large datasets, even with Polars' optimisations, lazy evaluation may be limited by system memory during collection.

- **Multi-dimensional Input**: Regressogram and KernelSmoother are designed for 1D→1D mappings. Multi-dimensional feature spaces require feature engineering or multiple univariate analyses.

- **Missing Values**: The current implementation does not explicitly handle missing values. Pre-processing with appropriate techniques (imputation, removal) is required.

- **Categorical Features**: Both classes require numerical input. Categorical variables must be encoded numerically before use.

## Future Improvements

### Completed

- [x] **Bandwidth Selection**: Silverman, Scott, and manual bandwidth methods for `KernelSmoother` (implemented in v0.2+)
- [x] **Robust Duplicate Handling**: Distribution-based binning now handles duplicate x values via `qcut(..., allow_duplicates=True)` (implemented in v0.2+)

### High Priority

- [ ] **Multivariate Kernel Smoothing**: Extend `KernelSmoother` to support multi-dimensional input with optimal bandwidth selection for each dimension
- [ ] **Cross-Validated Bandwidth Selection**: Automatic bandwidth tuning via leave-one-out or k-fold cross-validation
- [ ] **Missing Data Handling**: Built-in support for various imputation strategies and missing value indicators
- [ ] **Auto Binning Strategy Selection**: Data-driven method to select optimal binning strategy using cross-validation

### Medium Priority

- [ ] **Adaptive Binning**: Implement data-driven bin size selection using information-theoretic criteria
- [ ] **Confidence Band Methods**: Additional methods for computing confidence bands (e.g., bootstrap, Bayesian)
- [ ] **Plotting Utilities**: High-level visualisation functions with matplotlib/plotly backends
- [ ] **Performance Profiling**: Detailed benchmarks and optimisation for large-scale datasets

### Lower Priority

- [ ] **Additional Kernels**: Support for Gaussian, Triangular, and other kernel types
- [ ] **GPU Acceleration**: Polars GPU backend support
- [ ] **Advanced Statistics**: Additional statistical methods (local polynomial regression, etc.)

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies: `uv sync`
4. Make your changes and add tests
5. Run tests: `pytest`
6. Lint code: `ruff check .`
7. Commit: `git commit -am 'Add your feature'`
8. Push: `git push origin feature/your-feature`
9. Create a Pull Request

### Development Workflow with UV

```bash
# Install all dependencies including dev
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .

# Start interactive shell
uv run ipython
```

## References

- García-Portugués, E. (2023). _Notes for nonparametric statistics_. Carlos III University of Madrid. [Available online](https://egarpor.github.io/notes-nps/)
- Polars Documentation: https://pola-rs.github.io/
- Silverman, B. W. (1986). _Density Estimation for Statistics and Data Analysis_. Chapman and Hall/CRC: New York.
- Wand, M. P., & Jones, M. C. (1994). _Kernel Smoothing_. Chapman and Hall/CRC: New York.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Acknowledgments**: This library was inspired by nonparametric regression techniques in statistical computing and is built on the excellent [Polars](https://pola-rs.github.io/) data manipulation library.
