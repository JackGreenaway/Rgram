# Rgram

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Rgram** is a high-performance Python library for nonparametric regression analysis and visualization. It provides tools for creating **regressograms** (binned regression estimators) and performing **kernel smoothing** with the Epanechnikov kernel. Built on top of [Polars](https://pola-rs.github.io/) for rapid data processing, Rgram is designed for exploratory data analysis and statistical visualization.

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
  - [Working with Groups (Hue)](#working-with-groups-hue)
  - [Data Input Formats](#data-input-formats)
- [Benefits vs Limitations](#benefits-vs-limitations)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Features

- **Regressogram Analysis**: Multiple binning strategies (`dist`, `width`, `unique`, `int`) for flexible bin assignment
- **Confidence Intervals**: Customizable confidence interval computation via user-defined aggregation functions
- **Kernel Smoothing**: Epanechnikov kernel smoother with automatic bandwidth selection (Silverman's rule of thumb)
- **Grouped Analysis**: Support for grouping variables (hue) to analyze multiple subgroups simultaneously
- **Polars Backend**: High-performance DataFrame operations using lazy evaluation
- **Scikit-learn API**: Familiar `fit()`, `transform()`, and `fit_transform()` methods
- **Array-like or DataFrame Input**: Works seamlessly with Polars DataFrames or NumPy/Python arrays
- **Composable Design**: Clean, focused API allows users to easily compose additional statistical methods

## When to Use Rgram

Rgram is ideal for:

1. **Exploratory Data Analysis (EDA)**: Quickly visualize relationships between variables without assuming a specific functional form
2. **Non-parametric Regression**: When you don't want to assume the underlying relationship is linear or polynomial
3. **Binned Estimation**: When you need interpretable, step-wise predictions (e.g., age-based analysis, price ranges)
4. **Grouped Analysis**: When comparing multiple strata or groups simultaneously
5. **Robust Estimation**: When outliers exist and robust statistics (median, quantiles) are preferred
6. **Semi-parametric workflows**: As a first step before fitting parametric models or validating assumptions

Rgram is **NOT** the best choice for:

- High-dimensional feature spaces (use dimensionality reduction + Rgram, or scikit-learn alternatives)
- Time series with temporal dependencies (use specialized time series libraries)
- Classification tasks (Rgram is for regression only)
- When you need real-time predictions on streaming data (requires refitting)
- When extremely fast inference on massive datasets is critical (though Polars is reasonably fast)

## Requirements

- Python >= 3.9
- [Polars](https://pola-rs.github.io/) >= 1.28.1
- [polars-ols](https://github.com/cversteeg/polars-ols) >= 0.3.5
- [typing-extensions](https://github.com/python/typing_extensions) >= 4.15.0

### Optional Dependencies (for development)

- `pytest` >= 8.4.2 - Testing framework
- `matplotlib` >= 3.9.4 - Visualization
- `seaborn` >= 0.13.2 - Statistical graphics
- `scipy` >= 1.13.1 - Statistical functions
- `ruff` >= 0.14.7 - Code linting
- `ipykernel` >= 6.31.0 - Jupyter support

## Installation

### Using UV (Recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package installer and resolver written in Rust. It's the recommended way to work with this project.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JackGreenaway/Rgram.git
   cd Rgram
   ```

2. **Install with UV**:

   ```bash
   uv sync
   ```

3. **Install the package in development mode**:

   ```bash
   uv pip install -e .
   ```

4. **Verify installation**:
   ```bash
   python -c "from rgram import Regressogram, KernelSmoother; print('Installation successful!')"
   ```

### Using pip

If you prefer using standard pip:

```bash
git clone https://github.com/JackGreenaway/Rgram.git
cd Rgram
pip install -e .
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
result = rgram.fit_transform(data=df, x="x", y="y").collect()

print(result.head())
```

### Kernel Smoothing Example

```python
from rgram import KernelSmoother

# Apply kernel smoothing to regressogram output
smoother = KernelSmoother(n_eval_samples=100)
smoothed = smoother.fit_transform(data=result, x="x_val", y="y_pred_rgram").collect()

print(smoothed.head())
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
| `"unique"`         | Uses x values as unique bins              | Per-unique-value statistics          | One bin per unique x value   |

### Kernel Smoother Overview

A **kernel smoother** applies the Epanechnikov kernel to smooth predictions:

1. Defines evaluation points across the x-axis
2. For each point, computes a weighted average of nearby y values
3. Weights decay with distance from the evaluation point

**Key advantages**: Smooth predictions, continuous derivatives
**Trade-off**: More computationally expensive than regressograms

### Bandwidth Selection

Kernel Smoother uses **Silverman's rule of thumb** for automatic bandwidth:
$$h = 0.9 \min(\sigma, IQR/1.34) \cdot n^{-1/5}$$

This balances bias and variance automatically but may need adjustment for highly skewed data.

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
Output LazyFrame (collect() to materialize results)
```

Key design patterns:

- **Lazy evaluation**: Polars LazyFrames enable optimization and memory efficiency
- **Fit-predict separation**: Learn on one dataset, apply to another (e.g., train/test split)
- **Composable**: Chain output of one tool into input of another

### Regressogram

The `Regressogram` class performs binned regression on one or more features and targets with customizable aggregation and optional confidence intervals.

#### Parameters

```python
Regressogram(
    binning: Literal["dist", "width", "all", "int"] = "dist",
    agg: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
    ci: Optional[tuple[Callable, Callable]] = (lambda x: x.mean() - x.std(), lambda x: x.mean() + x.std()),
    allow_negative_y: Union[bool, Literal["auto"]] = "auto",
)
```

| Parameter          | Type               | Default                | Description                                                                                                                             |
| ------------------ | ------------------ | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `binning`          | str                | `"dist"`               | Binning strategy. Options: `"dist"` (distribution-based), `"width"` (fixed width), `"unique"` (unique x values), `"int"` (integer bins) |
| `agg`              | callable           | `lambda x: x.mean()`   | Aggregation function to apply to y values within each bin. Must accept and return a Polars expression                                   |
| `ci`               | tuple of callables | `(mean-std, mean+std)` | Tuple of functions for lower and upper confidence limit calculations. Set to `None` to disable                                          |
| `allow_negative_y` | bool or "auto"     | `"auto"`               | Whether to apply clipping to maintain non-negative y values. `"auto"` automatically detects based on input data                         |

#### Methods

**`fit(x, y, data=None, hue=None, keys=None) -> Regressogram`**

Fit the regressogram to data.

- **x**: Column name(s) if `data` provided, else array-like
- **y**: Column name(s) if `data` provided, else array-like
- **data**: `pl.DataFrame`, `pl.LazyFrame`, or `None`. If `None`, x/y treated as arrays
- **hue**: Optional grouping variable(s) for stratified analysis
- **keys**: Optional additional grouping columns

Returns: self (fitted estimator)

**`transform() -> pl.LazyFrame`**

Returns the fitted regressogram results.

Output columns:

- `x_val`: Binned x values
- `y_pred_rgram`: Aggregated y predictions (based on `agg` function)
- `y_pred_rgram_lci`, `y_pred_rgram_uci`: Confidence interval bounds (if `ci` provided)
- `y_val`: Original y values

**`fit_transform(x, y, data=None, hue=None, keys=None) -> pl.LazyFrame`**

Fit and return results in one step. Recommended for most use cases.

**`predict(x: Union[Sequence[float], pl.Series]) -> pl.LazyFrame`**

Make predictions on new x values using the fitted binning scheme.

**`ols_statistics_`**

Property that returns OLS regression statistics as a `pl.DataFrame` (only available if `ols` was enabled).

---

### KernelSmoother

The `KernelSmoother` class performs kernel smoothing using the Epanechnikov kernel with automatic bandwidth selection.

#### Parameters

```python
KernelSmoother(n_eval_samples: int = 100, hue: Optional[Sequence[str]] = None)
```

| Parameter        | Type            | Default | Description                                                         |
| ---------------- | --------------- | ------- | ------------------------------------------------------------------- |
| `n_eval_samples` | int             | `100`   | Number of evaluation points where the kernel smoother is evaluated  |
| `hue`            | sequence of str | `None`  | Optional grouping variable(s). Can be set here or passed to `fit()` |

#### Methods

**`fit(x, y, data=None, hue=None) -> KernelSmoother`**

Fit the kernel smoother to data. Uses Silverman's rule of thumb for bandwidth selection.

- **x**: Column name if `data` provided, else array-like (must be univariate)
- **y**: Column name if `data` provided, else array-like (must be univariate)
- **data**: `pl.DataFrame`, `pl.LazyFrame`, or `None`
- **hue**: Optional grouping variable(s) for stratified smoothing. Overrides `hue` from `__init__` if provided

Returns: self (fitted estimator)

**`transform() -> pl.LazyFrame`**

Returns the kernel smoothed results.

Output columns:

- `x_eval`: Evaluation points
- `y_kernel`: Kernel-smoothed y values

**`fit_transform(x, y, data=None, hue=None) -> pl.LazyFrame`**

Fit and return results in one step.

---

## Examples

### Example 1: Complete Regressogram Workflow with Visualization

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from rgram import Regressogram, KernelSmoother

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

for ax, binning in zip(axes.flat, ["dist", "width", "int", "unique"]):
    rgram = Regressogram(
        binning=binning,
        ci=(lambda x: x.mean() - 1.96 * x.std(), lambda x: x.mean() + 1.96 * x.std())
    )
    result = rgram.fit_transform(data=df, x="x", y="y_noisy").collect()

    ax.scatter(x, y_noisy, alpha=0.4, s=20, label="observations")
    ax.plot(x, y_true, "g-", linewidth=2, label="true function")
    ax.step(result["x_val"], result["y_pred_rgram"], "r-", linewidth=2, label="rgram")
    ax.fill_between(
        result["x_val"],
        result["y_pred_rgram_lci"],
        result["y_pred_rgram_uci"],
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

### Example 2: Grouped Analysis with Hue

```python
import polars as pl
import numpy as np
from rgram import Regressogram
import matplotlib.pyplot as plt

# Generate data with multiple groups
np.random.seed(42)
n = 200
x = np.tile(np.linspace(0, 10, 100), 2)
group = np.repeat(["Group A", "Group B"], 100)
y_a = np.sin(x[:100]) + np.random.normal(0, 0.4, 100)
y_b = np.cos(x[100:]) + np.random.normal(0, 0.4, 100)
y = np.concatenate([y_a, y_b])

df = pl.DataFrame({"x": x, "y": y, "group": group})

# Fit regressogram with grouping
rgram = Regressogram(binning="dist")
result = rgram.fit_transform(data=df, x="x", y="y", hue="group").collect()

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
for grp in ["Group A", "Group B"]:
    mask = result["group"] == grp
    ax.step(result.filter(mask)["x_val"], result.filter(mask)["y_pred_rgram"],
            label=f"rgram - {grp}", linewidth=2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Example 3: Combining Regressogram with Kernel Smoothing

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

# Step 1: Fit regressogram
rgram = Regressogram(binning="dist", ci=None)  # No CI for clarity
rgram_result = rgram.fit_transform(data=df, x="x", y="y").collect()

# Step 2: Smooth the regressogram predictions
smoother = KernelSmoother(n_eval_samples=200)
smoothed = smoother.fit_transform(
    data=rgram_result, x="x_val", y="y_pred_rgram"
).collect()

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x, y, alpha=0.3, s=20, label="Raw observations")
ax.step(rgram_result["x_val"], rgram_result["y_pred_rgram"],
        where="post", linewidth=2, label="Regressogram")
ax.plot(smoothed["x_eval"], smoothed["y_kernel"],
        linewidth=2.5, color="green", label="Kernel smoothed")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 4: Custom Aggregation Functions

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
result = rgram_median.fit_transform(data=df, x="x", y="y").collect()

print("Regressogram with median aggregation:")
print(result.select(["x_val", "y_pred_rgram", "y_pred_rgram_lci", "y_pred_rgram_uci"]).head())
```

## Guides

### Choosing a Binning Strategy

**Distribution-based binning (`"dist"`)** ← Default choice for most cases

- Uses Scott's bandwidth rule to adapt bin width to data density
- Fewer, wider bins where data is sparse; more bins where data is dense
- **Best for**: Normal or near-normal distributions, data exploration
- **Example**: Customer age analysis with uneven age distribution

```python
rgram = Regressogram(binning="dist")
# Automatically creates wider bins for underrepresented ages
result = rgram.fit_transform(data=df, x="age", y="purchase_amount").collect()
```

**Fixed-width binning (`"width"`)**

- Creates equal-sized bins across the entire range
- Consistent interpretation across bins
- **Best for**: When bin boundaries have business meaning (e.g., income brackets: $0-50K, $50-100K, etc.)
- **Drawback**: May have very sparse bins at data extremes

```python
rgram = Regressogram(binning="width")
# Creates consistent income brackets, though some may be nearly empty
result = rgram.fit_transform(data=df, x="annual_income", y="credit_score").collect()
```

**Integer binning (`"int"`)**

- Casts x values to integers and groups by integer value
- **Best for**: Truly discrete integer data (number of items, years of experience)
- **Example**: Product rating (1-5 stars) vs review count

```python
rgram = Regressogram(binning="int")
result = rgram.fit_transform(data=df, x="product_rating", y="num_reviews").collect()
```

**Unique value binning (`"unique"`)**

- Treats each unique x value as its own independent bin
- **Best for**: Computing statistics at each unique x value without binning across x values
- **Use case**: When x is already categorical or when you want predictions for each exact x value

```python
rgram = Regressogram(binning="unique", agg=lambda x: x.mean())
result = rgram.fit_transform(data=df, x="x_col", y="target").collect()
# Result has one row per unique x value with its corresponding y statistic
# No binning/grouping across different x values occurs
```

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

result = rgram_count.fit_transform(data=df, x="x", y="dummy_col").collect()
# y_pred_rgram now contains bin counts
```

### Working with Groups (Hue)

Use the `hue` parameter to analyze multiple subgroups simultaneously:

```python
from rgram import Regressogram
import polars as pl

# Sales by region
result = Regressogram(binning="dist").fit_transform(
    data=sales_df,
    x="time_of_week",
    y="revenue",
    hue="region"  # Analyze each region separately
).collect()

# Output includes a "region" column; group results show region-specific trends
for region in result["region"].unique():
    region_data = result.filter(pl.col("region") == region)
    print(f"Region {region} - Mean revenue: {region_data['y_pred_rgram'].mean()}")
```

**Benefits of hue**:

- Single fit call vs multiple calls for each group
- Ensures consistent binning across groups
- More efficient than separate analyses
- Output includes group identifier for easy filtering

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
result = rgram.fit_transform(data=df, x="age", y="salary").collect()
```

**Pattern 2: Raw Arrays/Series** (Best for quick analysis, interactive work)

```python
import numpy as np
from rgram import Regressogram

x = np.array([25, 30, 35, 40, 45])
y = np.array([50000, 55000, 60000, 70000, 80000])

# Pass arrays directly without a DataFrame (like seaborn.kdeplot with just x=)
rgram = Regressogram()
result = rgram.fit_transform(x=x, y=y).collect()
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
result = rgram.fit_transform(x=df["age"], y=df["salary"]).collect()
```

**Pattern 4: Multiple features/targets**

```python
# Multiple x columns (analyzed as separate x-y pair combinations)
result = rgram.fit_transform(
    data=df,
    x=["age", "experience"],
    y="salary"
).collect()
# Creates separate results for each x-y feature pair
```

**When to use which pattern:**

- **DataFrame + names**: Production code, complex pipelines, grouped analysis (hue parameter)
- **Raw arrays**: Quick exploration, notebooks, when data is already in memory
- **Series**: Intermediate between the two; good for simple scripts

## Benefits vs Limitations

### Advantages of Regressogram

| Benefit              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **Interpretability** | Step-wise predictions are easy to explain to stakeholders ("if age 25-35, avg salary is X") |
| **Robustness**       | Can use median or quantiles instead of mean for outlier-resistant estimates                 |
| **Flexibility**      | Custom aggregation functions support domain-specific logic (e.g., weighted means)           |
| **Multi-group**      | `hue` parameter enables simultaneous analysis of multiple subgroups with consistent binning |
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
| **1D only**                   | Cannot handle high dimensions directly           | Use dimensionality reduction or analyze features independently                |
| **No feature selection**      | All x variables are used                         | Pre-select relevant features based on domain knowledge                        |
| **Binning creates artifacts** | Regressogram has artificial step discontinuities | Use KernelSmoother after Regressogram, or use Regressogram only for EDA       |
| **Bandwidth sensitivity**     | Kernel results vary with bandwidth choice        | Silverman's rule is automatic; use cross-validation for critical applications |
| **Memory for large data**     | Lazy evaluation has limits during `.collect()`   | Process data in batches; use Polars partitioning                              |
| **No missing value handling** | NaN values cause errors                          | Impute or remove missing values before fitting                                |
| **No real-time predictions**  | Must refit to add new data                       | Refitting is fast enough for small->medium datasets                           |
| **Categorical inputs**        | X and Y must be numeric                          | Encode categorical variables (ordinal encoding or one-hot + aggregation)      |

## Limitations

- **Univariate Kernel Smoothing**: `KernelSmoother` currently only supports single-variable smoothing. Multivariate kernel smoothing is not yet implemented.

- **Bandwidth Selection**: Kernel smoothing uses Silverman's rule of thumb for bandwidth selection, which may not be optimal for all data distributions. Manual bandwidth tuning is not currently available.

- **Binning Strategy Selection**: The choice of binning strategy can significantly impact results. The library provides multiple strategies but does not automatically select the optimal one. Users should experiment or use cross-validation.

- **Memory Efficiency**: For very large datasets, even with Polars' optimizations, lazy evaluation may be limited by system memory during collection.

- **Multi-dimensional Input**: Regressogram and KernelSmoother are designed for 1D→1D mappings. Multi-dimensional feature spaces require feature engineering or multiple univariate analyses.

- **Missing Values**: The current implementation does not explicitly handle missing values. Pre-processing with appropriate techniques (imputation, removal) is required.

- **Categorical Features**: Both classes require numerical input. Categorical variables must be encoded numerically before use.

## Troubleshooting

### Common Issues and Solutions

**Q: I'm getting `RuntimeError: You must call fit() before transform()`**

A: Ensure you call `.fit()` before `.transform()`, or use `.fit_transform()` as a shortcut:

```python
# ❌ Wrong
smoother = KernelSmoother()
result = smoother.transform()  # Error!

# ✅ Correct
smoother = KernelSmoother()
smoother.fit(data=df, x="x", y="y")
result = smoother.transform()

# ✅ Alternative (recommended)
smoother = KernelSmoother()
result = smoother.fit_transform(data=df, x="x", y="y")
```

**Q: My results have very few bins or all data in one bin**

A: This often happens with highly skewed distributions. Try different binning strategies:

```python
from rgram import Regressogram

# If dist binning creates too few bins, try width or int
rgram_width = Regressogram(binning="width")
result = rgram_width.fit_transform(data=df, x="age", y="income").collect()

# Or check your data distribution first
print(f"X range: {df['age'].min()} to {df['age'].max()}")
print(f"X std: {df['age'].std()}, IQR: {df['age'].quantile(0.75) - df['age'].quantile(0.25)}")
```

**Q: KernelSmoother results look too smooth/too wiggly**

A: Adjust the `n_eval_samples` parameter (more samples = smoother curve with finer detail):

```python
# Too wiggly? Use fewer evaluation points
smoother_smooth = KernelSmoother(n_eval_samples=50)

# Want more detail? Use more evaluation points
smoother_detailed = KernelSmoother(n_eval_samples=500)
```

**Q: Getting `ValueError: If data is None, input must be an array-like`**

A: You mixed column names with array mode. Either pass column names with a DataFrame, or pass arrays without a DataFrame:

```python
import polars as pl
import numpy as np
from rgram import Regressogram

# ❌ Wrong mix
df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
result = Regressogram().fit_transform(x=df["x"], y="y", data=df)  # Error!

# ✅ Correct: use column names with data
result = Regressogram().fit_transform(data=df, x="x", y="y").collect()

# ✅ Correct: use arrays without data
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
result = Regressogram().fit_transform(x=x, y=y).collect()
```

**Q: My predictions have NaN values in confidence intervals**

A: This occurs when `allow_negative_y="auto"` clamps negative confidence bounds to None. Check if your CI functions produce negative values:

```python
from rgram import Regressogram

# Check your CI functions
rgram = Regressogram(
    ci=(
        lambda x: x.mean() - 2 * x.std(),  # May go negative!
        lambda x: x.mean() + 2 * x.std()
    ),
    allow_negative_y=True  # Allow negative predictions
)
result = rgram.fit_transform(data=df, x="x", y="y").collect()
```

**Q: Data collection `.collect()` is very slow or runs out of memory**

A: Process data in batches using Polars filtering:

```python
from rgram import Regressogram

rgram = Regressogram()
rgram.fit(data=large_df, x="x", y="y")

# Process in batches instead of collecting everything at once
for batch_df in large_df.partition_by("date"):  # or any grouping column
    results = rgram.transform().filter(
        pl.col("date") == batch_df["date"][0]
    ).collect()
    print(results.head())
```

**Q: Getting different results between runs with the same data**

A: Polars operations are deterministic. Different results usually mean:

1. Random seed not set (if you generated synthetic data)
2. Parallelization order differences (set `polars.Config.set_streaming_chunk_size()`)
3. Data changed between runs (verify your input data)

```python
import polars as pl
import numpy as np

# Set reproducible random seed
np.random.seed(42)
pl.Config.set_random_seed(42)

# Your analysis...
```

**Q: Hue grouping not working as expected**

A: Ensure the `hue` column exists and reference it correctly:

```python
from rgram import Regressogram
import polars as pl

# Verify hue column exists
print(df.columns)

# Make sure data is not None when using column names
result = Regressogram().fit_transform(
    data=df,  # Must provide data when using column names
    x="x_col",
    y="y_col",
    hue="group_col"  # This must be a column in df
).collect()
```

## Future Improvements

### High Priority

- [ ] **Multivariate Kernel Smoothing**: Extend `KernelSmoother` to support multi-dimensional input with optimal bandwidth selection for each dimension
- [ ] **Bandwidth Tuning**: Add methods for cross-validated bandwidth selection and user-specified bandwidth parameters
- [ ] **Missing Data Handling**: Built-in support for various imputation strategies and missing value indicators
- [ ] **Auto Binning Strategy Selection**: Data-driven method to select optimal binning strategy using cross-validation

### Medium Priority

- [ ] **Adaptive Binning**: Implement data-driven bin size selection using information-theoretic criteria
- [ ] **Confidence Band Methods**: Additional methods for computing confidence bands (e.g., bootstrap, Bayesian)
- [ ] **Plotting Utilities**: High-level visualization functions with matplotlib/plotly backends
- [ ] **Performance Profiling**: Detailed benchmarks and optimization for large-scale datasets

### Lower Priority

- [ ] **Additional Kernels**: Support for Gaussian, Triangular, and other kernel types
- [ ] **Categorical Grouping Optimization**: Special handling for categorical `hue` variables
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
