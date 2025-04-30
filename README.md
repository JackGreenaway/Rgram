# Rgram

Rgram is a Python library for performing regressograms and kernel smoothing. This repository is designed to provide tools for data analysis and visualization. It uses `uv` for dependency management.

## Features

- Regression histograms with naive and quantile binning.
- Kernel smoothing using the Epanechnikov kernel.
- Easy-to-use API for data analysis.

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

Here are examples of how to use the `regressorgram` and `kernel_smoothing` functions:

### Example: Regression Histogram with Kernel Smoothing
```python
import numpy as np
import matplotlib.pyplot as plt
from rgram.np_rgram import regressorgram, kernel_smoothing

# Generate sample data
n = 50
x = np.sort(np.random.normal(0, 1, n))
y = 1 + x
y_noise = y + np.random.normal(0, np.sqrt(2), n)

# Apply regression histogram with quantile binning
regressogram = regressorgram(x=x, y=y_noise, bin_style="index")

fig, ax = plt.subplots(figsize=(5, 5))

# Plot the regressogram
ax.plot(x, y, label="true function", color="black", lw=0.5)
ax.scatter(x, y_noise, s=15, alpha=0.3, marker="o", color="black")

ax.step(x, regressogram, label="regressogram", where="mid", lw=0.5)

x_eval = np.linspace(x.min(), x.max(), 1000)
for kernel in [
    "epanchenkov",
    # "nadaraya_watson",
    # "priestley_chao"
]:
    kernel_smoothed = kernel_smoothing(
        x_train=x, y_train=y_noise, x_eval=x_eval, kernel=kernel
    )

    ax.plot(
        x_eval,
        kernel_smoothed,
        label=kernel,
        lw=0.5,
    )

plt.legend()
plt.tight_layout()
plt.show()
```

This example demonstrates how to generate a regressogram and apply kernel smoothing using the Epanechnikov kernel. Uncomment additional kernels to explore their effects.

<p align="center">
  <img src="example.png" />
</p>
