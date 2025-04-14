# Rgram

Rgram is a Python library for performing regression histograms and kernel smoothing. This repository is designed to provide tools for data analysis and visualization. It uses `uv` for dependency management.

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

Here are examples of how to use the `regressorgram` and `epanchenkov_kernel` functions:

### Example: Regression Histogram with Kernel Smoothing
```python
import numpy as np
import matplotlib.pyplot as plt
from rgram.rgram import regressorgram, epanchenkov_kernel

# Generate sample data
x = np.sort(np.random.uniform(0, 10, 250))
y = np.sin(x) + np.cos(x) ** 2
y_noise = y + np.random.normal(0, 0.5, size=x.shape)

# Apply regression histogram with quantile binning
regressogram = regressorgram(x=x, y=y_noise, bins_param=10, bin_type="naive")

# Smooth the regression histogram output using the Epanechnikov kernel
kernel = epanchenkov_kernel(x_train=x, y_train=regressogram)

# Plot the regressogram
plt.plot(x, y, label="True Function", color="green")
plt.scatter(x, y_noise, s=3, label="Noisy Data", alpha=0.5)
plt.step(x, regressogram, label="Regression Histogram", color="blue", where="mid", alpha=0.3)
plt.plot(
    np.linspace(x.min(), x.max(), 250), kernel, label="Kernel Smoothed", color="red"
)
plt.legend()
plt.show()
```

![Example Output](example.png)
