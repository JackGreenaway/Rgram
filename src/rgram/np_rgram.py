"""
This module provides utility functions for statistical analysis and regression,
including Silverman's rule, Freedman-Diaconis rule, regressor grams, and kernel smoothing.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Literal


def silvermans_rule(y: ArrayLike) -> float:
    """
    Calculate the bandwidth using Silverman's rule of thumb.

    Parameters:
        y (ArrayLike): Input array of data points.

    Returns:
        float: The calculated bandwidth.

    Raises:
        ValueError: If the input array 'y' is empty.
    """
    if y.size == 0:
        raise ValueError("Input array 'y' must not be empty.")

    std = np.std(y)
    iqr = (np.quantile(y, 0.75) - np.quantile(y, 0.25)) / 1.34
    return 0.9 * min(std, iqr) * (y.shape[0] ** (-1 / 5))


def freedman_diaconis_rule(y: ArrayLike) -> int:
    """
    Calculate the number of bins using the Freedman-Diaconis rule.

    Parameters:
        y (ArrayLike): Input array of data points.

    Returns:
        int: The calculated number of bins.

    Raises:
        ValueError: If the input array 'y' is empty.
    """
    if y.size == 0:
        raise ValueError("Input array 'y' must not be empty.")

    iqr = np.quantile(y, 0.75) - np.quantile(y, 0.25)

    return int(np.ceil(2 * (iqr / y.shape[0] ** (1 / 3))))


def regressorgram(
    x: ArrayLike,
    y: ArrayLike,
    n_bins: int = None,
    bin_style: Literal["index", "dist"] = "index",
) -> ArrayLike:
    """
    Generate a regressor gram by binning the data and calculating mean values for each bin.

    Parameters:
        x (ArrayLike): Independent variable data.
        y (ArrayLike): Dependent variable data.
        n_bins (int, optional): Number of bins (default is calculated using a heuristic).
        bin_type (Literal["index", "dist"], optional): Binning method, either "index" or "dist"
            (default is "index").

    Returns:
        ArrayLike: Array of mean values for each bin.

    Raises:
        ValueError: If an invalid bin_type is provided.
    """
    n_bins = n_bins or int(np.ceil(2 * (x.shape[0] ** (1 / 3))))
    # n_bins = n_bins or freedman_diaconis_rule(y=y)

    if bin_style == "index":
        bins = np.arange(x.shape[0]) // n_bins
    elif bin_style == "dist":
        bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.digitize(x, bin_edges, right=True) - 1, 0, n_bins - 1)
    else:
        raise ValueError("Invalid bin_type. Choose 'index' or 'dist'.")

    bin_sums = np.bincount(bins, weights=y)
    bin_counts = np.bincount(bins)
    bin_means = np.divide(bin_sums, bin_counts, where=bin_counts > 0)

    return bin_means[bins]


def kernel_smoothing(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_eval: ArrayLike = None,
    h: float = None,
    kernel: Literal["epanchenkov", "nadaraya_watson", "priestley_chao"] = "epanchenkov",
) -> ArrayLike:
    """
    Perform kernel smoothing on the input data.

    Parameters:
        x_train (ArrayLike): Training data for the independent variable.
        y_train (ArrayLike): Training data for the dependent variable.
        x_eval (ArrayLike, optional): Evaluation points for the independent variable
            (default is a linear space between min and max of x_train).
        h (float, optional): Bandwidth for the kernel (default is calculated using Silverman's rule).
        kernel (Literal["epanchenkov", "nadaraya_watson", "priestley_chao"], optional): Kernel type
            (default is "epanchenkov").

    Returns:
        ArrayLike: Smoothed values for the evaluation points.

    Raises:
        ValueError: If input arrays 'x_train' or 'y_train' are empty.
    """
    if x_train.size == 0 or y_train.size == 0:
        raise ValueError("Input arrays 'x_train' and 'y_train' must not be empty.")

    if x_eval is None:
        x_eval = np.linspace(x_train.min(), x_train.max(), x_train.shape[0])

    h = h or silvermans_rule(y_train)

    u = (x_eval[:, None] - x_train) / h

    if kernel == "epanchenkov":
        weights = 0.75 * (1 - u**2) * (np.abs(u) <= 1)

    elif kernel == "nadaraya_watson":
        weights = np.exp(-0.5 * u**2) / h

    elif kernel == "priestley_chao":
        dx = np.diff(x_train)
        dx = np.append(dx, dx[-1])
        weights = np.exp(-0.5 * u**2) * dx

    sum_weights = np.sum(weights, axis=1)
    kernel = np.divide(np.dot(weights, y_train), sum_weights, where=sum_weights > 0)

    return kernel
