import numpy as np
from numpy.typing import ArrayLike
from typing import Literal


def silvermans_rule(y: ArrayLike) -> float:
    if y.size == 0:
        raise ValueError("Input array 'y' must not be empty.")

    std = np.std(y)
    iqr = (np.quantile(y, 0.75) - np.quantile(y, 0.25)) / 1.34
    return 0.9 * min(std, iqr) * (y.shape[0] ** (-1 / 5))


def freedman_diaconis_rule(y: ArrayLike) -> int:
    if y.size == 0:
        raise ValueError("Input array 'y' must not be empty.")

    iqr = np.quantile(y, 0.75) - np.quantile(y, 0.25)

    return int(np.ceil(2 * (iqr / y.shape[0] ** (1 / 3))))


def regressorgram(
    x: ArrayLike,
    y: ArrayLike,
    n_bins: int = None,
    bin_type: Literal["naive", "quantile"] = "naive",
) -> ArrayLike:
    n_bins = n_bins or int(np.ceil(2 * (x.shape[0] ** (1 / 3))))
    # n_bins = n_bins or freedman_diaconis_rule(y=y)

    if bin_type == "naive":
        bins = np.arange(x.shape[0]) // n_bins
    elif bin_type == "quantile":
        bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.digitize(x, bin_edges, right=True) - 1, 0, n_bins - 1)
    else:
        raise ValueError("Invalid bin_type. Choose 'naive' or 'quantile'.")

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
    if x_train.size == 0 or y_train.size == 0:
        raise ValueError("Input arrays 'x_train' and 'y_train' must not be empty.")

    x_eval = x_eval or np.linspace(x_train.min(), x_train.max(), x_train.shape[0])
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
