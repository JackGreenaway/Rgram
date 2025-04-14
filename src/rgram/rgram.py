import numpy as np
from numpy.typing import ArrayLike
from typing import Literal


def regressorgram(
    x: ArrayLike,
    y: ArrayLike,
    bins_param: int,
    bin_type: Literal["naive", "quantile"] = "naive",
) -> ArrayLike:
    if bin_type == "naive":
        bins = np.arange(x.shape[0]) // bins_param

        bin_sums = np.bincount(bins, weights=y)
        bin_counts = np.bincount(bins)

        bin_means = bin_sums / bin_counts

        return bin_means[bins]

    elif bin_type == "quantile":
        bin_edges = np.quantile(x, np.linspace(0, 1, bins_param + 1))
        bins = np.digitize(x, bin_edges, right=True) - 1
        bins = np.clip(bins, 0, bins_param - 1)

        bin_sums = np.bincount(bins, weights=y)
        bin_counts = np.bincount(bins)

        bin_means = bin_sums / bin_counts

        return bin_means[bins]


def rule_of_thumb(y: ArrayLike) -> float:
    std = np.std(y)
    iqr = np.subtract(*np.quantile(y, [0.75, 0.25])) / 1.34

    return 0.9 * np.minimum(std, iqr) * (y.shape[0] ** (-1 / 5))


def epanchenkov_kernel(
    x_train: ArrayLike, y_train: ArrayLike, x_eval: ArrayLike = None, h: float = None
) -> ArrayLike:
    x_eval = (
        np.linspace(x_train.min(), x_train.max(), x_train.shape[0])
        if not x_eval
        else x_eval
    )
    h = rule_of_thumb(y_train) if not h else h

    u = (x_eval.reshape(-1, 1) - x_train) / h

    mask = np.abs(u) <= 1
    weight = 0.75 * (1 - u**2) * mask

    sum_weight = np.sum(weight, axis=1)
    kernel = np.divide(np.dot(weight, y_train), sum_weight, where=sum_weight > 0)

    return kernel
