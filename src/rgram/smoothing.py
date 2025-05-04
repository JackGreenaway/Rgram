import polars as pl
import numpy as np
from numpy.typing import ArrayLike


def silverman_rot(y: pl.Series | ArrayLike) -> float:
    """
    Calculate the bandwidth using Silverman's rule of thumb.

    Parameters:
        y (pl.Series | ArrayLike): Input array of data points.

    Returns:
        float: The calculated bandwidth.
    """
    std = y.std()
    iqr = np.subtract(*np.quantile(y, [0.75, 0.25])) / 1.34

    return 0.9 * min(std, iqr) * (y.shape[0] ** (-1 / 5))

