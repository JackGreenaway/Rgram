import polars as pl
import numpy as np
from numpy.typing import ArrayLike
from typing import Literal


def silverman_rot(y: pl.Series | ArrayLike) -> float:
    """
    Calculate the bandwidth using Silverman's rule of thumb.

    Parameters
    ----------
    y : pl.Series or ArrayLike
        Input array of data points.

    Returns
    -------
    float
        The calculated bandwidth.
    """
    std = y.std()
    iqr = np.subtract(*np.quantile(y, [0.75, 0.25])) / 1.34

    return 0.9 * min(std, iqr) * (y.shape[0] ** (-1 / 5))


def kernel_smoothing(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_eval: ArrayLike = None,
    h: float = None,
    kernel: Literal["epanchenkov", "nadaraya_watson", "priestley_chao"] = "epanchenkov",
) -> ArrayLike:
    """
    Perform kernel smoothing using the Epanechnikov kernel.

    Parameters
    ----------
    x_train : ArrayLike
        Training data input (independent variable).
    y_train : ArrayLike
        Training data output (dependent variable).
    x_eval : ArrayLike, optional
        Evaluation points where the smoothed values are computed. If None,
        250 evenly spaced points between the minimum and maximum of `x_train`
        are used. Default is None.
    h : float, optional
        Bandwidth for the kernel. If None, it is calculated using Silverman's
        rule of thumb. Default is None.
    kernel : {'epanchenkov'}, optional
        The kernel type to use. Supports 'epanchenkov', 'nadaraya_watson', and 'priestley_chao' kernels.
        Default is 'epanchenkov'.

    Returns
    -------
    ArrayLike
        Smoothed values at the evaluation points.
    """
    if x_eval is None:
        x_eval = pl.LazyFrame(
            {"x_eval": np.linspace(x_train.min(), x_train.max(), 250)}
        )
    elif isinstance(x_eval, np.ndarray):
        x_eval = pl.LazyFrame({"x_eval": x_eval})
    elif isinstance(x_eval, pl.DataFrame):
        x_eval = x_eval.lazy()

    h = silverman_rot(x_train)

    training_frame = pl.LazyFrame({"x_train": x_train, "y_train": y_train})

    kernel_technique = {
        "epanchenkov": (0.75 * (1 - pl.col("u").pow(2)) * (pl.col("u").abs() <= 1)),
        "nadaraya_watson": (-0.5 * pl.col("u").pow(2)).exp() / h,
        "priestley_chao": (-0.5 * pl.col("u").pow(2)).exp()
        * pl.col("x_train").diff().fill_null(strategy="forward"),
    }

    kernel = (
        training_frame.lazy()
        .select(["x_train", "y_train"])
        .join(x_eval.lazy(), how="cross")
        .unique()
        .with_columns([pl.col("x_eval").sub(pl.col("x_train")).truediv(h).alias("u")])
        .with_columns(
            [
                # (0.75 * (1 - pl.col("u").pow(2)) * (pl.col("u").abs() <= 1)).alias("weight")
                kernel_technique[kernel].alias("weight")
            ]
        )
        .group_by(["x_eval"])
        .agg(
            [
                (
                    pl.col("y_train").dot(pl.col("weight")) / pl.col("weight").sum()
                ).alias("kernel"),
            ]
        )
        .sort("x_eval")
        .collect()
    )

    return kernel
