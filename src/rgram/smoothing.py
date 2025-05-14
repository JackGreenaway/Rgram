import polars as pl
import numpy as np
from numpy.typing import ArrayLike


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
        The calculated bandwidth based on the input data.
    """
    std = y.std()
    iqr = np.subtract(*np.quantile(y, [0.75, 0.25])) / 1.34

    return 0.9 * min(std, iqr) * (y.shape[0] ** (-1 / 5))


def kernel_smoothing(
    df: pl.DataFrame | pl.LazyFrame,
    x: str,
    y: str,
    hue: list[str] = [],
    n_eval_samples: int = 500,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Perform kernel smoothing using the Epanechnikov kernel.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data containing the independent and dependent variables.
    x : str
        Name of the column representing the independent variable.
    y : str
        Name of the column representing the dependent variable.
    hue : list of str, default=[]
        List of column names to group the data by before applying smoothing.
    n_eval_samples : int, default=500
        Number of evenly spaced evaluation points for the independent variable.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        A DataFrame or LazyFrame containing the smoothed values for the
        dependent variable (`y`) at the evaluation points for the independent
        variable (`x`).

    Notes
    -----
    The kernel smoothing is performed using the Epanechnikov kernel, defined as
    `0.75 * (1 - u^2)` for `|u| <= 1`, where `u` is the scaled distance between
    the evaluation and training points. The bandwidth `h` is calculated using
    Silverman's rule of thumb.

    The function generates `n_eval_samples` evenly spaced evaluation points for
    the independent variable (`x`) within its range. For each evaluation point,
    the smoothed value of the dependent variable (`y`) is computed as a weighted
    average of nearby points, with weights determined by the kernel function.

    Examples
    --------
    >>> import polars as pl
    >>> from smoothing import kernel_smoothing
    >>> df = pl.DataFrame({
    ...     "x": [1, 2, 3, 4, 5],
    ...     "y": [2, 4, 6, 8, 10]
    ... })
    >>> smoothed_df = kernel_smoothing(df, x="x", y="y", n_eval_samples=100)
    >>> smoothed_df.head()
    """

    def over_function(x: pl.Expr) -> pl.Expr:
        if hue:
            return (x).over(hue)

        return x

    ks_df = (
        df.with_columns(
            [
                over_function(
                    0.9
                    * pl.min_horizontal(
                        [
                            pl.col(x).std(),
                            (
                                (pl.col(x).quantile(0.75) - pl.col(x).quantile(0.25))
                                / 1.34
                            ),
                        ]
                    )
                    * (pl.len() ** (-1 / 5))
                ).alias("h")
            ]
        )
        .with_columns(
            [
                over_function(
                    pl.linear_spaces(
                        pl.col(x).min(), pl.col(x).max(), n_eval_samples, as_array=True
                    )
                ).alias("x_eval")
            ]
        )
        .explode("x_eval")
        .with_columns([(pl.col("x_eval") - pl.col(x)).truediv("h").alias("u")])
        .filter(pl.col("u").abs() <= 1)
        .with_columns([(0.75 * (1 - pl.col("u").pow(2))).alias("weight")])
        .group_by(["x_eval"] + hue)
        .agg(
            [
                ((pl.col(y) * pl.col("weight")).sum() / pl.col("weight").sum()).alias(
                    "kernel"
                ),
            ]
        )
        .sort(by="x_eval")
    )

    return ks_df
