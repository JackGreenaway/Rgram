import polars as pl
from typing import Sequence
import warnings


def kernel_smoothing(
    df: pl.DataFrame | pl.LazyFrame,
    x: str,
    y: str,
    hue: Sequence[str] | None = None,
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
    hue : list of str or None, optional
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

    warnings.warn(
        "kernel_smoothing is deprecated and will be removed in a future release."
        "Use KernelSmoother from rgram.rgram instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise ValueError("df must be a Polars DataFrame or LazyFrame")

    for col in [x, y] + (list(hue) if hue else []):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    hue = list(hue) if hue else []

    def over_function(expr: pl.Expr) -> pl.Expr:
        return expr.over(hue) if hue else expr

    ldf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # calculate bandwidth h using silverman's rule of thumb
    h_expr = over_function(
        0.9
        * pl.min_horizontal(
            [
                pl.col(x).std(),
                (pl.col(x).quantile(0.75) - pl.col(x).quantile(0.25)) / 1.34,
            ]
        )
        * (pl.len() ** (-1 / 5))
    ).alias("h")

    # generate evaluation points
    x_eval_expr = over_function(
        pl.linear_spaces(
            pl.col(x).min(), pl.col(x).max(), n_eval_samples, as_array=True
        )
    ).alias("x_eval")

    ks_df = (
        ldf.with_columns([h_expr, x_eval_expr])
        .explode("x_eval")
        .with_columns([(pl.col("x_eval") - pl.col(x)).truediv("h").alias("u")])
        .filter(pl.col("u").abs() <= 1)
        .with_columns([(0.75 * (1 - pl.col("u").pow(2))).alias("weight")])
        .group_by(["x_eval"] + hue)
        .agg(
            [
                ((pl.col(y) * pl.col("weight")).sum() / pl.col("weight").sum()).alias(
                    "y_kernel"
                )
            ]
        )
        .sort("x_eval")
    )

    return ks_df
