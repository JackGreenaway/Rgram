"""
This module provides the `pl_rgram` function, which generates a regression gram (rgram)
using Polars DataFrame or LazyFrame. It supports binning by index or distribution and
optionally adds Ordinary Least Squares (OLS) regression calculations.
"""

import polars as pl
import polars_ols as pls  # noqa: F401
from typing import Callable, Literal


def rgram(
    df: pl.LazyFrame | pl.DataFrame,
    x: str | list[str],
    y: str,
    metric: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
    hue: str | list = None,
    add_ols: bool = True,
    bin_style: Literal["width", "dist"] = "width",
    allow_negative_ols: bool = False,
    keys: str | list = None,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Generate a regression gram (rgram) to analyse relationships between variables.

    Parameters
    ----------
    df : pl.LazyFrame or pl.DataFrame
        Input data as a Polars LazyFrame or DataFrame.
    x : str or list of str
        Independent variable(s) to analyse.
    y : str
        Dependent variable to analyse.
    metric : Callable[[pl.Expr], pl.Expr], default=lambda x: x.mean()
        Function to compute the metric for the dependent variable.
    hue : str or list, optional
        Categorical variable(s) for grouping. Default is None.
    add_ols : bool, default=True
        Whether to include Ordinary Least Squares (OLS) regression calculations.
    bin_style : {'width', 'dist'}, default='width'
        Binning style, either 'width' or 'dist'.
    allow_negative_ols : bool, default=False
        Whether to allow negative OLS predictions.
    keys : str or list, optional
        Additional grouping keys. Default is None.

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        A Polars LazyFrame or DataFrame containing the rgram results.
    """
    x = [x] if isinstance(x, str) else x
    hue = [hue] if isinstance(hue, str) else hue
    keys = [keys] if isinstance(keys, str) else keys

    idx_features = [y]
    over_features = ["x_var"]

    if hue:
        idx_features.extend(hue)
        over_features.extend(hue)
    if keys:
        idx_features.extend(keys)

    friedman_rot = 2 * (
        pl.col("x_val").quantile(0.75).sub(pl.col("x_val").quantile(0.25))
        / pl.len().pow(1 / 3)
    )
    data_range = pl.col("x_val").max() - pl.col("x_val").min()

    bin_style_dict = {
        "dist": (pl.col("x_val").rank(method="ordinal") * (data_range / friedman_rot))
        // pl.len(),
        "width": (pl.col("x_val") // friedman_rot),
    }
    bin_calc = [bin_style_dict[bin_style].over(over_features).alias("rgram_bin")]

    ols_calc = (
        [
            pl.col(y)
            .least_squares.ols(
                pl.col("x_val").alias("coef"),
                mode=mode,
                add_intercept=True,
                null_policy="drop",
            )
            .over(over_features)
            .alias(alias)
            for mode, alias in [
                ("predictions", "y_pred_ols"),
                ("coefficients", "ols_coef"),
            ]
        ]
        if add_ols
        else []
    )

    rgram = (
        df.select(x + idx_features)
        .unpivot(
            on=x,
            index=idx_features,
            variable_name="x_var",
            value_name="x_val",
        )
        .with_columns(
            [
                # ensure target is float (important for boolean targets)
                pl.col(y).cast(float).alias(y),
            ]
            + bin_calc
        )
        .with_columns(
            [
                metric(pl.col(y))
                .over(over_features + ["rgram_bin"])
                .alias("y_pred_rgram")
            ]
            + ols_calc
        )
    )

    if add_ols:
        rgram = rgram.with_columns(
            [
                pl.col("ols_coef").struct.field("coef").alias("coef"),
                pl.col("ols_coef").struct.field("const").alias("const"),
            ]
            + (
                [
                    pl.when(pl.col("y_pred_ols") < 0)
                    .then(None)
                    .otherwise(pl.col("y_pred_ols"))
                    .alias("y_pred_ols")
                ]
                if not allow_negative_ols
                else []
            )
        ).drop(["ols_coef"])

    return rgram.sort(by=["x_val"])
