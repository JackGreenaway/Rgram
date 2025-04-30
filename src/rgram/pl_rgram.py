"""
This module provides the `pl_rgram` function, which generates a regression gram (rgram)
using Polars DataFrame or LazyFrame. It supports binning by index or distribution and
optionally adds Ordinary Least Squares (OLS) regression calculations.
"""

import polars as pl
import polars_ols as pls  # registered namespace
from typing import Callable, Literal


def pl_rgram(
    df: pl.LazyFrame | pl.DataFrame,
    x: str | list[str],
    y: str,
    metric: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
    hue: str = None,
    add_ols: bool = True,
    bin_style: Literal["index", "dist"] = "index",
) -> pl.LazyFrame | pl.DataFrame:
    """
    Generate a regression gram (rgram) for visualizing relationships between variables
    in a Polars DataFrame or LazyFrame.

    Parameters:
        df (pl.LazyFrame | pl.DataFrame): The input data as a Polars LazyFrame or DataFrame.
        x (str | list[str]): The independent variable(s) to analyze.
        y (str): The dependent variable to analyze.
        metric (Callable[[pl.Expr], pl.Expr], optional): A function to compute the metric
            for the dependent variable (default is mean).
        hue (str, optional): A categorical variable for grouping (default is None).
        add_ols (bool, optional): Whether to include OLS regression calculations (default is True).
        bin_style (Literal["index", "dist"], optional): The binning style, either "index" or "dist"
            (default is "index").

    Returns:
        pl.LazyFrame | pl.DataFrame: A Polars LazyFrame or DataFrame containing the rgram results.
    """
    x = list(x)

    idx_features = [y, hue] if hue else [y]
    over_features = [hue, "x_var"] if hue else ["x_var"]

    friedman_rot = (2 * (pl.len().pow(1 / 3))).ceil()

    if bin_style == "index":
        bin_calc = [
            (pl.arange(0, pl.len()) // friedman_rot)
            .over(over_features)
            .alias("rgram_bin")
        ]

    elif bin_style == "dist":
        bin_calc = [
            (((pl.col("x_var").rank(method="ordinal") - 1) * friedman_rot) // pl.len())
            .over(over_features)
            .cast(float)
            .alias("rgram_bin")
        ]

    ols_calc = []
    if add_ols:
        ols_calc = [
            pl.col(y)
            .least_squares.ols(
                pl.col("x_val"), mode=mode, add_intercept=True, null_policy="drop"
            )
            .over(over_features)
            .alias(alias)
            for mode, alias in [
                ("predictions", "y_pred_ols"),
                # ("coefficients", "ols_coef")
            ]
        ]

    rgram = (
        df.lazy()
        .select(x + idx_features)
        .unpivot(on=x, index=idx_features, variable_name="x_var", value_name="x_val")
        .sort(by=["x_val"])
        .with_columns(
            [
                # ensure target is float (important for boolean targets)
                pl.col(y).cast(float).alias(y),
            ]
            # index or dist binning
            + bin_calc
        )
        .with_columns(
            # rgram
            [
                metric(pl.col(y))
                .over(over_features + ["rgram_bin"])
                .alias("y_pred_rgram")
            ]
            # ols
            + ols_calc
        )
    )

    return rgram
