"""
This module provides the `rgram` function, which generates a regression gram (rgram)
using a Polars DataFrame or LazyFrame. The function supports various binning styles,
optional Ordinary Least Squares (OLS) regression calculations, and confidence interval
metrics for analysing relationships between variables.
"""

import polars as pl
import polars_ols as pls  # noqa: F401
from typing import Callable, Literal


def rgram(
    df: pl.LazyFrame | pl.DataFrame,
    x: str | list[str],
    y: str,
    metric: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
    ci_metric: tuple[Callable[[pl.Expr], pl.Expr], Callable[[pl.Expr], pl.Expr]]
    | None = (
        lambda x: x.mean() - x.std(),
        lambda x: x.mean() + x.std(),
    ),
    hue: str | list = None,
    calc_ols: bool = True,
    calc_cum_sum: bool = True,
    bin_style: Literal["width", "dist", "unique", "int"] = "width",
    allow_negative_y: Literal[True, False, "auto"] = "auto",
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
    metric : callable, default=lambda x: x.mean()
        Function to compute the metric for the dependent variable.
    ci_metric : tuple of callables or None, default=(lambda x: x.mean() - x.std(), lambda x: x.mean() + x.std())
        Functions to compute the lower and upper confidence intervals for the dependent
        variable. If None, confidence intervals are not calculated.
    hue : str or list, default=None
        Categorical variable(s) for grouping.
    calc_ols : bool, default=True
        Whether to include Ordinary Least Squares (OLS) regression calculations.
    bin_style : {'width', 'dist', 'unique', 'int'}, default='width'
        Binning style for the independent variable:
        - 'width': Fixed-width bins.
        - 'dist': Distribution-based bins.
        - 'unique': Unique values as bins.
        - 'int': Integer-based bins.
    allow_negative_y : {'auto', True, False}, default='auto'
        Whether to allow negative OLS predictions:
        - 'auto': Automatically determine based on the minimum value of `y`.
        - True: Allow negative predictions.
        - False: Disallow negative predictions.
    keys : str or list, default=None
        Additional grouping keys.

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        A Polars LazyFrame or DataFrame containing the rgram results.

    Notes
    -----
    The `rgram` function is designed to analyse relationships between variables by
    generating a regression gram. It supports various binning styles and can optionally
    include OLS regression calculations and confidence intervals.

    Examples
    --------
    >>> import polars as pl
    >>> from rgram import rgram
    >>> df = pl.DataFrame({
    ...     "x": [1, 2, 3, 4, 5],
    ...     "y": [2, 4, 6, 8, 10]
    ... })
    >>> result = rgram(df, x="x", y="y")
    >>> result
    """

    def to_list(val):
        if val is None:
            return []

        return [val] if isinstance(val, str) else val

    x = to_list(x)
    hue = to_list(hue)
    keys = to_list(keys)
    idx_features = to_list(y)
    over_features = ["x_var"]

    if hue:
        idx_features.extend(hue)
        over_features.extend(hue)
    if keys:
        idx_features.extend(keys)

    if allow_negative_y == "auto":
        allow_negative_y = df[y].min().item() <= 0

    friedman_rot = 2 * (
        pl.col("x_val").quantile(0.75).sub(pl.col("x_val").quantile(0.25))
        / pl.len().pow(1 / 3)
    )
    data_range = pl.col("x_val").max() - pl.col("x_val").min()

    bin_style_dict = {
        "dist": (pl.col("x_val").rank(method="ordinal") * (data_range / friedman_rot))
        // pl.len(),
        "width": (pl.col("x_val") // friedman_rot),
        "unique": pl.col("x_val"),
        "int": pl.col("x_val").cast(int),
    }
    bin_calc = [bin_style_dict[bin_style].over(over_features).alias("rgram_bin")]

    rgram = (
        df.select(x + idx_features)
        .unpivot(on=x, index=idx_features, variable_name="x_var", value_name="x_val")
        .unpivot(
            on=y,
            index=["x_val", "x_var"] + hue,
            variable_name="y_var",
            value_name="y_val",
        )
        .filter(pl.col("x_var") != pl.col("y_var"))
        .with_columns(
            [
                pl.col("x_val").cast(float).alias("x_val"),
                pl.col("y_val").cast(float).alias("y_val"),
            ]
            + bin_calc
        )
        .with_columns(
            [
                metric(pl.col("y_val"))
                .over(over_features + ["rgram_bin"])
                .alias("y_pred_rgram")
            ]
        )
    )

    if ci_metric:
        ci_cols = ["y_pred_rgram_lci", "y_pred_rgram_uci"]

        ci_calc = [
            ci_m(pl.col("y_val")).over(over_features + ["rgram_bin"]).alias(alias)
            for ci_m, alias in zip(ci_metric, ci_cols)
        ]

        rgram = rgram.with_columns(ci_calc).with_columns(
            (
                []
                if allow_negative_y
                else [
                    pl.when(pl.col(col) < 0).then(0).otherwise(pl.col(col)).alias(col)
                    for col in ci_cols
                ]
            )
        )

    if calc_cum_sum:
        rgram = rgram.sort(by=["x_val"], descending=False).with_columns(
            pl.col("y_val").cum_sum().over(over_features)
        )

    if calc_ols:
        ols_calc = [
            pl.col("y_val")
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

        rgram = (
            rgram.with_columns(ols_calc)
            .with_columns(
                [
                    pl.col("ols_coef").struct.field("coef").alias("coef"),
                    pl.col("ols_coef").struct.field("const").alias("const"),
                ]
                + (
                    []
                    if allow_negative_y
                    else [
                        pl.when(pl.col("y_pred_ols") < 0)
                        .then(None)
                        .otherwise(pl.col("y_pred_ols"))
                        .alias("y_pred_ols")
                    ]
                )
            )
            .drop(["ols_coef"])
        )

    return rgram.sort(by=["x_val"])
