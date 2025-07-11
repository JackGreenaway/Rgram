import polars as pl
import polars_ols as pls  # noqa: F401

from typing import Callable, Literal, Sequence, Optional, Union
from dataclasses import dataclass


class BaseUtils:
    """
    BaseUtils

    Utility base class for DataFrame-related utilities, such as list conversion and group-over operations.

    Methods
    -------
    _to_list(item)
        Convert a string or sequence to a list, or return None.
    _over_function(x)
        Apply a Polars expression over the 'hue' columns if present.
    """

    @staticmethod
    def _to_list(item: Optional[Union[str, Sequence]]) -> Optional[list]:
        """
        Convert a string or sequence to a list, or return None.

        Parameters
        ----------
        item : str, sequence, or None
            The item to convert.

        Returns
        -------
        list or None
            The converted list or None if input is None.
        """
        if item is None:
            return None
        elif isinstance(item, str):
            return [item]
        return list(item)

    def _over_function(self, x: pl.Expr) -> pl.Expr:
        """
        Apply a Polars expression over the 'hue' columns if present.

        Parameters
        ----------
        x : pl.Expr
            The Polars expression to apply.

        Returns
        -------
        pl.Expr
            The possibly grouped expression.
        """
        if hasattr(self, "hue") and self.hue:
            return x.over(self.hue)
        return x


@dataclass
class OlsKws:
    calc_ols: bool = True
    order: int = 1
    add_intercept: bool = True
    ols_y_target: Literal["y_val", "y_pred_rgram", "y_val_cum_sum"] = "y_val"


@dataclass
class CumsumKws:
    calc_cum_sum: bool = True
    reverse: bool = False


class Regressogram(BaseUtils):
    """
    Regressogram

    Binned regression and visualisation for one or more features and targets.

    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data.
    x : str or sequence of str
        Feature(s) to bin.
    y : str or sequence of str
        Target(s).
    hue : str or sequence of str, optional
        Optional grouping variable(s).
    binning : {'dist', 'width', 'all', 'int'}, default='dist'
        Binning strategy.
    agg : callable, default=mean
        Aggregation function for y in each bin.
    ci : tuple of callables, optional
        Tuple of lower/upper confidence interval functions.
    ols : OlsKws or dict, optional
        OLS regression options.
    cumsum : CumsumKws or dict, optional
        Cumulative sum options.
    allow_negative_y : bool or 'auto', default='auto'
        Whether to allow negative y values in output.
    keys : str or sequence of str, optional
        Additional grouping columns.

    Methods
    -------
    calculate()
        Compute the regressogram and return a LazyFrame with results.
    ols_statistics_
        Returns OLS statistics if OLS was computed.
    """

    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame],
        x: Union[str, Sequence[str]],
        y: Union[str, Sequence[str]],
        hue: Optional[Union[str, Sequence[str]]] = None,
        binning: Literal["dist", "width", "all", "int"] = "dist",
        agg: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
        ci: Optional[
            tuple[Callable[[pl.Expr], pl.Expr], Callable[[pl.Expr], pl.Expr]]
        ] = (
            lambda x: x.mean() - x.std(),
            lambda x: x.mean() + x.std(),
        ),
        ols: Optional[Union[OlsKws, dict]] = None,
        cumsum: Optional[Union[CumsumKws, dict]] = None,
        allow_negative_y: Union[bool, Literal["auto"]] = "auto",
        keys: Optional[Union[str, Sequence[str]]] = None,
    ):
        """
        Construct a Regressogram instance.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input data.
        x : str or sequence of str
            Feature(s) to bin.
        y : str or sequence of str
            Target(s).
        hue : str or sequence of str, optional
            Optional grouping variable(s).
        binning : {'dist', 'width', 'all', 'int'}, default='dist'
            Binning strategy.
        agg : callable, default=mean
            Aggregation function for y in each bin.
        ci : tuple of callables, optional
            Tuple of lower/upper confidence interval functions.
        ols : OlsKws or dict, optional
            OLS regression options.
        cumsum : CumsumKws or dict, optional
            Cumulative sum options.
        allow_negative_y : bool or 'auto', default='auto'
            Whether to allow negative y values in output.
        keys : str or sequence of str, optional
            Additional grouping columns.
        """
        self.data = data.lazy()
        self.x = self._to_list(x)
        self.y = self._to_list(y)
        self.hue = self._to_list(hue)
        self.binning = binning
        self.agg = agg
        self.ci = ci

        self.ols_kws = (
            OlsKws(**ols)
            if isinstance(ols, dict)
            else ols
            if isinstance(ols, OlsKws)
            else OlsKws()
        )
        self.cumsum_kws = (
            CumsumKws(**cumsum)
            if isinstance(cumsum, dict)
            else cumsum
            if isinstance(cumsum, CumsumKws)
            else CumsumKws()
        )

        self.allow_negative_y = allow_negative_y
        self.keys = self._to_list(keys)

    def _bin_expr(self) -> pl.Expr:
        """
        Returns a Polars expression for binning x values.

        Returns
        -------
        pl.Expr
            The binning expression.
        """
        # Cache quantiles and range to avoid recomputation
        q75 = pl.col("x_val").quantile(0.75)
        q25 = pl.col("x_val").quantile(0.25)
        data_range = pl.col("x_val").max() - pl.col("x_val").min()
        freedman_rot = 2 * (q75 - q25) / (pl.len() ** (1 / 3))

        if self.binning == "dist":
            return (
                pl.col("x_val").rank(method="ordinal")
                * (data_range / freedman_rot)
                // pl.len()
            ).floor()
        elif self.binning == "width":
            return (pl.col("x_val") // freedman_rot).floor()
        elif self.binning == "all":
            return pl.col("x_val")
        elif self.binning == "int":
            return pl.col("x_val").cast(int)
        else:
            raise ValueError(f"Unknown binning type: {self.binning}")

    def calculate(self) -> pl.LazyFrame:
        """
        Compute the regressogram and return a LazyFrame with results.

        Returns
        -------
        pl.LazyFrame
            The regressogram results.
        """
        idx_cols = (self.y or []) + (self.keys or []) + (self.hue or [])
        over_cols = ["x_var", "y_var"] + (self.hue or [])

        # Only select necessary columns
        data = (
            self.data.select((self.x or []) + idx_cols)
            .unpivot(
                on=self.x, index=idx_cols, variable_name="x_var", value_name="x_val"
            )
            .unpivot(
                on=self.y,
                index=["x_val", "x_var"] + (self.hue or []) + (self.keys or []),
                variable_name="y_var",
                value_name="y_val",
            )
            .filter(pl.col("x_var") != pl.col("y_var"))
            .with_columns([pl.col("y_val").cast(float)])
        )

        # Combine with_columns to reduce scans
        data = data.with_columns(
            [
                self._bin_expr().over(over_cols).alias("rgram_bin"),
                # Optionally, precompute mean for y_val if used multiple times
            ]
        )

        data = data.with_columns(
            [
                self.agg(pl.col("y_val"))
                .over(over_cols + ["rgram_bin"])
                .alias("y_pred_rgram")
            ]
        )

        if self.ci or self.ols_kws.calc_ols:
            if self.allow_negative_y == "auto":
                data = data.with_columns(
                    [
                        (pl.col("y_val").min() < 0)
                        .over(over_cols)
                        .alias("allow_neg_y_val")
                    ]
                )
            else:
                data = data.with_columns(
                    [pl.lit(self.allow_negative_y).alias("allow_neg_y_val")]
                )

        if self.ci:
            ci_cols = ["y_pred_rgram_lci", "y_pred_rgram_uci"]
            ci_exprs = [
                metric(pl.col("y_val").fill_null(pl.col("y_val").mean()))
                .over(over_cols + ["rgram_bin"])
                .alias(alias)
                for metric, alias in zip(self.ci, ci_cols)
            ]
            data = data.with_columns(ci_exprs)
            data = data.with_columns([self._neg_y_helper(col) for col in ci_cols])

        if self.ols_kws.calc_ols:
            ols_exprs = [
                pl.col(self.ols_kws.ols_y_target)
                .least_squares.ols(
                    *[
                        (pl.col("x_val") ** i).alias(
                            "x_val" if i == 1 else f"x_val**{i}"
                        )
                        for i in range(1, self.ols_kws.order + 1)
                    ],
                    mode=mode,
                    add_intercept=self.ols_kws.add_intercept,
                    null_policy="drop",
                )
                .over(over_cols)
                .alias(alias)
                for mode, alias in [
                    ("statistics", "ols_statistics"),
                    ("predictions", "y_pred_ols"),
                ]
            ]
            # Only collect statistics once
            self._ols_statistics = (
                data.select(over_cols + [ols_exprs[0]])
                .unique()
                .unnest("ols_statistics")
            ).collect()

            data = (
                data.with_columns([ols_exprs[1]])
                .with_columns([self._neg_y_helper("y_pred_ols")])
                .drop(["allow_neg_y_val"])
            )

        if self.cumsum_kws.calc_cum_sum:
            data = data.sort(by=["x_val"]).with_columns(
                pl.col("y_val")
                .cum_sum(reverse=self.cumsum_kws.reverse)
                .over(over_cols)
                .alias("y_val_cum_sum")
            )

        return data.sort(by=["x_val"])

    @property
    def ols_statistics_(self) -> pl.DataFrame:
        """
        OLS statistics

        Returns
        -------
        pl.DataFrame or None
            Returns OLS statistics if OLS was computed, else None.
        """
        return getattr(self, "_ols_statistics", None)

    @staticmethod
    def _neg_y_helper(col: str) -> pl.Expr:
        """
        Helper to set negative y values to null if not allowed.

        Parameters
        ----------
        col : str
            The column name to check.

        Returns
        -------
        pl.Expr
            The expression with negative values set to null if not allowed.
        """
        return (
            pl.when(pl.col("allow_neg_y_val"))
            .then(pl.col(col))
            .when(pl.col(col) < 0)
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )


@dataclass
class KernelSmoothKws:
    calc_kws: bool = True
    n_eval_points: int = 150


class KernelSmoother(BaseUtils):
    """
    KernelSmoother

    Epanechnikov kernel regression smoother for one-dimensional data.

    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data.
    x : str
        Feature column.
    y : str
        Target column.
    hue : sequence of str, optional
        Optional grouping variable(s).
    n_eval_samples : int, default=100
        Number of evaluation points for the smoother.

    Methods
    -------
    calculate()
        Compute the kernel smoother and return a LazyFrame with results.
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        x: str,
        y: str,
        hue: Sequence[str] | None = None,
        n_eval_samples: int = 100,
    ) -> None:
        """
        Construct a KernelSmoother instance.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input data.
        x : str
            Feature column.
        y : str
            Target column.
        hue : sequence of str, optional
            Optional grouping variable(s).
        n_eval_samples : int, default=100
            Number of evaluation points for the smoother.
        """
        self.data = data.lazy()
        self.x = self._to_list(x)
        self.y = self._to_list(y)
        self.hue = self._to_list(hue)
        self.n_eval_samples = n_eval_samples

    def _calculate_bandwidth(self) -> pl.Expr:
        """
        Calculate the kernel bandwidth using Silverman's rule of thumb.

        Returns
        -------
        pl.Expr
            The bandwidth expression.
        """
        # Compute std and IQR only once for efficiency
        std_expr = pl.col(self.x).std()
        iqr_expr = (
            pl.col(self.x).quantile(0.75) - pl.col(self.x).quantile(0.25)
        ) / 1.34
        bw = self._over_function(
            0.9 * pl.min_horizontal([std_expr, iqr_expr]) * (pl.len() ** (-1 / 5))
        ).alias("h")
        return bw

    def _calculate_x_eval(self) -> pl.Expr:
        """
        Calculate the evaluation points for the kernel smoother.

        Returns
        -------
        pl.Expr
            The evaluation points expression.
        """
        x_eval = self._over_function(
            pl.linear_spaces(
                pl.col(self.x).min(),
                pl.col(self.x).max(),
                self.n_eval_samples,
                as_array=True,
            )
        ).alias("x_eval")

        return x_eval

    def calculate(self) -> pl.LazyFrame:
        """
        Compute the kernel smoother and return a LazyFrame with results.

        Returns
        -------
        pl.LazyFrame
            The kernel smoothed results.
        """
        bw = self._calculate_bandwidth()
        x_eval = self._calculate_x_eval()

        ks = (
            self.data.with_columns([bw, x_eval])
            .explode("x_eval")
            .with_columns(
                [
                    ((pl.col("x_eval") - pl.col(self.x)) / pl.col("h")).alias("u"),
                ]
            )
            .with_columns(
                [
                    (0.75 * (1 - (pl.col("u") ** 2))).alias("weight"),
                ]
            )
            .filter(pl.col("u").abs() <= 1)
            .group_by(["x_eval"] + (self.hue or []))
            .agg(
                [
                    # Epanechnikov kernel
                    (
                        (pl.col(self.y) * pl.col("weight")).sum()
                        / pl.col("weight").sum()
                    ).alias("y_kernel")
                ]
            )
            .sort(by="x_eval")
        )

        return ks
