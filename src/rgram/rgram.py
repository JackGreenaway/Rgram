import polars as pl
import polars_ols as pls  # noqa: F401

from src.rgram.base import BaseUtils
from src.rgram.dclasses import OlsKws, CumsumKws

from typing import Callable, Literal, Sequence, Optional, Union, cast, List


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

        self.hue = cast(List[str], self._to_list(hue) or [])
        # super().__init__(hue=hue)

        self.binning = binning
        self.agg = agg
        self.ci = ci

        self.ols_kws = self._init_kws(var_input=ols, dataclass=OlsKws)
        self.cumsum_kws = self._init_kws(var_input=cumsum, dataclass=CumsumKws)

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

    def fit(self) -> "Regressogram":
        """
        Fit the regressogram to the data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        idx_cols = (self.y or []) + (self.keys or []) + (self.hue or [])
        over_cols = ["x_var", "y_var"] + (self.hue or [])

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

        data = data.with_columns(
            [
                self._bin_expr().over(over_cols).alias("rgram_bin"),
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
                .least_squares.ols(  # type: ignore
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

        return_data = data.sort(by=["x_val"])
        self._regressogram_result = return_data
        return self

    def transform(self) -> pl.LazyFrame:
        """
        Return the regressogram results after fitting.

        Returns
        -------
        pl.LazyFrame
            The regressogram results.
        """
        if not hasattr(self, "_regressogram_result"):
            raise RuntimeError("You must call fit() before transform().")
        return self._regressogram_result

    def fit_transform(self) -> pl.LazyFrame:
        """
        Fit to data, then return the regressogram results.

        Returns
        -------
        pl.LazyFrame
            The regressogram results.
        """
        self.fit()
        return self.transform()

    @property
    def ols_statistics_(self) -> pl.DataFrame | None:
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
