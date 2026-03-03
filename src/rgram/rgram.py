from __future__ import annotations

import polars as pl
from rgram.base import BaseUtils
from typing import Callable, Literal, Sequence, Optional, Union, Any


class Regressogram(BaseUtils):
    """
    Regressogram

    Binned regression estimator for one or more features and targets.

    Parameters
    ----------
    binning : {'dist', 'width', 'unique', 'int'}, default='dist'
        Binning strategy.
    agg : callable, default=mean
        Aggregation function for y in each bin.
    ci : tuple of callables, optional
        Tuple of lower/upper confidence interval functions.
    allow_negative_y : bool or 'auto', default='auto'
        Whether to allow negative y values in output.

    Methods
    -------
    fit(data, x, y, hue=None, keys=None)
        Fit the regressogram to data.
    transform()
        Return the fitted regressogram results.
    fit_transform(data, x, y, hue=None, keys=None)
        Fit to data and return results.
    """

    def __init__(
        self,
        binning: Literal["dist", "width", "unique", "int"] = "dist",
        agg: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
        ci: Optional[
            tuple[Callable[[pl.Expr], pl.Expr], Callable[[pl.Expr], pl.Expr]]
        ] = (
            lambda x: x.mean() - x.std(),
            lambda x: x.mean() + x.std(),
        ),
        allow_negative_y: Union[bool, Literal["auto"]] = "auto",
    ):
        """
        Construct a Regressogram instance.

        Parameters
        ----------
        binning : {'dist', 'width', 'unique', 'int'}, default='dist'
            Binning strategy.
        agg : callable, default=mean
            Aggregation function for y in each bin.
        ci : tuple of callables, optional
            Tuple of lower/upper confidence interval functions.
        allow_negative_y : bool or 'auto', default='auto'
            Whether to allow negative y values in output.
        """
        self.binning = binning
        self.agg = agg
        self.ci = ci
        self.allow_negative_y = allow_negative_y

    def _learn_bin_params(self, data: pl.LazyFrame) -> None:
        x_min, x_max, q25, q75, n = (
            data.select(
                pl.col("x_val").min().alias("min"),
                pl.col("x_val").max().alias("max"),
                pl.col("x_val").quantile(0.25).alias("q25"),
                pl.col("x_val").quantile(0.75).alias("q75"),
                pl.len(),
            )
            .collect()
            .row(0)
        )

        self._x_min = x_min
        self._x_max = x_max

        if self.binning in ("int", "unique"):
            self._min_bin = x_min
            self._max_bin = x_max

        if self.binning in ("width", "dist"):
            self._bin_width = 2 * (q75 - q25) / (n ** (1 / 3))

            # Handle edge case where bin_width is 0 (all x values identical)
            if self._bin_width == 0:
                self._bin_width = 1.0

            if self.binning == "dist":
                n_bins = max(1, int((x_max - x_min) // self._bin_width))

                if n_bins < 2:
                    qs = []  # no quantiles
                    self._bin_edges = []

                else:
                    qs = [i / n_bins for i in range(1, n_bins)]
                    self._bin_edges = list(
                        data.select(
                            [pl.col("x_val").quantile(q).alias(str(q)) for q in qs]
                        )
                        .collect()
                        .rows()[0]
                    )

                self._min_bin = 0
                self._max_bin = len(self._bin_edges)

            else:
                self._min_bin = 0
                self._max_bin = int((x_max - x_min) // self._bin_width)

    def _predict_bin_expr(self) -> pl.Expr:
        """
        Returns a Polars expression for binning x values.

        Returns
        -------
        pl.Expr
            The binning expression.
        """
        if self.binning == "width":
            bin_id = ((pl.col("x_val") - self._x_min) // self._bin_width).cast(int)

        elif self.binning == "dist":
            bin_id = pl.col("x_val").cut(breaks=self._bin_edges).rank(method="dense")

        elif self.binning == "int":
            bin_id = pl.col("x_val").cast(int)

        elif self.binning == "unique":
            return pl.col("x_val")

        else:
            raise ValueError(f"Unknown binning type: {self.binning}")

        # clip to edge bins
        return bin_id.clip(self._min_bin, self._max_bin)

    def fit(
        self,
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        hue: Optional[Union[str, Sequence[str]]] = None,
        keys: Optional[Union[str, Sequence[Any]]] = None,
    ) -> "Regressogram":
        """
        Fit the regressogram to the data.

        Supports flexible input similar to seaborn (e.g., kdeplot):
        - Provide DataFrame + column names (recommended for production)
        - Provide raw arrays/Series (convenient for exploration)

        Parameters
        ----------
        x : str or array-like
            Feature(s) to bin. Column name(s) if `data` provided, else array-like.
        y : str or array-like
            Target(s). Column name(s) if `data` provided, else array-like.
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If provided, x/y/hue/keys are treated as column names.
            If None, x/y/keys are treated as array-like values.
        hue : str or sequence of str, optional
            Optional grouping variable(s).
        keys : str or array-like, optional
            Additional grouping column(s).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _df_defined = data is not None

        data_lf, x_cols, y_cols, keys_cols = self._prepare_data(
            data=data, x=x, y=y, keys=keys
        )

        x_list = self._to_list(x_cols) or [x_cols]
        y_list = self._to_list(y_cols) or [y_cols]
        hue_list = self._to_list(hue) or [] if data is not None else []
        keys_list = self._to_list(keys_cols) or [keys_cols] if keys_cols else []

        idx_cols = (y_list or []) + (keys_list or []) + (hue_list or [])
        self.over_cols = ["x_var", "y_var"] + (hue_list or [])

        data = (
            data_lf.select(x_list + idx_cols)
            .unpivot(
                on=x_list, index=idx_cols, variable_name="x_var", value_name="x_val"
            )
            .unpivot(
                on=y_list,
                index=["x_val", "x_var"] + hue_list + keys_list,
                variable_name="y_var",
                value_name="y_val",
            )
            .filter(pl.col("x_var") != pl.col("y_var"))
            .with_columns([pl.col("y_val").cast(float)])
        )

        # learn bin parameters and assign bins
        self._learn_bin_params(data)
        data = data.with_columns(
            [self._predict_bin_expr().over(self.over_cols).alias("rgram_bin")]
        )

        # aggregate y values per bin
        data = data.with_columns(
            [
                self.agg(pl.col("y_val"))
                .over(self.over_cols + ["rgram_bin"])
                .alias("y_pred_rgram")
            ]
        )

        self._bin_to_y = data.select(["rgram_bin", "y_pred_rgram"]).unique().collect()

        self._regressogram_result = data

        return self

    def transform(self) -> pl.LazyFrame:
        if not hasattr(self, "_regressogram_result"):
            raise RuntimeError("You must call fit() before transform().")

        data = self._regressogram_result

        if self.allow_negative_y == "auto":
            data = data.with_columns(
                [
                    (pl.col("y_val").min() < 0)
                    .over(self.over_cols)
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
                ci_calc(pl.col("y_val").fill_null(pl.col("y_val").mean()))
                .over(self.over_cols + ["rgram_bin"])
                .alias(alias)
                for ci_calc, alias in zip(self.ci, ci_cols)
            ]

            data = data.with_columns(ci_exprs)
            data = data.with_columns([self._neg_y_helper(col) for col in ci_cols])

        cols_to_drop = ["allow_neg_y_val"]

        schema = data.collect_schema()
        cols_to_drop.extend([i for i in schema if i in ["x_var", "y_var"]])

        return data.sort(by=["x_val"]).drop(cols_to_drop)

        # def transform(self) -> pl.LazyFrame:
        #     """
        #     Return the regressogram results after fitting.

        #     Returns
        #     -------
        #     pl.LazyFrame
        #         The regressogram results.
        #     """
        #     if not hasattr(self, "_regressogram_result"):
        raise RuntimeError("You must call fit() before transform().")

        return self._regressogram_result

    def fit_transform(
        self,
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        hue: Optional[Union[str, Sequence[str]]] = None,
        keys: Optional[Union[str, Sequence[Any]]] = None,
    ) -> pl.LazyFrame:
        """
        Fit and return results in one call (recommended).

        Supports flexible input similar to seaborn:
        - Provide DataFrame + column names (recommended for production)
        - Provide raw arrays/Series (convenient for exploration)

        Parameters
        ----------
        x : str or array-like
            Feature(s) to bin. Column name(s) if `data` provided, else array-like.
        y : str or array-like
            Target(s). Column name(s) if `data` provided, else array-like.
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If provided, x/y/hue/keys are column names.
            If None, x/y/keys are array-like.
        hue : str or sequence of str, optional
            Optional grouping variable(s).
        keys : str or array-like, optional
            Additional grouping column(s).

        Returns
        -------
        pl.LazyFrame
            The regressogram results. Call `.collect()` to materialize.
        """
        self.fit(data=data, x=x, y=y, hue=hue, keys=keys)

        return self.transform()

    def predict(self, x: Union[Sequence[float], pl.Series]) -> pl.LazyFrame:
        if not hasattr(self, "_bin_to_y"):
            raise RuntimeError("Call fit() before predict().")

        lf = pl.DataFrame({"x_val": x}).lazy()

        lf = lf.with_columns(self._predict_bin_expr().alias("rgram_bin"))

        lf = lf.join(
            self._bin_to_y.lazy(),
            on="rgram_bin",
            how="left",
        )

        return lf.select("y_pred_rgram")

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
