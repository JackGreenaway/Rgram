from __future__ import annotations

import polars as pl
import polars_ols as pls  # noqa: F401

from rgram.base import BaseUtils
from typing import Sequence, Union, Optional, Any, Literal
from typing_extensions import Self


class KernelSmoother(BaseUtils):
    """
    KernelSmoother

    Epanechnikov kernel regression smoother for one-dimensional data.

    Parameters
    ----------
    n_eval_samples : int, default=100
        Number of evaluation points for the smoother.
    bandwidth : {'silverman', 'scott', 'manual'}, default='silverman'
        Bandwidth selection method.
        - 'silverman': Silverman's rule of thumb (0.9 * min(std, IQR/1.34) * n^(-1/5))
        - 'scott': Scott's rule (1.06 * std * n^(-1/5))
        - 'manual': Use bandwidth_value parameter
    bandwidth_value : float, optional
        Manual bandwidth value. Required if bandwidth='manual'.

    Methods
    -------
    fit(data, x, y, hue=None)
        Fit the kernel smoother to the data.
    transform()
        Return the kernel smoothed results after fitting.
    fit_transform(data, x, y, hue=None)
        Fit to data, then return the kernel smoothed results.
    predict(x_new)
        Predict on new data points.
    """

    def __init__(
        self,
        n_eval_samples: int = 100,
        bandwidth: Literal["silverman", "scott", "manual"] = "silverman",
        bandwidth_value: Optional[float] = None,
        hue: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Construct a KernelSmoother instance.

        Parameters
        ----------
        n_eval_samples : int, default=100
            Number of evaluation points for the smoother.
        bandwidth : {'silverman', 'scott', 'manual'}, default='silverman'
            Bandwidth selection method.
        bandwidth_value : float, optional
            Manual bandwidth value. Required if bandwidth='manual'.
        hue : sequence of str, optional
            Optional grouping variable(s).
        """
        super().__init__(hue=hue)
        self.n_eval_samples = n_eval_samples
        self.bandwidth = bandwidth
        self.bandwidth_value = bandwidth_value

        if bandwidth == "manual" and bandwidth_value is None:
            raise ValueError(
                "bandwidth_value must be specified when bandwidth='manual'"
            )

    def _calculate_bandwidth(self, x_col: str) -> pl.Expr:
        """
        Calculate the kernel bandwidth based on the selected method.

        Parameters
        ----------
        x_col : str
            The feature column name.

        Returns
        -------
        pl.Expr
            The bandwidth expression.
        """
        if self.bandwidth == "manual":
            bw = pl.lit(self.bandwidth_value).alias("h")
        elif self.bandwidth == "scott":
            # Scott's rule: 1.06 * std * n^(-1/5)
            std_expr = pl.col(x_col).std()
            bw = self._over_function(1.06 * std_expr * (pl.len() ** (-1 / 5))).alias(
                "h"
            )
        else:  # silverman (default)
            # Silverman's rule: 0.9 * min(std, IQR/1.34) * n^(-1/5)
            std_expr = pl.col(x_col).std()
            iqr_expr = (
                pl.col(x_col).quantile(0.75) - pl.col(x_col).quantile(0.25)
            ) / 1.34
            bw = self._over_function(
                0.9 * pl.min_horizontal([std_expr, iqr_expr]) * (pl.len() ** (-1 / 5))
            ).alias("h")

        return bw

    def _calculate_x_eval(self, x_col: str) -> pl.Expr:
        """
        Calculate the evaluation points for the kernel smoother.

        Parameters
        ----------
        x_col : str
            The feature column name.

        Returns
        -------
        pl.Expr
            The evaluation points expression.
        """
        x_eval = self._over_function(
            pl.linear_spaces(
                pl.col(x_col).min(),
                pl.col(x_col).max(),
                self.n_eval_samples,
                as_array=True,
            )
        ).alias("x_eval")

        return x_eval

    def fit(
        self,
        y: Union[str, Sequence[Any]],
        x: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        hue: Optional[Sequence[str]] = None,
    ) -> Self:
        """
        Fit the kernel smoother to the data.

        Supports flexible input similar to seaborn (e.g., kdeplot):
        - Provide DataFrame + column names (recommended for production)
        - Provide raw arrays/Series (convenient for exploration)

        Parameters
        ----------
        x : str or array-like
            Feature column. Column name if `data` provided, else array-like.
        y : str or array-like
            Target column. Column name if `data` provided, else array-like.
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If provided, x/y/hue are column names.
            If None, x/y are array-like.
        hue : sequence of str, optional
            Optional grouping variable(s). Overrides hue from __init__ if provided.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Prepare data: convert arrays to DataFrame if needed
        data_lf, x_cols, y_cols, _ = self._prepare_data(data=data, x=x, y=y, keys=None)

        # Update hue if provided, otherwise use initialized hue
        if hue is not None:
            self.hue = self._to_list(hue) or []

        # Extract first column name from x/y (single feature/target for KernelSmoother)
        x_col = x_cols if isinstance(x_cols, str) else x_cols[0]
        y_col = y_cols if isinstance(y_cols, str) else y_cols[0]

        bw = self._calculate_bandwidth(x_col)
        x_eval = self._calculate_x_eval(x_col)

        ks = (
            data_lf.with_columns([bw, x_eval])
            .explode("x_eval")
            .with_columns(
                [
                    ((pl.col("x_eval") - pl.col(x_col)) / pl.col("h")).alias("u"),
                ]
            )
            .with_columns(
                [
                    (0.75 * (1 - (pl.col("u") ** 2))).alias("weight"),
                ]
            )
            .filter(pl.col("u").abs() <= 1)
            .group_by(["x_eval"] + self.hue)
            .agg(
                [
                    # epanechnikov kernel
                    (
                        (pl.col(y_col) * pl.col("weight")).sum()
                        / pl.col("weight").sum()
                    ).alias("y_kernel")
                ]
            )
            .sort(by="x_eval")
        )

        # Store fitted data for prediction and calculate/store bandwidth value
        self._x_col = x_col
        self._y_col = y_col
        self._hue = self.hue
        self._fitted_data_lf = data_lf
        self._ks_result = ks

        # Compute and store the bandwidth value for use in predict()
        bw_value_df = data_lf.select(bw).collect()
        self._bw_value = bw_value_df["h"][0]

        return self

    def transform(self) -> pl.LazyFrame:
        """
        Return the kernel smoothed results after fitting.

        Returns
        -------
        pl.LazyFrame
            The kernel smoothed results.
        """
        if not hasattr(self, "_ks_result"):
            raise RuntimeError("You must call fit() before transform().")

        return self._ks_result

    def fit_transform(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame],
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        hue: Optional[Sequence[str]] = None,
    ) -> pl.LazyFrame:
        """
        Fit and return results in one call (recommended).

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input data.
        x : str or array-like
            Feature column. Column name if `data` provided, else array-like.
        y : str or array-like
            Target column. Column name if `data` provided, else array-like.
        hue : sequence of str, optional
            Optional grouping variable(s).

        Returns
        -------
        pl.LazyFrame
            The kernel smoothed results. Call `.collect()` to materialize.
        """
        self.fit(data=data, x=x, y=y, hue=hue)

        return self.transform()

    def predict(self, x_new: Union[Sequence[float], pl.Series]) -> pl.LazyFrame:
        """
        Predict smooth values at new x points.

        Parameters
        ----------
        x_new : array-like or pl.Series
            New x values at which to predict.

        Returns
        -------
        pl.LazyFrame
            Predictions with columns ['x_eval', 'y_kernel'].
        """
        if not hasattr(self, "_ks_result"):
            raise RuntimeError("Call fit() before predict().")

        # Create prediction dataframe directly from input
        pred_df = pl.DataFrame({self._x_col: x_new}).lazy()

        # Use the bandwidth learned during fit
        bw = pl.lit(self._bw_value).alias("h")

        predictions = (
            pred_df.with_columns(bw)
            .join(self._fitted_data_lf.select([self._x_col, self._y_col]), how="cross")
            .rename({self._x_col: "x_new", (self._x_col + "_right"): self._x_col})
            .with_columns(
                ((pl.col(self._x_col) - pl.col("x_new")) / pl.col("h")).alias("u")
            )
            .with_columns((0.75 * (1 - (pl.col("u") ** 2))).alias("weight"))
            .filter(pl.col("u").abs() <= 1)
            .group_by("x_new")
            .agg(
                (
                    (pl.col(self._y_col) * pl.col("weight")).sum()
                    / pl.col("weight").sum()
                ).alias("y_kernel")
            )
            .rename({"x_new": "x_eval"})
            .sort(by="x_eval")
        )

        return predictions
