from __future__ import annotations

import polars as pl
import polars_ols as pls  # noqa: F401

from rgram.base import BaseUtils
from typing import Sequence, Union, Optional, Any
from typing_extensions import Self


class KernelSmoother(BaseUtils):
    """
    KernelSmoother

    Epanechnikov kernel regression smoother for one-dimensional data.

    Parameters
    ----------
    n_eval_samples : int, default=100
        Number of evaluation points for the smoother.

    Methods
    -------
    fit(data, x, y, hue=None)
        Fit the kernel smoother to the data.
    transform()
        Return the kernel smoothed results after fitting.
    fit_transform(data, x, y, hue=None)
        Fit to data, then return the kernel smoothed results.
    """

    def __init__(
        self,
        n_eval_samples: int = 100,
        hue: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Construct a KernelSmoother instance.

        Parameters
        ----------
        n_eval_samples : int, default=100
            Number of evaluation points for the smoother.
        hue : sequence of str, optional
            Optional grouping variable(s).
        """
        super().__init__(hue=hue)
        self.n_eval_samples = n_eval_samples

    def _calculate_bandwidth(self, x_col: str) -> pl.Expr:
        """
        Calculate the kernel bandwidth using Silverman's rule of thumb.

        Parameters
        ----------
        x_col : str
            The feature column name.

        Returns
        -------
        pl.Expr
            The bandwidth expression.
        """
        # Compute std and IQR only once for efficiency
        std_expr = pl.col(x_col).std()
        iqr_expr = (pl.col(x_col).quantile(0.75) - pl.col(x_col).quantile(0.25)) / 1.34
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

        self._ks_result = ks

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
