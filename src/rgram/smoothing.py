from __future__ import annotations

import numpy as np
import polars as pl

from rgram.base import BaseUtils
from typing import Sequence, Union, Optional, Any, Literal
from typing_extensions import Self


class KernelSmoother(BaseUtils):
    """
    KernelSmoother

    Epanechnikov kernel regression smoother for one-dimensional data.
    Predicts smooth values with optional confidence intervals.

    Parameters
    ----------
    n_eval_samples : int, default=100
        Number of evaluation points for the smoother during fit_predict.
    bandwidth : {'silverman', 'scott', 'manual'}, default='silverman'
        Bandwidth selection method.
        - 'silverman': Silverman's rule of thumb (0.9 * min(std, IQR/1.34) * n^(-1/5))
        - 'scott': Scott's rule (1.06 * std * n^(-1/5))
        - 'manual': Use bandwidth_value parameter
    bandwidth_value : float, optional
        Manual bandwidth value. Required if bandwidth='manual'.

    Methods
    -------
    fit(data, x, y)
        Learn smoothing parameters from training data.
    predict(x_new, return_ci=False)
        Predict smooth values at new x points.
        Returns array or tuple with optional confidence intervals.
    fit_predict(data, x, y, return_ci=False)
        Fit and predict at n_eval_samples evaluation points.
    """

    def __init__(
        self,
        bandwidth: Literal["silverman", "scott", "manual"] = "silverman",
        bandwidth_value: Optional[float] = None,
        bandwidth_adjust: float = 1.0,
    ) -> None:
        """
        Construct a KernelSmoother instance.

        Parameters
        ----------
        bandwidth : {'silverman', 'scott', 'manual'}, default='silverman'
            Bandwidth selection method.
        bandwidth_value : float, optional
            Manual bandwidth value. Required if bandwidth='manual'.
        bandwidth_adjust : float
            Multiplicative bandwidth adjust value.
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.bandwidth_value = bandwidth_value
        self.bandwidth_adjust = bandwidth_adjust

        if bandwidth == "manual" and bandwidth_value is None:
            raise ValueError(
                "bandwidth_value must be specified when bandwidth='manual'"
            )

        self._fitted = False

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
            bw = pl.lit(self.bandwidth_value)

        elif self.bandwidth == "scott":
            # Scott's rule: 1.06 * std * n^(-1/5)
            std_expr = pl.col(x_col).std()
            bw = self._over_function(1.06 * std_expr * (pl.len() ** (-1 / 5)))

        else:  # silverman (default)
            # Silverman's rule: 0.9 * min(std, IQR/1.34) * n^(-1/5)
            std_expr = pl.col(x_col).std()
            iqr_expr = (
                pl.col(x_col).quantile(0.75) - pl.col(x_col).quantile(0.25)
            ) / 1.34
            bw = self._over_function(
                0.9 * pl.min_horizontal([std_expr, iqr_expr]) * (pl.len() ** (-1 / 5))
            )

        return (bw * self.bandwidth_adjust).alias("h")

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
            Input data. If provided, x/y are column names.
            If None, x/y are array-like.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate bandwidth parameter
        valid_bandwidths = ("silverman", "scott", "manual")
        if self.bandwidth not in valid_bandwidths:
            raise ValueError(
                f"bandwidth must be one of {valid_bandwidths}, got '{self.bandwidth}'"
            )

        # Prepare data: convert arrays to DataFrame if needed
        data_lf, x_cols, y_cols = self._prepare_data(data=data, x=x, y=y)

        # Extract first column name from x/y (single feature/target for KernelSmoother)
        x_col = x_cols if isinstance(x_cols, str) else x_cols[0]
        y_col = y_cols if isinstance(y_cols, str) else y_cols[0]

        bw = self._calculate_bandwidth(x_col)
        # x_eval = self._calculate_x_eval(x_col)

        # ks = (
        #     data_lf.with_columns([bw, x_eval])
        #     .explode("x_eval")
        #     .with_columns(
        #         [
        #             ((pl.col("x_eval") - pl.col(x_col)) / pl.col("h")).alias("u"),
        #         ]
        #     )
        #     .with_columns(
        #         [
        #             (0.75 * (1 - (pl.col("u") ** 2))).alias("weight"),
        #         ]
        #     )
        #     .filter(pl.col("u").abs() <= 1)
        #     .group_by("x_eval")
        #     .agg(
        #         [
        #             # epanechnikov kernel
        #             (
        #                 (pl.col(y_col) * pl.col("weight")).sum()
        #                 / pl.col("weight").sum()
        #             ).alias("y_kernel")
        #         ]
        #     )
        #     .sort(by="x_eval")
        # )

        # Store fitted data for prediction and calculate/store bandwidth value
        self._x_col = x_col
        self._y_col = y_col
        self._fitted_data_lf = data_lf
        # self._ks_result = ks

        # Compute and store the bandwidth value for use in predict()
        bw_value_df = data_lf.select(bw).collect()
        self._bw_value = bw_value_df["h"][0]

        self._fitted = True

        return self

    def fit_predict(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame],
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        x_eval: Optional[Sequence[Any]] = None,
        return_ci: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Fit and predict at evaluation points in one call.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input data.
        x : str or array-like
            Feature column. Column name if `data` provided, else array-like.
        y : str or array-like
            Target column. Column name if `data` provided, else array-like.
        return_ci : bool, default=False
            If True, return confidence intervals along with predictions.

        Returns
        -------
        np.ndarray or tuple
            If return_ci=False: array of predictions at evaluation points
            If return_ci=True: tuple of (y_pred, y_ci_low, y_ci_high)
        """
        self.fit(data=data, x=x, y=y)

        if x_eval is None:
            x_eval = self._fitted_data_lf.collect().get_column(self._x_col)

        return self.predict(x_eval, return_ci=return_ci)

    def predict(
        self, x_eval: Union[Sequence[float], pl.Series], return_ci: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Predict smooth values at new x points.

        Parameters
        ----------
        x_eval : array-like or pl.Series
            New x values at which to predict.
        return_ci : bool, default=False
            If True, return confidence intervals along with predictions.
            Note: Currently, KernelSmoother returns None for CIs.

        Returns
        -------
        np.ndarray or tuple
            If return_ci=False: numpy array of predictions (same length as x_eval)
            If return_ci=True: tuple of (y_pred, y_ci_low, y_ci_high)
                y_ci_low and y_ci_high are None (not yet implemented for kernel smoother)
        """
        if not self._fitted:
            raise RuntimeError("You must call fit() before predict")

        x_eval = self._validate_arraylike_input(x_eval)

        # Create prediction dataframe directly from input
        pred_df = (
            pl.DataFrame({self._x_col: x_eval}).with_row_index(name="row_index").lazy()
        )

        # Use the bandwidth learned during fit
        bw = pl.lit(self._bw_value).alias("h")

        predictions = (
            pred_df.with_columns(bw)
            .join(self._fitted_data_lf.select([self._x_col, self._y_col]), how="cross")
            .rename({self._x_col: "x_eval", (self._x_col + "_right"): self._x_col})
            .with_columns(
                ((pl.col(self._x_col) - pl.col("x_eval")) / pl.col("h")).alias("u")
            )
            .with_columns((0.75 * (1 - (pl.col("u") ** 2))).alias("weight"))
            .filter(pl.col("u").abs() <= 1)
            .group_by(["x_eval", "row_index"], maintain_order=True)
            .agg(
                (
                    (pl.col(self._y_col) * pl.col("weight")).sum()
                    / pl.col("weight").sum()
                ).alias("y_kernel")
            )
            .drop("row_index")
            .collect()
        )

        y_pred = predictions["y_kernel"].to_numpy()

        if not return_ci:
            return y_pred

        # For kernel smoother, confidence intervals would require bootstrap or analytically computed
        # Currently not implemented, return None
        return y_pred, None, None

    @staticmethod
    def _validate_arraylike_input(input: Any) -> np.typing.ArrayLike:
        # Validate input type
        try:
            # Try to convert to numeric array
            if isinstance(input, pl.Series):
                x_array = input.to_numpy()
            else:
                x_array = np.asarray(input)
            # Check if numeric
            if not np.issubdtype(x_array.dtype, np.number):
                raise TypeError(f"x_eval must be numeric, got {x_array.dtype}")

            return x_array

        except (ValueError, TypeError) as e:
            raise TypeError(f"x_eval must contain numeric values, got: {e}")
