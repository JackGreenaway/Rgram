from __future__ import annotations

import numpy as np
import polars as pl
import warnings

from rgram.base import BaseUtils
from typing import Sequence, Union, Optional, Any, Literal


class KernelSmoother(BaseUtils):
    """
    KernelSmoother

    Epanechnikov kernel regression smoother for one-dimensional data.
    Predicts smooth values with optional confidence intervals.

    Parameters
    ----------
    bandwidth : {'silverman', 'scott', 'manual'}, default='silverman'
        Bandwidth selection method.
        - 'silverman': Silverman's rule of thumb (0.9 * min(std, IQR/1.34) * n^(-1/5))
        - 'scott': Scott's rule (1.06 * std * n^(-1/5))
        - 'manual': Use bandwidth_value parameter
    bandwidth_value : float, optional
        Manual bandwidth value. Required if bandwidth='manual'.
    bandwidth_adjust : float, default=1.0
        Multiplicative bandwidth adjustment factor.
    n_eval_samples : int, default=100
        Number of evaluation points for the smoother during fit_predict.

    Methods
    -------
    fit(x, y, data=None)
        Learn smoothing parameters from training data.
    predict(x_eval, return_ci=False)
        Predict smooth values at new x points.
        Returns array or tuple with optional confidence intervals.
    fit_predict(x, y, data=None, x_eval=None, return_ci=False)
        Fit and predict at evaluation points.
    """

    def __init__(
        self,
        bandwidth: Literal["silverman", "scott", "manual"] = "silverman",
        bandwidth_value: Optional[float] = None,
        bandwidth_adjust: float = 1.0,
        n_eval_samples: int = 100,
    ) -> None:
        """
        Construct a KernelSmoother instance.

        Parameters
        ----------
        bandwidth : {'silverman', 'scott', 'manual'}, default='silverman'
            Bandwidth selection method.
        bandwidth_value : float, optional
            Manual bandwidth value. Required if bandwidth='manual'.
        bandwidth_adjust : float, default=1.0
            Multiplicative bandwidth adjustment factor for the calculated bandwidth.
        n_eval_samples : int, default=100
            Number of evaluation points for generating smooth predictions during fit_predict.
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.bandwidth_value = bandwidth_value
        self.bandwidth_adjust = bandwidth_adjust
        self.n_eval_samples = n_eval_samples

        if bandwidth == "manual" and bandwidth_value is None:
            raise ValueError(
                "bandwidth_value must be specified when bandwidth='manual'"
            )

        self._is_fitted = False

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
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
    ) -> "KernelSmoother":
        """
        Fit the kernel smoother to the data.

        Supports flexible input similar to seaborn (e.g., kdeplot):
        - Provide DataFrame + column names (recommended for production)
        - Provide raw arrays/Series (convenient for exploration)

        Parameters
        ----------
        x : str or array-like
            Feature column name if `data` provided, else array-like (must be univariate).
        y : str or array-like
            Target column name if `data` provided, else array-like (must be univariate).
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If provided, x/y must be column names (str).
            If None, x/y must be array-like data (list, ndarray, Series).

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If x and y arrays have different lengths, are empty, or bandwidth is invalid.
        TypeError
            If input contains complex numbers.

        Examples
        --------
        >>> import polars as pl
        >>> from rgram import KernelSmoother
        >>>
        >>> # DataFrame mode
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
        >>> smoother = KernelSmoother().fit(data=df, x="x", y="y")
        >>>
        >>> # Array mode
        >>> smoother = KernelSmoother().fit(x=[1.0, 2.0, 3.0], y=[1.0, 4.0, 9.0])
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

        # Store fitted data for prediction and calculate/store bandwidth value
        self._x_col = x_col
        self._y_col = y_col
        self._is_fitted_data_lf = data_lf

        # Compute and store the bandwidth value for use in predict()
        bw_value_df = data_lf.select(bw).collect()
        self._bw_value = bw_value_df["h"][0]

        self._is_fitted = True

        return self

    def fit_predict(
        self,
        x: Union[str, Any],
        y: Union[str, Any],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        x_eval: Optional[Sequence[Any]] = None,
        return_ci: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Fit and predict at evaluation points in one call (univariate only).

        Parameters
        ----------
        x : str or array-like
            Single feature column name if `data` provided, else single array-like data.
        y : str or array-like
            Single target column name if `data` provided, else single array-like data.
        data : pl.DataFrame or pl.LazyFrame, optional
            Input data. If provided, x/y must be column names (str).
            If None, x/y must be array-like data.
        x_eval : array-like, optional
            Evaluation points for prediction. If None, uses training x values.
        return_ci : bool, default=False
            If True, return confidence intervals along with predictions.
            Note: Currently returns (y_pred, None, None) as CIs not yet implemented.

        Returns
        -------
        np.ndarray or tuple
            If return_ci=False: array of predictions at evaluation points
            If return_ci=True: tuple of (y_pred, y_ci_low, y_ci_high)

        Raises
        ------
        TypeError
            If x_eval is not array-like or contains non-numeric values (when provided).
        ValueError
            If x_eval is empty or if x/y are sequences (multivariate not supported).
        """
        # Validate univariate constraint when data is provided
        if data is not None:
            if isinstance(x, (list, tuple)):
                raise ValueError(
                    "fit_predict only supports univariate (single feature) input. "
                    "When data is provided, x must be a single column name (str), not a list/tuple of column names."
                )
            if isinstance(y, (list, tuple)):
                raise ValueError(
                    "fit_predict only supports univariate (single target) input. "
                    "When data is provided, y must be a single column name (str), not a list/tuple of column names."
                )

        self.fit(data=data, x=x, y=y)

        if x_eval is None:
            x_eval = self._is_fitted_data_lf.collect().get_column(self._x_col)
        else:
            # Validate user-provided x_eval
            x_eval = self._validate_single_array(x_eval, "x_eval", allow_empty=False)

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

        Raises
        ------
        RuntimeError
            If called before fit().
        TypeError
            If x_eval is not array-like or contains non-numeric values.
        ValueError
            If x_eval is empty.
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() before predict")

        # Validate input x_eval using BaseUtils validation
        x_eval = self._validate_single_array(x_eval, "x_eval", allow_empty=False)

        x_eval_col = "x_eval"
        x_train_col = "x_train"
        y_train_col = "y_train"

        pred_df = pl.DataFrame({x_eval_col: x_eval}).with_row_index("row_index").lazy()

        train_df = self._is_fitted_data_lf.select(
            [
                pl.col(self._x_col).alias(x_train_col),
                pl.col(self._y_col).alias(y_train_col),
            ]
        )

        bw = pl.lit(self._bw_value).alias("h")

        predictions = (
            pred_df.with_columns(bw)
            .join(train_df, how="cross")
            .with_columns(
                ((pl.col(x_train_col) - pl.col(x_eval_col)) / pl.col("h")).alias("u")
            )
            .with_columns(
                pl.when(pl.col("u").abs() <= 1)
                .then(0.75 * (1 - pl.col("u") ** 2))
                .otherwise(0.0)
                .alias("weight")
            )
            .group_by([x_eval_col, "row_index"], maintain_order=True)
            .agg(
                pl.when(pl.col("weight").sum() > 0)
                .then(
                    (pl.col(y_train_col) * pl.col("weight")).sum()
                    / pl.col("weight").sum()
                )
                .otherwise(None)
                .alias("y_kernel")
            )
            .sort("row_index")
            .collect()
        )

        y_pred = predictions.get_column("y_kernel").to_numpy()

        if not return_ci:
            return y_pred

        # For kernel smoother, confidence intervals would require bootstrap or analytically computed
        # Currently not implemented, return None

        warnings.warn(
            "Confidence intervals are not implemented for KernelSmoother yet. "
            "Returning (y_pred, None, None).",
            UserWarning,
            stacklevel=2,
        )

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
