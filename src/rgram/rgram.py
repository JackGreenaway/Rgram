from __future__ import annotations

import numpy as np
import polars as pl
from rgram.base import BaseUtils
from typing import Callable, Literal, Sequence, Optional, Union, Any


class Regressogram(BaseUtils):
    """
    Regressogram

    Binned regression estimator for one or more features and targets.
    Predicts using binned aggregation with optional confidence intervals.

    Parameters
    ----------
    binning : {'dist', 'width', 'none', 'int'}, default='dist'
        Binning strategy.
    agg : callable, default=mean
        Aggregation function for y in each bin.
    ci : tuple of callables, optional
        Tuple of lower/upper confidence interval functions.
    n_bins : int, optional
        Number of bins for 'dist' binning. If None, automatically calculated
        using Freedman-Diaconis rule. Ignored for other binning strategies.

    Methods
    -------
    fit(data, x, y)
        Learn bin parameters from training data.
    predict(x, return_ci=False)
        Predict binned regression values at new x points.
        Returns array or tuple with optional confidence intervals.
    fit_predict(data, x, y, return_ci=False)
        Fit and predict on training x values.
    """

    ALLOW_DUPLICATE_EDGES = True

    def __init__(
        self,
        *,
        binning: Literal["dist", "width", "none", "int"] = "dist",
        agg: Callable[[pl.Expr], pl.Expr] = lambda x: x.mean(),
        ci: Optional[
            tuple[Callable[[pl.Expr], pl.Expr], Callable[[pl.Expr], pl.Expr]]
        ] = (
            lambda x: x.mean() - x.std(),
            lambda x: x.mean() + x.std(),
        ),
        n_bins: Optional[int] = None,
    ):
        """
        Construct a Regressogram instance.

        Parameters
        ----------
        binning : {'dist', 'width', 'none', 'int'}, default='dist'
            Binning strategy.
        agg : callable, default=mean
            Aggregation function for y in each bin.
        ci : tuple of callables, optional
            Tuple of lower/upper confidence interval functions.
        n_bins : int, optional
            Number of bins for 'dist' binning. If None, automatically calculated
            using Freedman-Diaconis rule. Ignored for other binning strategies.
        """
        # Validate agg is callable
        if not callable(agg):
            raise TypeError(f"agg must be callable, got {type(agg).__name__}")

        # Validate ci is None or tuple of exactly 2 callables
        if ci is not None:
            if not isinstance(ci, tuple):
                raise TypeError(f"ci must be None or tuple, got {type(ci).__name__}")
            if len(ci) != 2:
                raise ValueError(
                    f"ci tuple must have exactly 2 elements, got {len(ci)}"
                )
            if not all(callable(c) for c in ci):
                raise TypeError("All elements in ci tuple must be callable")

        self.binning = binning
        self.agg = agg
        self.ci = ci
        self.n_bins = n_bins

        self._is_fitted = False

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

        if self.binning in ("int", "none"):
            self._min_bin = x_min
            self._max_bin = x_max

        if self.binning in ("width", "dist"):
            self._bin_width = 2 * (q75 - q25) / (n ** (1 / 3))

            # Handle edge case where bin_width is 0 (all x values identical)
            if self._bin_width == 0:
                self._bin_width = 1.0

            if self.binning == "dist":
                # Use user-specified n_bins or calculate from Freedman-Diaconis rule
                if self.n_bins is not None:
                    n_bins = max(1, self.n_bins)

                else:
                    n_bins = max(1, int((x_max - x_min) // self._bin_width))

                self._n_bins = n_bins

                self._min_bin = 0
                self._max_bin = n_bins - 1

                # Store bin edges from qcut for use during predict
                # This ensures stability by using the same bins at predict time
                self._bin_edges = (
                    data.select(
                        pl.col("x_val")
                        .qcut(
                            quantiles=self._n_bins,
                            allow_duplicates=self.ALLOW_DUPLICATE_EDGES,
                            include_breaks=True,
                        )
                        .struct.field("breakpoint")
                        .clip(pl.col("x_val").min(), pl.col("x_val").max())
                        .alias("bin_edge")
                    )
                    .unique()
                    .sort("bin_edge")
                    .collect()
                    .get_column("bin_edge")
                    .to_list()
                )

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
            if not hasattr(self, "_bin_edges"):
                raise RuntimeError(
                    "Bin edges not stored. This should not happen if fit() was called properly."
                )

            # left_closed=True means each bin is [left, right)
            bin_id = (
                pl.col("x_val")
                .cut(breaks=self._bin_edges, left_closed=True, include_breaks=True)
                .struct.field("breakpoint")
                .rank(method="dense")
                .cast(int)
            )

        elif self.binning == "int":
            bin_id = pl.col("x_val").cast(int)

        elif self.binning == "none":
            return pl.col("x_val")

        else:
            raise ValueError(f"Unknown binning type: {self.binning}")

        # clip to edge bins
        return bin_id.clip(self._min_bin, self._max_bin)

    def fit(
        self,
        x: Union[str, Any],
        y: Union[str, Any],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
    ) -> "Regressogram":
        """
        Learn bin parameters from training data (univariate only).

        Supports flexible input similar to seaborn (e.g., kdeplot):
        - Provide DataFrame + single column name (recommended for production)
        - Provide raw array/Series (convenient for exploration)

        Parameters
        ----------
        x : str or array-like
            Single feature to bin. Column name if `data` provided, else array-like.
        y : str or array-like
            Single target. Column name if `data` provided, else array-like.
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If provided, x/y are treated as column names (str).
            If None, x/y are treated as array-like values.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If x/y are sequences of column names (multivariate not supported).
            If x and y arrays have different lengths or are empty.
        TypeError
            If agg or ci are not callable, or if input contains complex numbers.

        Examples
        --------
        >>> import polars as pl
        >>> import numpy as np
        >>> from rgram import Regressogram
        >>>
        >>> # DataFrame mode
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> rgram = Regressogram().fit(data=df, x="x", y="y")
        >>>
        >>> # Array mode
        >>> rgram = Regressogram().fit(x=[1, 2, 3], y=[4, 5, 6])
        """
        # Validate univariate constraint - only when data is provided with column names
        # Lists/tuples are valid as data input, but list of column names must be single column
        if data is not None:
            # When data is provided, check if x/y are lists of column names
            if (
                isinstance(x, (list, tuple))
                and len(x) > 0
                and all(isinstance(item, str) for item in x)
            ):
                raise ValueError(
                    "fit only supports univariate (single feature) input. "
                    "When data is provided, x must be a single column name (str), not a list/tuple of column names."
                )
            if (
                isinstance(y, (list, tuple))
                and len(y) > 0
                and all(isinstance(item, str) for item in y)
            ):
                raise ValueError(
                    "fit only supports univariate (single target) input. "
                    "When data is provided, y must be a single column name (str), not a list/tuple of column names."
                )

        data_lf, x_cols, y_cols = self._prepare_data(data=data, x=x, y=y)

        x_list = self._to_list(x_cols) or [x_cols]
        y_list = self._to_list(y_cols) or [y_cols]

        idx_cols = y_list or []
        self.over_cols = ["x_var", "y_var"]

        data = (
            data_lf.select(x_list + idx_cols)
            .unpivot(
                on=x_list, index=idx_cols, variable_name="x_var", value_name="x_val"
            )
            .unpivot(
                on=y_list,
                index=["x_val", "x_var"],
                variable_name="y_var",
                value_name="y_val",
            )
            .filter(pl.col("x_var") != pl.col("y_var"))
            .with_columns([pl.col("y_val").cast(float)])
        )

        # Validate that x and y don't contain complex numbers (check after initial processing)
        try:
            sample = data.select(["x_val", "y_val"]).limit(1).collect()
            for col in ["x_val", "y_val"]:
                if col in sample.columns:
                    dtype = sample[col].dtype
                    if "complex" in str(dtype).lower():
                        raise TypeError(f"Complex numbers are not supported in {col}")
        except TypeError:
            raise
        except Exception:
            # If we can't check dtype early, it will fail later
            pass

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

        # Compute and store confidence intervals for each bin (if configured)
        select_cols = ["rgram_bin", "y_pred_rgram"]
        if self.ci:
            ci_cols = ["y_pred_rgram_lci", "y_pred_rgram_uci"]
            ci_exprs = [
                ci_calc(pl.col("y_val").fill_null(pl.col("y_val").mean()))
                .over(self.over_cols + ["rgram_bin"])
                .alias(alias)
                for ci_calc, alias in zip(self.ci, ci_cols)
            ]
            data = data.with_columns(ci_exprs)
            select_cols.extend(ci_cols)

        self._bin_to_y = data.select(select_cols).unique().collect()

        self._is_fitted = True

        return self

    def predict(
        self, x: Union[float, Sequence[float], pl.Series], return_ci: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Predict binned regression values at new x points (univariate only).

        Parameters
        ----------
        x : array-like or pl.Series
            Single feature values at which to predict.
        return_ci : bool, default=False
            If True, return confidence intervals along with predictions.
            Returns tuple (y_pred, y_ci_low, y_ci_high).
            If False, returns just the predictions array.

        Returns
        -------
        np.ndarray or tuple
            If return_ci=False: numpy array of predicted values (same length as x)
            If return_ci=True: tuple of (y_pred, y_ci_low, y_ci_high)
                y_pred: numpy array of predictions
                y_ci_low: numpy array of lower CI or None if ci not configured
                y_ci_high: numpy array of upper CI or None if ci not configured

        Raises
        ------
        RuntimeError
            If called before fit().
        TypeError
            If x is not array-like or contains non-numeric values.
        ValueError
            If x is empty or multivariate (must be univariate).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        # Validate input x is univariate
        x = self._validate_single_array(x, "x", allow_empty=False)

        lf = pl.DataFrame({"x_val": x}).lazy()

        lf = lf.with_columns(self._predict_bin_expr().alias("rgram_bin"))

        lf = lf.join(
            self._bin_to_y.lazy(),
            on="rgram_bin",
            how="left",
        )

        result_df = lf.select("y_pred_rgram").collect()
        y_pred = result_df["y_pred_rgram"].to_numpy()

        if not return_ci:
            return y_pred

        # Retrieve pre-computed CIs from _bin_to_y (stored at fit time)
        y_ci_low = None
        y_ci_high = None

        if self.ci:
            ci_cols = ["y_pred_rgram_lci", "y_pred_rgram_uci"]
            if ci_cols[0] in self._bin_to_y.columns:
                ci_result = lf.select(
                    [
                        "y_pred_rgram_lci",
                        "y_pred_rgram_uci",
                    ]
                ).collect()

                y_ci_low = ci_result["y_pred_rgram_lci"].to_numpy()
                y_ci_high = ci_result["y_pred_rgram_uci"].to_numpy()

        return y_pred, y_ci_low, y_ci_high

    def fit_predict(
        self,
        x: Union[str, Any],
        y: Union[str, Any],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        return_ci: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Fit and predict on training x values in one call (univariate only).

        Parameters
        ----------
        x : str or array-like
            Single feature column name if `data` provided, else single array-like data values.
        y : str or array-like
            Single target column name if `data` provided, else single array-like data values.
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If provided, x/y must be column names (str).
            If None, x/y must be array-like data (list, ndarray, Series).
        return_ci : bool, default=False
            If True, return confidence intervals along with predictions.

        Returns
        -------
        np.ndarray or tuple
            If return_ci=False: array of predictions at training x values
            If return_ci=True: tuple of (y_pred, y_ci_low, y_ci_high)

        Raises
        ------
        TypeError
            If x/y are not str when data is provided, or not array-like when data is None.
        ValueError
            If x/y are sequences of column names (univariate only).
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

        # Extract actual x values for predict
        if data is not None:
            # When data is provided, x must be a str (column name)
            if not isinstance(x, str):
                raise TypeError(
                    f"When data is provided, x must be a column name (str), "
                    f"got {type(x).__name__}"
                )
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            x = data.get_column(x).to_numpy()

        return self.predict(x=x, return_ci=return_ci)
