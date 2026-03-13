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

        self.binning = binning
        self.agg = agg
        self.ci = ci
        self.n_bins = n_bins

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
                            allow_duplicates=True,
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
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
    ) -> "Regressogram":
        """
        Learn bin parameters from training data.

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
            Input data. If provided, x/y are treated as column names.
            If None, x/y are treated as array-like values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate ci is None or tuple of exactly 2 callables
        if self.ci is not None:
            if not isinstance(self.ci, tuple):
                raise TypeError(
                    f"ci must be None or tuple, got {type(self.ci).__name__}"
                )
            if len(self.ci) != 2:
                raise ValueError(
                    f"ci tuple must have exactly 2 elements, got {len(self.ci)}"
                )
            if not all(callable(c) for c in self.ci):
                raise TypeError("All elements in ci tuple must be callable")

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
        self._training_data = data.collect()

        return self

    def _get_full_predictions(self) -> pl.DataFrame:
        """Internal method to compute full predictions with all columns."""
        if not hasattr(self, "_training_data"):
            raise RuntimeError("You must call fit() before getting predictions.")

        data = self._training_data.lazy()

        if self.ci:
            ci_cols = ["y_pred_rgram_lci", "y_pred_rgram_uci"]
            ci_exprs = [
                ci_calc(pl.col("y_val").fill_null(pl.col("y_val").mean()))
                .over(self.over_cols + ["rgram_bin"])
                .alias(alias)
                for ci_calc, alias in zip(self.ci, ci_cols)
            ]

            data = data.with_columns(ci_exprs)

        # Collect once to check columns and decide what to drop
        collected = data.collect()
        cols_to_drop = []

        schema = collected.schema
        # Only drop x_var and y_var if there's only one unique value
        # (they add no information in that case). Keep them for multi-feature/multi-target cases.
        if "x_var" in schema:
            unique_x_count = collected["x_var"].n_unique()
            if unique_x_count <= 1:
                cols_to_drop.append("x_var")

        if "y_var" in schema:
            unique_y_count = collected["y_var"].n_unique()
            if unique_y_count <= 1:
                cols_to_drop.append("y_var")

        return collected.drop(cols_to_drop).sort(by=["x_val"])

    def transform(self) -> pl.LazyFrame:
        """
        Return the full binned regression results (for backward compatibility).

        Returns full data with bin assignments and predictions.

        Returns
        -------
        pl.LazyFrame
            Full results including all columns. Call `.collect()` to materialize.
        """
        return self._get_full_predictions().lazy()

    def predict(
        self, x: Union[Sequence[float], pl.Series], return_ci: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Predict binned regression values at new x points.

        Parameters
        ----------
        x : array-like or pl.Series
            New x values at which to predict.
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
        """
        if not hasattr(self, "_bin_to_y"):
            raise RuntimeError("Call fit() before predict().")

        # Check for empty input
        try:
            if len(x) == 0:
                raise ValueError("Cannot predict with empty array")
        except (TypeError, AttributeError):
            # If len(x) fails, let normal processing handle it
            pass

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
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        return_ci: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Fit and predict on training x values in one call.

        Parameters
        ----------
        x : str or array-like
            Feature(s) to bin. Column name(s) if `data` provided, else array-like.
        y : str or array-like
            Target(s). Column name(s) if `data` provided, else array-like.
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If provided, x/y are column names.
            If None, x/y are array-like.
        return_ci : bool, default=False
            If True, return confidence intervals along with predictions.

        Returns
        -------
        np.ndarray or tuple
            If return_ci=False: array of predictions at training x values
            If return_ci=True: tuple of (y_pred, y_ci_low, y_ci_high)
        """
        self.fit(data=data, x=x, y=y)

        # Get unique x values from training data for prediction
        x_train = (
            self._training_data.select("x_val").get_column("x_val").sort().to_numpy()
        )

        return self.predict(x_train, return_ci=return_ci)
