"""Binned summaries for one numeric feature–response relationship."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from rgram._typing import CV, Input, Prediction
from rgram.aggregation import (
    AGGREGATIONS,
    Aggregation,
    ArrayAggregation,
    aggregation_expr,
)
from rgram.base import BaseUtils
from rgram.warnings import BinningWarning, ExtrapolationWarning, SupportWarning


def _mean(x: pl.Expr) -> pl.Expr:
    return x.mean()


def _lower_spread(x: pl.Expr) -> pl.Expr:
    return x.mean() - x.std()


def _upper_spread(x: pl.Expr) -> pl.Expr:
    return x.mean() + x.std()


class Regressogram(RegressorMixin, BaseEstimator, BaseUtils):
    """
    Regressogram

    Binned regression estimator for one feature and target.
    Predicts using binned aggregation with optional confidence intervals.

    Parameters
    ----------
    binning : {'dist', 'width', 'none', 'int'}, default='dist'
        Binning strategy.
    agg : str or callable, default='mean'
        Named reduction, Polars expression callable, or array_aggregation adapter.
        Every bin must produce one real numeric scalar; weights are never ignored.
    ci : tuple of aggregations or None, default=None
        Explicit lower/upper descriptive summaries. No intervals are computed by
        default. Use predict_interval for pointwise bootstrap inference.
    n_bins : int, str or None, default=None
        Count or selection rule for both dist and width. None means 'auto'.
        Rules: auto, cv, fd, scott, sturges, sqrt, rice. Dist auto uses a bounded
        cube-root count; width auto combines FD and Sturges. CV minimizes held-out
        regression MSE and requires full prediction support in every fold.
    bin_width : float or None, default=None
        Explicit width in feature units; requires width binning and n_bins=None.
        The last bin may be shorter. Counts that exceed max_bins raise.
    max_bins : int, default=512
        Allocation guard. Automatic rules are capped; larger explicit counts raise.
    min_samples_bin : int, default=5
        Bounds the default dist-auto count and generated CV candidates by n // value.
        This is a target occupancy, not a guarantee with ties or explicit counts.
    cv : int, splitter or iterable, default=5
        Used only for n_bins='cv'. Integers use shuffled KFold.
    bin_candidates : sequence of int or None, default=None
        Candidate counts for CV. Defaults to a bounded geometric grid.
    random_state : int or None, default=0
        Seed for default CV splitting.
    extrapolation : {'clip', 'nan', 'raise'}, default='clip'
        Outside the training range, map to edge bins (with a warning), leave
        unsupported, or raise. Query values are never modified.
    unsupported : {'nan', 'raise'}, default='nan'
        Preserve missing estimates as NaN with a warning, or raise.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        Independent snapshot of original training features in row order.
    y_ : ndarray of shape (n_samples,)
        Independent snapshot of original training responses in row order.
    sample_weight_ : ndarray of shape (n_samples,)
        Original supplied weights, or ones if absent. Independent snapshot.
    sample_weight_provided_ : bool
        Whether observation weights were supplied during fit.
    feature_names_in_ : ndarray of shape (1,)
        Training feature name, present only for named input.
    data_summary_ : dict
        Data integrity, original dtype, fitted policy and computation summary.
    n_features_in_ : int
        Always one.
    n_samples_in_ : int
        Number of rows, including zero-weight rows.
    selected_n_bins_ : int, str or None
        Count selected by internal CV, or original n_bins option without CV.
    requested_n_bins_ : int
        Nominal cell count after rule/cap selection but before tied boundaries
        are merged. Present for dist/width only. Sample/constant-x constraints
        can reduce an explicit request before this attribute is assigned.
    n_bins_ : int
        Number of fitted cells (empty cells may exist).
    bin_edges_ : ndarray
        Interior fitted boundaries for dist/width binning.
    bin_width_ : float or None
        Width for width binning; None for other strategies.
    bins_ : polars.DataFrame
        Occupied bin estimates and original observation counts.
    cv_results_ : dict
        Error and coverage by candidate, available when n_bins='cv'.

    """

    def __init__(
        self,
        *,
        binning: Literal["dist", "width", "none", "int"] = "dist",
        agg: Aggregation = "mean",
        ci: Optional[tuple[Aggregation, Aggregation]] = None,
        n_bins: Optional[Union[int, str]] = None,
        bin_width: Optional[float] = None,
        max_bins: int = 512,
        min_samples_bin: int = 5,
        cv: CV = 5,
        bin_candidates: Optional[Sequence[int]] = None,
        random_state: Optional[int] = 0,
        extrapolation: str = "clip",
        unsupported: str = "nan",
    ) -> None:
        self.binning = binning
        self.agg = agg
        self.ci = ci
        self.n_bins = n_bins
        self.bin_width = bin_width
        self.max_bins = max_bins
        self.min_samples_bin = min_samples_bin
        self.cv = cv
        self.bin_candidates = bin_candidates
        self.random_state = random_state
        self.extrapolation = extrapolation
        self.unsupported = unsupported

    def _validate_parameters(self) -> None:
        """Validate bin rules, support policies, and declared scalar/weighted aggregation contracts."""
        if isinstance(self.agg, str):
            if self.agg not in AGGREGATIONS:
                raise ValueError(
                    f"Unknown aggregation {self.agg!r}; choose {AGGREGATIONS} or supply a callable"
                )
        elif not callable(self.agg):
            raise TypeError("agg must be a named aggregation or callable")
        if self.extrapolation not in ("clip", "nan", "raise"):
            raise ValueError("extrapolation must be 'clip', 'nan', or 'raise'")
        if self.unsupported not in ("nan", "raise"):
            raise ValueError("unsupported must be 'nan' or 'raise'")

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
            if not all(callable(c) or isinstance(c, str) for c in self.ci):
                raise TypeError(
                    "All elements in ci tuple must be named aggregations or callable"
                )

        if self.binning not in ("dist", "width", "none", "int"):
            raise ValueError(f"Unknown binning type: {self.binning}")
        rules = ("auto", "cv", "fd", "scott", "sturges", "sqrt", "rice")
        if isinstance(self.n_bins, str):
            if self.n_bins not in rules:
                raise ValueError(
                    f"n_bins must be a positive integer, None, or one of {rules}"
                )
        elif self.n_bins is not None and (
            isinstance(self.n_bins, bool)
            or not isinstance(self.n_bins, (int, np.integer))
            or self.n_bins < 1
        ):
            raise ValueError(
                "n_bins must be a positive integer, None, or a selection rule"
            )
        for name in ("max_bins", "min_samples_bin"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.n_bins, (int, np.integer)) and self.n_bins > self.max_bins:
            raise ValueError("n_bins exceeds max_bins; increase max_bins explicitly")
        if self.bin_width is not None:
            if self.binning != "width" or self.n_bins is not None:
                raise ValueError("bin_width requires binning='width' and n_bins=None")
            if (
                isinstance(self.bin_width, (bool, str))
                or not np.isscalar(self.bin_width)
                or not np.isfinite(self.bin_width)
                or self.bin_width <= 0
            ):
                raise ValueError("bin_width must be a finite positive number")
        if self.binning not in ("dist", "width") and self.n_bins is not None:
            raise ValueError("n_bins applies only to dist or width binning")

    def _cv_bin_count(
        self, x: Input, y: Input, sample_weight: Optional[ArrayLike] = None
    ) -> int:
        """Run explicit coverage-aware bin-count CV and retain candidate results without mutating parameters."""
        from rgram.model_selection import CoverageSearchCV

        limit = min(self.max_bins, max(1, len(x) // self.min_samples_bin))
        candidates = self.bin_candidates
        if candidates is None:
            candidates = np.unique(
                np.geomspace(1, limit, min(limit, 12)).round().astype(int)
            ).tolist()
        else:
            candidates = list(candidates)
        if not candidates or any(
            isinstance(k, bool)
            or not isinstance(k, (int, np.integer))
            or k < 1
            or k > self.max_bins
            for k in candidates
        ):
            raise ValueError(
                "bin_candidates must contain positive integers no greater than max_bins"
            )
        search = CoverageSearchCV(
            clone(self).set_params(n_bins=1),
            {"n_bins": candidates},
            cv=self.cv,
            random_state=self.random_state,
        ).fit(x, y, sample_weight=sample_weight)
        self.cv_results_ = search.cv_results_
        return search.best_params_["n_bins"]

    def _learn_bin_params(self, data: pl.LazyFrame) -> None:
        """Learn bounds, count and interior boundaries; merge ties and expose fitted resolution."""
        x = data.select("x_val").collect()["x_val"].to_numpy()
        n = len(x)
        self._x_min, self._x_max = x.min().item(), x.max().item()
        if self.binning in ("int", "none"):
            self._min_bin = int(self._x_min) if self.binning == "int" else self._x_min
            self._max_bin = int(self._x_max) if self.binning == "int" else self._x_max
            self.n_bins_ = len(
                np.unique(x.astype(np.int64) if self.binning == "int" else x)
            )
            return
        x = self._float_array(x, "x")
        with np.errstate(over="ignore"):
            span = self._x_max - self._x_min
        if not np.isfinite(span):
            raise ValueError("Feature range is too large; rescale x before binning")
        choice = self.selected_n_bins_
        cap = min(n, self.max_bins)
        if span == 0:
            count = 1
        elif self.bin_width is not None:
            with np.errstate(over="ignore"):
                count = np.ceil(span / self.bin_width)
            if not np.isfinite(count) or count > self.max_bins:
                raise ValueError(
                    "bin_width requires more than max_bins; increase width or max_bins"
                )
            count = max(1, int(count))
        elif isinstance(choice, (int, np.integer)):
            count = min(int(choice), cap) if self.binning == "dist" else int(choice)
        else:
            rule = "auto" if choice is None else choice
            sturges = np.ceil(np.log2(n) + 1)
            if rule == "auto" and self.binning == "dist":
                # Quantile cells already adapt to density; range/IQR can explode
                # under outliers and is not a suitable default for their count.
                count = min(np.ceil(np.cbrt(n)), max(1, n // self.min_samples_bin))
            elif rule in ("sturges", "sqrt", "rice"):
                count = {
                    "sturges": sturges,
                    "sqrt": np.ceil(np.sqrt(n)),
                    "rice": np.ceil(2 * np.cbrt(n)),
                }[rule]
            else:
                # Work in unit range to avoid overflow in scale estimation.
                z = (x - self._x_min) / span
                if rule == "scott":
                    width = np.std(z) * np.cbrt(24 * np.sqrt(np.pi) / n)
                else:
                    width = 2 * np.subtract(*np.quantile(z, [0.75, 0.25])) / np.cbrt(n)
                with np.errstate(over="ignore", divide="ignore"):
                    count = np.ceil(1 / width) if width > 0 else sturges
                if rule == "auto":
                    count = max(count, sturges)
            if count > cap:
                self._warn(
                    f"Automatic bin count capped at {cap}; all observations are retained",
                    BinningWarning,
                )
            count = max(1, int(min(count, cap)))
        if isinstance(choice, (int, np.integer)) and count != choice:
            self._warn(
                f"Requested {choice} bins but fitted {count} because of sample count or constant x; all rows are retained",
                BinningWarning,
            )
        self.requested_n_bins_ = (
            int(choice) if isinstance(choice, (int, np.integer)) else int(count)
        )
        if self.binning == "dist":
            edges = np.unique(np.quantile(x, np.arange(1, count) / count))
            edges = edges[edges < self._x_max]
        elif span == 0:
            edges = np.array([], dtype=float)
        elif self.bin_width is not None:
            edges = self._x_min + np.arange(1, count) * self.bin_width
        else:
            edges = np.linspace(self._x_min, self._x_max, count + 1)[1:-1]
        self._bin_edges = np.unique(edges).tolist()
        if len(self._bin_edges) + 1 != count:
            self._warn(
                f"Repeated boundaries merged {count} requested cells into {len(self._bin_edges) + 1}; all rows are retained",
                BinningWarning,
            )
        self._min_bin = 0
        self._max_bin = len(self._bin_edges)
        self.n_bins_ = self._max_bin + 1
        self._n_bins = self.n_bins_
        self.bin_width_ = (
            (self.bin_width if self.bin_width is not None else span / count)
            if self.binning == "width"
            else None
        )
        self._bin_width = self.bin_width_

    def _predict_bin_expr(self) -> pl.Expr:
        """
        Returns a Polars expression for binning x values.

        Returns
        -------
        pl.Expr
            The binning expression.
        """
        if self.binning in ("width", "dist"):
            # Quantiles are right-closed; widths are left-closed with the maximum
            # included in the last bin. Both assignments are fixed after fitting.
            side = "left" if self.binning == "dist" else "right"
            bin_id = pl.col("x_val").map_batches(
                lambda values: pl.Series(
                    np.searchsorted(self._bin_edges, values.to_numpy(), side=side)
                ),
                return_dtype=pl.Int64,
                is_elementwise=True,
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
        X: Input = None,
        y: Input = None,
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        sample_weight: Optional[ArrayLike] = None,
        *,
        x: Input = None,
    ) -> "Regressogram":
        """Fit one numeric feature and one numeric response.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1), or str, default=None
            Numeric feature, or its column name when data is supplied.
        y : array-like of shape (n_samples,) or (n_samples, 1), or str, default=None
            Numeric response, or its column name when data is supplied. Required.
        data : polars.DataFrame or polars.LazyFrame, default=None
            Source for named columns. Selected columns are materialized once.
        sample_weight : array-like of shape (n_samples,), default=None
            Finite nonnegative influence weights with positive total mass.
        x : array-like or str, default=None
            Legacy alias for X. Supply exactly one of X and x.

        Returns
        -------
        self : estimator
            Fitted estimator, with independent training snapshots.

        Raises
        ------
        ValueError
            For incompatible shapes, unequal lengths, missing/non-finite values,
            invalid settings, or unsupported weighting conventions.
        TypeError
            For nonnumeric/complex input or invalid callback/data types.

        Notes
        -----
        Rows and duplicates are preserved. Unsorted input emits an advisory warning.
        A failed refit invalidates prediction access. Zero-weight rows still participate
        in feature-based bin or bandwidth selection. Normal fitting does not run CV;
        Regressogram(n_bins='cv') explicitly requests it.

        Examples
        --------
        >>> import numpy as np
        >>> from rgram import Regressogram
        >>> X = np.arange(10.0).reshape(-1, 1)
        >>> model = Regressogram().fit(X, X[:, 0])
        >>> model.predict(X).shape
        (10,)
        """
        x = self._resolve_X(X, x)
        self._is_fitted = False
        self._validate_parameters()
        self._bin_edges = []
        self.bin_width_ = None
        self.__dict__.pop("cv_results_", None)
        data_lf, _, _ = self._prepare_data(data=data, x=x, y=y)
        frame = data_lf.collect()
        self.X_ = frame["x"].to_numpy().copy()
        self.y_ = frame["y"].to_numpy().copy()
        self._float_array(self.y_, "y")
        self._record_feature_name(x, data)
        self.sample_weight_provided_ = sample_weight is not None
        self.sample_weight_ = self._sample_weights(sample_weight, len(self.X_))
        if not np.isfinite(self.sample_weight_.sum(dtype=float)):
            raise ValueError(
                "sample_weight sum overflows float64; rescale weights explicitly"
            )
        self._check_order(self.X_)
        self.selected_n_bins_ = (
            self._cv_bin_count(self.X_, self.y_, sample_weight)
            if self.n_bins == "cv"
            else self.n_bins
        )
        data = data_lf.select(
            pl.col("x").alias("x_val"),
            pl.col("y").cast(float).alias("y_val"),
            pl.lit("x").alias("x_var"),
            pl.lit("y").alias("y_var"),
        )
        data = data.with_columns(pl.Series("sample_weight", self.sample_weight_))
        self.over_cols = ["x_var", "y_var"]
        self.n_features_in_ = 1
        self.n_samples_in_ = data.select(pl.len()).collect().item()

        # learn bin parameters and assign bins
        self._learn_bin_params(data)
        data = data.with_columns(
            [self._predict_bin_expr().over(self.over_cols).alias("rgram_bin")]
        )

        # Native reductions stay in Polars; numeric callbacks run in Python so
        # their exceptions cannot cross a Rust UDF boundary as engine panics.
        reductions = [("y_pred_rgram", self.agg)]
        if self.ci is not None:
            reductions.extend(zip(("y_pred_rgram_lci", "y_pred_rgram_uci"), self.ci))
        expressions = [
            pl.len().alias("n_samples"),
            pl.col("sample_weight").sum().alias("weight_sum"),
            (pl.col("sample_weight") > 0).sum().alias("n_positive_weight"),
            pl.col("x_val").min().alias("x_min_observed"),
            pl.col("x_val").max().alias("x_max_observed"),
        ]
        expressions.extend(
            aggregation_expr(
                calc,
                pl.col("y_val"),
                pl.col("sample_weight") if self.sample_weight_provided_ else None,
            ).alias(name)
            for name, calc in reductions
            if not isinstance(calc, ArrayAggregation)
        )
        self._bin_to_y = (
            data.group_by("rgram_bin", maintain_order=True).agg(expressions).collect()
        )
        if any(isinstance(calc, ArrayAggregation) for _, calc in reductions):
            groups = data.collect().partition_by("rgram_bin", maintain_order=True)
            for name, calc in reductions:
                if isinstance(calc, ArrayAggregation):
                    values = [
                        calc(
                            group["y_val"].to_numpy(),
                            group["sample_weight"].to_numpy()
                            if self.sample_weight_provided_
                            else None,
                        )
                        for group in groups
                    ]
                    self._bin_to_y = self._bin_to_y.with_columns(
                        pl.Series(name, values)
                    )
        for name in self._bin_to_y.columns:
            if (
                name.startswith("y_pred")
                and not self._bin_to_y[name].dtype.is_numeric()
            ):
                raise ValueError("agg and ci must return one numeric scalar per bin")
        for name in self._bin_to_y.columns:
            if (
                name.startswith("y_pred")
                and np.isinf(self._bin_to_y[name].to_numpy()).any()
            ):
                raise ValueError(
                    "Aggregation produced infinity; rescale inputs or check the aggregation"
                )
        self.bins_ = self._bin_to_y.clone()
        if self.binning in ("dist", "width"):
            limits = np.r_[self._x_min, self._bin_edges, self._x_max]
            ids = self.bins_["rgram_bin"].to_numpy().astype(int)
            self.bins_ = self.bins_.with_columns(
                pl.Series("bin_left", limits[ids]),
                pl.Series("bin_right", limits[ids + 1]),
            )
        self.data_summary_ = self._data_summary()
        self.data_summary_.update(
            {
                "binning": self.binning,
                "n_bins_requested": self.n_bins,
                "n_bins_fitted": self.n_bins_,
                "cv_used": self.n_bins == "cv",
                "extrapolation": self.extrapolation,
            }
        )
        self.bin_edges_ = np.asarray(getattr(self, "_bin_edges", []))
        self._is_fitted = True

        return self

    def _prediction_frame(self, values: ArrayLike) -> pl.DataFrame:
        """Join fitted bin estimates in query order, retaining support/range flags and applying policies."""
        check_is_fitted(self)
        x = self._prediction_features(values)
        self._check_order(x)
        if self.binning in ("dist", "width"):
            self._float_array(x, "x")
        outside = (x < self._x_min) | (x > self._x_max)
        if outside.any():
            if self.extrapolation == "raise":
                raise ValueError("Query lies outside the training range")
            self._warn(
                f"{outside.sum()} queries are outside the training range; extrapolation={self.extrapolation!r}. Original query values and row order are unchanged.",
                ExtrapolationWarning,
            )
        lf = pl.DataFrame({"x_val": x}).with_row_index("row_index").lazy()
        lf = lf.with_columns(self._predict_bin_expr().alias("rgram_bin"))
        result = lf.join(
            self._bin_to_y.lazy(),
            on="rgram_bin",
            how="left",
            validate="m:1",
            maintain_order="left",
        ).collect()
        if self.extrapolation == "nan" and outside.any():
            result = result.with_columns(
                pl.when(pl.Series(outside))
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name in result.columns
                if name.startswith("y_pred")
            )
        result = result.with_columns(
            pl.col(name).cast(pl.Float64).fill_null(float("nan"))
            for name in result.columns
            if name.startswith("y_pred")
        )
        finite = np.isfinite(result["y_pred_rgram"].to_numpy())
        if not finite.all():
            if self.unsupported == "raise":
                raise ValueError("Some query bins have no finite estimate")
            self._warn(
                "Some query bins have no finite estimate; returning NaN without dropping rows",
                SupportWarning,
            )
        return result.with_columns(
            pl.Series("in_training_range", ~outside), pl.Series("supported", finite)
        )

    def predict(
        self, X: Input = None, return_ci: bool = False, *, x: Input = None
    ) -> Prediction:
        """Evaluate the fitted bin summary at each query.

        Parameters
        ----------
        X : array-like of shape (n_queries,) or (n_queries, 1), default=None
            Query feature values. Named inputs must match the fitted feature.
        return_ci : bool, default=False
            Return configured descriptive ci endpoints alongside predictions.
        x : array-like, default=None
            Legacy alias for X; supply exactly one alias.

        Returns
        -------
        prediction : ndarray of shape (n_queries,) or tuple
            Default: one estimate per query, in input order. With return_ci=True,
            return (prediction, lower, upper); endpoints are None if ci=None.
            Configured endpoints are descriptive summaries, not bootstrap intervals.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If no successful fit is available.
        ValueError
            For invalid queries, feature-name mismatch, or a strict support/range policy.

        See Also
        --------
        predict_interval : Bootstrap confidence intervals for the curve.
        predict_diagnostics : Bin assignment, counts, support and range flags.
        """
        x = self._resolve_X(X, x)
        frame = self._prediction_frame(x)
        prediction = frame["y_pred_rgram"].to_numpy()
        if not return_ci:
            return prediction
        if self.ci is None:
            return prediction, None, None
        return (
            prediction,
            frame["y_pred_rgram_lci"].to_numpy(),
            frame["y_pred_rgram_uci"].to_numpy(),
        )

    def predict_diagnostics(self, x: Input) -> pl.DataFrame:
        """Return bin assignment and support information for each query.

        Parameters
        ----------
        x : array-like of shape (n_queries,) or (n_queries, 1)
            One query feature, with fitted feature name if named.

        Returns
        -------
        diagnostics : polars.DataFrame
            Columns: x, prediction, rgram_bin, n_samples, weight_sum,
            n_positive_weight, supported, in_training_range. Each row corresponds
            to the query at the same position. Missing bin information is null;
            unsupported predictions are NaN unless a strict policy raises.
        """
        return self._prediction_frame(x).select(
            pl.col("x_val").alias("x"),
            pl.col("y_pred_rgram").alias("prediction"),
            "rgram_bin",
            "n_samples",
            "weight_sum",
            "n_positive_weight",
            "supported",
            "in_training_range",
        )

    def fit_predict(
        self,
        X: Input = None,
        y: Input = None,
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        return_ci: bool = False,
        sample_weight: Optional[ArrayLike] = None,
        *,
        x: Input = None,
    ) -> Prediction:
        """Fit and evaluate at the original training feature values.

        Parameters
        ----------
        X : array-like or str, default=None
            One training feature, or its column name when data is supplied.
        y : array-like or str, default=None
            One training response, or its column name when data is supplied. Required.
        data : polars.DataFrame or polars.LazyFrame, default=None
            Source for named columns.
        return_ci : bool, default=False
            Return configured descriptive ci endpoints, not bootstrap intervals.
        sample_weight : array-like of shape (n_samples,), default=None
            Observation influence weights; see fit.
        x : array-like or str, default=None
            Legacy alias for X; supply exactly one alias.

        Returns
        -------
        prediction : ndarray or tuple
            Training predictions in original row order. With return_ci=True, return
            (prediction, lower, upper), with None endpoints if ci=None.

        Notes
        -----
        Uses fit followed by predict on the stored training features. In-sample
        accuracy does not measure generalization to new observations.
        """
        x = self._resolve_X(X, x)
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

        self.fit(data=data, x=x, y=y, sample_weight=sample_weight)

        return self.predict(x=self.X_, return_ci=return_ci)
