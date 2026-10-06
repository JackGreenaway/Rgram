"""Local-constant and local-linear univariate kernel regression."""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from rgram._typing import Array, FloatArray, Frame, Input, Prediction
from rgram.base import BaseUtils
from rgram.warnings import (
    DataHandlingWarning,
    ExtrapolationWarning,
    NumericalWarning,
    SupportWarning,
)


class KernelSmoother(RegressorMixin, BaseEstimator, BaseUtils):
    """One-dimensional local-constant or local-linear kernel regression.

    Parameters
    ----------
    bandwidth : {'silverman', 'scott', 'manual'}, default='silverman'
        Rule of thumb or manual bandwidth selection. Rules are based on x,
        not regression error; use cross-validation to tune predictive accuracy.
    bandwidth_value : float or None, default=None
        Positive bandwidth in feature units, required for 'manual'.
    bandwidth_adjust : float, default=1.0
        Positive multiplier applied to the selected bandwidth.
    kernel : str or callable, default='epanechnikov'
        One of epanechnikov, gaussian, uniform, triangular, cosine, logistic,
        biweight, tricube.
        A callable accepts a Polars expression and returns nonnegative weights.
    n_eval_samples : int, default=100
        Number of points returned by predict_grid().
    batch_size : int, default=128
        Maximum query rows processed together. Blocks are also capped at
        approximately one million query/training pairs (at least one query).
    unsupported : {'nan', 'raise'}, default='nan'
        Policy when a query has no positive kernel weight. 'nan' preserves
        the row and warns; 'raise' fails explicitly. Extrapolation has a separate policy.

    regression : {'local_constant', 'local_linear'}, default='local_constant'
        Local constant (Nadaraya–Watson) or weighted local linear fit. Local linear
        reduces boundary bias but can extrapolate and use negative coefficients.
    singular : {'constant', 'raise'}, default='constant'
        Fall back with a warning to a local constant for singular local-linear
        fits, or raise. Diagnostics always identify fallback rows.
    algorithm : {'auto', 'brute', 'neighbors'}, default='auto'
        Auto uses support windows only for already-sorted x and compact kernels;
        otherwise bounded dense blocks. Neighbors requires eligible input and
        raises otherwise. This estimator never sorts observations or an index.
    extrapolation : {'allow', 'nan', 'raise'}, default='allow'
        Outside-range queries warn and use the stated policy. Query values are
        never clipped. Allow still requires positive kernel support.

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
    training_sorted_ : bool
        Whether training features were already nondecreasing.
    algorithm_ : str
        Computational path selected during fitting. Prediction resolves the
        current algorithm option again, so changing it can alter the path.
    weight_scale_ : float
        Maximum training observation weight used for computational normalization.
    bandwidth_ : float
        Effective fitted bandwidth, including bandwidth_adjust.
    n_samples_in_ : int
        Number of training rows retained, including zero-weight rows.
    n_features_in_ : int
        Always one. Both 1D inputs and (n_samples, 1) arrays are accepted.

    Notes
    -----
    Training inputs are copied. Null, non-finite and multivariate inputs are
    rejected, never silently dropped. Kernel inspection columns follow the
    original training row order. Compact-kernel prediction visits only supported
    neighbors when training x is already sorted. No sorting is performed. Infinite-support
    and custom kernels use O(n_train * n_query) time with bounded blocks. get_weights() explicitly
    materializes a dense matrix and should be used on small query selections.
    """

    _KERNELS = {
        "epanechnikov",
        "gaussian",
        "uniform",
        "triangular",
        "cosine",
        "logistic",
        "biweight",
        "tricube",
    }

    def __init__(
        self,
        bandwidth: str = "silverman",
        bandwidth_value: Optional[float] = None,
        bandwidth_adjust: float = 1.0,
        kernel: str = "epanechnikov",
        n_eval_samples: int = 100,
        *,
        batch_size: int = 128,
        unsupported: str = "nan",
        regression: str = "local_constant",
        singular: str = "constant",
        algorithm: str = "auto",
        extrapolation: str = "allow",
    ) -> None:
        self.bandwidth = bandwidth
        self.bandwidth_value = bandwidth_value
        self.bandwidth_adjust = bandwidth_adjust
        self.kernel = kernel
        self.n_eval_samples = n_eval_samples
        self.batch_size = batch_size
        self.unsupported = unsupported
        self.regression = regression
        self.singular = singular
        self.algorithm = algorithm
        self.extrapolation = extrapolation

    @staticmethod
    def _positive(value: float, name: str) -> None:
        """Require a finite positive scalar setting and reject booleans and strings."""
        if isinstance(value, (bool, str)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a finite positive number")
        try:
            valid = np.isfinite(value) and value > 0
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite positive number")

    def _validate_parameters(self) -> None:
        """Validate bandwidth, kernel, computation, singularity, support and range options."""
        if self.extrapolation not in ("allow", "nan", "raise"):
            raise ValueError("extrapolation must be 'allow', 'nan', or 'raise'")
        if self.regression not in ("local_constant", "local_linear"):
            raise ValueError("regression must be 'local_constant' or 'local_linear'")
        if self.singular not in ("constant", "raise"):
            raise ValueError("singular must be 'constant' or 'raise'")
        if self.algorithm not in ("auto", "brute", "neighbors"):
            raise ValueError("algorithm must be 'auto', 'brute', or 'neighbors'")
        if self.bandwidth not in ("silverman", "scott", "manual"):
            raise ValueError("bandwidth must be one of silverman, scott, manual")
        if self.bandwidth == "manual":
            if self.bandwidth_value is None:
                raise ValueError(
                    "bandwidth_value must be specified when bandwidth='manual'"
                )
            self._positive(self.bandwidth_value, "bandwidth_value")
        self._positive(self.bandwidth_adjust, "bandwidth_adjust")
        for name in ("n_eval_samples", "batch_size"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.kernel, str):
            if self.kernel not in self._KERNELS:
                raise ValueError(f"kernel must be one of {sorted(self._KERNELS)}")
        elif not callable(self.kernel):
            raise TypeError("kernel must be a string or callable")
        if self.unsupported not in ("nan", "raise"):
            raise ValueError("unsupported must be 'nan' or 'raise'")

    def fit(
        self,
        X: Input,
        y: Input,
        *,
        data: Optional[Frame] = None,
        sample_weight: Optional[ArrayLike] = None,
    ) -> KernelSmoother:
        """Fit one numeric feature and one numeric response.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1), or str
            Numeric feature, or its column name when data is supplied.
        y : array-like of shape (n_samples,) or (n_samples, 1), or str
            Numeric response, or its column name when data is supplied.
        data : pandas.DataFrame, polars.DataFrame or polars.LazyFrame, default=None
            Source for named columns. Pandas columns are converted internally to
            Polars. Selected columns are snapshotted once; indices are ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Finite nonnegative influence weights with positive total mass.

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
        >>> from rgram import KernelSmoother
        >>> X = np.arange(10.0).reshape(-1, 1)
        >>> model = KernelSmoother().fit(X, X[:, 0])
        >>> model.predict(X).shape
        (10,)
        """
        x = X
        self._is_fitted = False
        self._validate_parameters()
        frame, _, _ = self._prepare_data(x, y, data)
        frame = frame.collect()
        self._record_feature_name(x, data)
        self.X_ = frame["x"].to_numpy().copy()
        self.y_ = frame["y"].to_numpy().copy()
        x_train = self._float_array(self.X_, "x")
        y_train = self._float_array(self.y_, "y")
        self.sample_weight_provided_ = sample_weight is not None
        weights = self._sample_weights(sample_weight, len(x_train))
        if self.bandwidth == "manual":
            bandwidth = self.bandwidth_value
        else:
            std = x_train.std(ddof=1) if len(x_train) > 1 else 0.0
            scale = std
            if self.bandwidth == "silverman":
                # Match Polars' historical nearest-quantile rule, with a
                # standard-deviation fallback for tied distributions.
                q = frame["x"].quantile(0.75) - frame["x"].quantile(0.25)
                scale = min(std, q / 1.34) if q > 0 else std
                if q == 0 and std > 0:
                    self._warn(
                        "Silverman IQR is zero; using standard deviation for bandwidth. No observations were removed.",
                        DataHandlingWarning,
                    )
            bandwidth = (
                (0.9 if self.bandwidth == "silverman" else 1.06)
                * scale
                * len(x_train) ** (-0.2)
            )
        bandwidth = bandwidth * self.bandwidth_adjust
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError(
                "Cannot determine a positive bandwidth; specify bandwidth='manual' and bandwidth_value"
            )
        self._check_order(x_train)
        self.training_sorted_ = not np.any(x_train[1:] < x_train[:-1])
        self._X_numeric = x_train
        self._y_numeric = y_train
        self.sample_weight_ = weights.copy()
        # Computational normalization is exposed, and never changes the snapshot.
        self.weight_scale_ = float(weights.max())
        self._normalized_sample_weight = (
            self._float_array(weights, "sample_weight") / self.weight_scale_
        )
        if ((weights > 0) & (self._normalized_sample_weight == 0)).any():
            raise ValueError(
                "sample_weight dynamic range underflows float64; revise weights explicitly"
            )
        self.algorithm_ = self._resolve_algorithm()
        self.bandwidth_ = float(bandwidth)
        self.n_features_in_ = 1
        self.n_samples_in_ = len(x_train)
        self.data_summary_ = self._data_summary()
        self.data_summary_.update(
            {
                "algorithm": self.algorithm_,
                "weight_scale": self.weight_scale_,
                "bandwidth": self.bandwidth_,
                "extrapolation": self.extrapolation,
            }
        )
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        """Require a successful fit through sklearn check_is_fitted."""
        check_is_fitted(self)

    _COMPACT_KERNELS = {
        "epanechnikov",
        "uniform",
        "triangular",
        "cosine",
        "biweight",
        "tricube",
    }

    def _normalized_kernel(
        self, u: FloatArray, observation_weights: FloatArray
    ) -> FloatArray:
        """Normalize positive kernel weights, preserving zeros exactly."""
        with np.errstate(over="ignore", invalid="ignore"):
            a = np.abs(u)
            if callable(self.kernel):
                raw = (
                    pl.DataFrame({"u": u.ravel()})
                    .select(self.kernel(pl.col("u")).alias("weight"))["weight"]
                    .to_numpy()
                )
                if raw.size != u.size:
                    raise ValueError(
                        "Custom kernel must return one weight per input value"
                    )
                w = raw.reshape(u.shape)
            elif self.kernel in ("gaussian", "logistic"):
                log_w = (
                    -0.5 * u**2
                    if self.kernel == "gaussian"
                    else -a - 2 * np.log1p(np.exp(-a))
                )
                log_w[:, observation_weights == 0] = -np.inf
                maximum = log_w.max(axis=1, keepdims=True)
                w = np.exp(log_w - np.where(np.isfinite(maximum), maximum, 0))
            elif self.kernel == "uniform":
                w = (a <= 1).astype(float)
            elif self.kernel == "triangular":
                w = np.maximum(1 - a, 0)
            elif self.kernel == "cosine":
                w = np.where(a < 1, np.cos(np.pi / 2 * np.minimum(a, 1)), 0)
            elif self.kernel == "tricube":
                w = (1 - np.minimum(a, 1) ** 3) ** 3
            elif self.kernel == "biweight":
                w = (1 - np.minimum(a, 1) ** 2) ** 2
            else:
                w = np.maximum(1 - np.minimum(a, 1) ** 2, 0)
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("Kernel must produce finite nonnegative weights")
        maximum = w.max(axis=1, keepdims=True)
        w = np.divide(w, maximum, out=np.zeros_like(w, dtype=float), where=maximum > 0)
        w *= observation_weights
        totals = w.sum(axis=1, keepdims=True)
        return np.divide(w, totals, out=np.zeros_like(w), where=totals > 0)

    def _resolve_algorithm(self) -> str:
        """Resolve auto/window/brute paths from the current kernel and fitted training order."""
        compact = isinstance(self.kernel, str) and self.kernel in self._COMPACT_KERNELS
        eligible = compact and self.training_sorted_
        if self.algorithm == "neighbors" and not eligible:
            raise ValueError(
                "algorithm='neighbors' requires already-sorted training x and a compact kernel; rgram never sorts input"
            )
        return "neighbors" if self.algorithm != "brute" and eligible else "brute"

    def _kernel_rows(self, x: Input) -> Iterator[tuple[int, Array, FloatArray]]:
        """Visit existing ordered support windows, or use bounded dense blocks."""
        if self._resolve_algorithm() == "neighbors":
            with np.errstate(over="ignore"):
                lower = np.searchsorted(
                    self._X_numeric,
                    np.nextafter(x - self.bandwidth_, -np.inf),
                    side="left",
                )
                upper = np.searchsorted(
                    self._X_numeric,
                    np.nextafter(x + self.bandwidth_, np.inf),
                    side="right",
                )
            for row, (left, right) in enumerate(zip(lower, upper)):
                indices = np.arange(left, right)
                if not len(indices):
                    yield row, indices, np.array([], dtype=float)
                    continue
                u = (self._X_numeric[indices] - x[row]) / self.bandwidth_
                weights = self._normalized_kernel(
                    u[None, :], self._normalized_sample_weight[indices]
                )[0]
                yield row, indices, weights
        else:
            size = min(self.batch_size, max(1, 1_000_000 // self.n_samples_in_))
            indices = np.arange(self.n_samples_in_)
            for start in range(0, len(x), size):
                with np.errstate(over="ignore", invalid="ignore"):
                    u = (
                        self._X_numeric[None, :] - x[start : start + size, None]
                    ) / self.bandwidth_
                weights = self._normalized_kernel(u, self._normalized_sample_weight)
                for offset, row_weights in enumerate(weights):
                    yield start + offset, indices, row_weights

    def _prediction_weights(
        self, query: float, indices: Array, weights: FloatArray
    ) -> tuple[FloatArray, bool]:
        """Return local-constant or local-linear coefficients and an explicit fallback flag."""
        if self.regression == "local_constant" or not weights.any():
            return weights, False
        with np.errstate(over="ignore", invalid="ignore"):
            d = (self._X_numeric[indices] - query) / self.bandwidth_
        d = np.where(weights > 0, d, 0)
        scale = np.max(np.abs(d))
        if np.isfinite(scale) and scale > 0:
            d = d / scale
            mean = weights @ d
            centered = d - mean
            variance = weights @ (centered**2)
            if variance > 100 * np.finfo(float).eps * max(
                weights @ (d**2), np.finfo(float).tiny
            ):
                return weights * (1 - mean * centered / variance), False
        if self.singular == "raise":
            raise ValueError(
                "Singular local-linear fit; increase bandwidth or use singular='constant'"
            )
        return weights, True

    def _handle_unsupported(self, supported: NDArray[np.bool_]) -> None:
        """Apply the configured support warning/NaN or exception policy."""
        count = np.count_nonzero(~supported)
        if count:
            message = f"{count} evaluation point(s) have no positive kernel support"
            if self.unsupported == "raise":
                raise ValueError(message)
            self._warn(
                message + "; returning NaN without dropping rows", SupportWarning
            )

    def _evaluate(
        self, x_eval: Input, weight_kind: Optional[str] = None
    ) -> tuple[pl.DataFrame, Optional[FloatArray]]:
        """Evaluate queries and local diagnostics, optionally allocating dense inspection coefficients."""
        self._check_fitted()
        self._validate_parameters()
        original_x = self._prediction_features(x_eval)
        x = self._float_array(original_x, "x")
        self._check_order(x)
        outside = (x < self._X_numeric.min()) | (x > self._X_numeric.max())
        if outside.any():
            if self.extrapolation == "raise":
                raise ValueError("Query lies outside the training range")
            self._warn(
                f"{outside.sum()} queries are outside the training range; extrapolation={self.extrapolation!r}. Query values are unchanged.",
                ExtrapolationWarning,
            )
        prediction = np.full(len(x), np.nan)
        effective = np.zeros(len(x))
        neighbors = np.zeros(len(x), dtype=int)
        supported = np.zeros(len(x), dtype=bool)
        fallback = np.zeros(len(x), dtype=bool)
        result = np.zeros((len(x), self.n_samples_in_)) if weight_kind else None
        for row, indices, weights in self._kernel_rows(x):
            if not weights.any() or (self.extrapolation == "nan" and outside[row]):
                continue
            coefficients, fallback[row] = self._prediction_weights(
                x[row], indices, weights
            )
            prediction[row] = coefficients @ self._y_numeric[indices]
            supported[row] = True
            neighbors[row] = np.count_nonzero(weights)
            effective[row] = 1 / (weights @ weights)
            if result is not None:
                result[row, indices] = (
                    weights if weight_kind == "kernel" else coefficients
                )
        self._handle_unsupported(supported)
        if fallback.any():
            self._warn(
                f"{fallback.sum()} singular local-linear fit(s) used local-constant fallback",
                NumericalWarning,
            )
        return pl.DataFrame(
            {
                "x": original_x,
                "in_training_range": ~outside,
                "prediction": prediction,
                "supported": supported,
                "n_neighbors": neighbors,
                "effective_n": effective,
                "local_linear_fallback": fallback,
            }
        ), result

    def get_weights(self, X: Input, *, kind: str = "prediction") -> FloatArray:
        """Inspect the dense matrix of local influences in training-row order.

        Parameters
        ----------
        X : array-like of shape (n_queries,) or (n_queries, 1)
            Query feature values.
        kind : {'prediction', 'kernel'}, default='prediction'
            Prediction coefficients or normalized nonnegative kernel/observation
            weights. Local-linear prediction coefficients can be negative.

        Returns
        -------
        weights : ndarray of shape (n_queries, ``n_samples_in_``)
            Columns follow original training order. Unsupported rows contain zeros.
            Supported prediction rows combine ``y_`` into predicted values. For local
            constants, prediction and kernel weights coincide.

        Notes
        -----
        This intentionally allocates a dense matrix. Use small query selections;
        ordinary prediction does not require storing this full matrix.
        """
        if kind not in ("prediction", "kernel"):
            raise ValueError("kind must be 'prediction' or 'kernel'")
        return self._evaluate(X, weight_kind=kind)[1]

    def predict_diagnostics(self, X: Input) -> pl.DataFrame:
        """Return local support and numerical information for each query.

        Parameters
        ----------
        X : array-like of shape (n_queries,) or (n_queries, 1)
            One query feature, with fitted feature name if named.

        Returns
        -------
        diagnostics : polars.DataFrame
            Columns: x, in_training_range, prediction, supported, n_neighbors,
            effective_n, local_linear_fallback. One row per query in original order.
            Unsupported rows have NaN prediction, zero neighbors and effective_n,
            and false supported, unless a strict policy raises.

        Notes
        -----
        effective_n = 1 / sum(normalized kernel/observation weights ** 2). This
        measures weight concentration, not confidence or signed-coefficient variance.
        """
        return self._evaluate(X)[0]

    def predict(self, X: Input) -> Prediction:
        """Evaluate the local-constant or local-linear response curve.

        Parameters
        ----------
        X : array-like of shape (n_queries,) or (n_queries, 1)
            Query feature values. Named inputs must match the fitted feature.

        Returns
        -------
        prediction : ndarray of shape (n_queries,)
            One value per query in input order. Unsupported estimates are
            NaN unless the configured support policy raises.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If no successful fit is available.
        ValueError
            For invalid queries, feature-name mismatch, or strict policies.

        See Also
        --------
        predict_interval : Pointwise bootstrap bounds and draw-support diagnostics.
        """
        return self.predict_diagnostics(X)["prediction"].to_numpy()

    def predict_grid(self) -> pl.DataFrame:
        """Evaluate diagnostics on an evenly spaced training-range grid.

        Returns
        -------
        diagnostics : polars.DataFrame
            The predict_diagnostics schema at n_eval_samples locations spanning
            the observed feature minimum and maximum, endpoints included.

        Notes
        -----
        This changes evaluation locations, not smoothing. With a constant feature,
        all grid coordinates are equal. Fit must have succeeded first.
        """
        self._check_fitted()
        self._validate_parameters()
        return self.predict_diagnostics(
            np.linspace(self.X_.min(), self.X_.max(), self.n_eval_samples)
        )

    def fit_predict(
        self,
        X: Input,
        y: Input,
        *,
        data: Optional[Frame] = None,
        sample_weight: Optional[ArrayLike] = None,
    ) -> Prediction:
        """Fit and evaluate at the original training feature values.

        Parameters
        ----------
        X : array-like or str
            One training feature, or its column name when data is supplied.
        y : array-like or str
            Aligned numeric response, or its column name when data is supplied.
        data : pandas.DataFrame, polars.DataFrame or polars.LazyFrame, default=None
            Source for named columns.
        sample_weight : array-like of shape (n_samples,), default=None
            Observation influence weights; see fit.

        Returns
        -------
        prediction : ndarray of shape (n_samples,)
            Predictions at original training rows, preserving their order.

        Notes
        -----
        Training predictions describe the fit, not held-out performance.
        Fit followed by predict evaluates new queries; use predict_interval
        for bootstrap confidence bounds.
        """
        self.fit(X, y, data=data, sample_weight=sample_weight)
        return self.predict(self.X_)
