"""Shared validation, snapshots, diagnostics, and interval delegation."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Type, Union

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from rgram._typing import Array, FloatArray, Frame, Input


class BaseUtils:
    """
    BaseUtils

    Utility base class for DataFrame-related utilities, such as list conversion and group-over operations.

    """

    def __init__(
        self,
    ) -> None:
        pass

    @staticmethod
    def _resolve_X(X: Input, legacy: Input, *, alias: str = "x") -> Input:
        """Accept sklearn's X keyword while retaining the exploration API."""
        if X is not None and legacy is not None:
            raise TypeError(f"Pass either X or {alias}, not both")
        values = X if X is not None else legacy
        if values is None:
            raise TypeError("Missing required input X")
        return values

    @staticmethod
    def _to_list(item: Optional[Union[str, Sequence[Any]]]) -> Optional[list[Any]]:
        """
        Convert a string or sequence to a list, or return None.

        Parameters
        ----------
        item : str, sequence, or None
            The item to convert.

        Returns
        -------
        list or None
            The converted list or None if input is None.
        """
        if item is None:
            return None

        elif isinstance(item, str):
            return [item]

        return list(item)

    @staticmethod
    def _init_kws(var_input: Any, dataclass: Type) -> Any:
        """
        Initialise keyword arguments for dataclass instantiation.

        Parameters
        ----------
        var_input : any
            The input data, can be a dataclass instance or a dictionary-like object.
        dataclass : type
            The dataclass type to instantiate.

        Returns
        -------
        any
            An instance of the dataclass or an empty dataclass if input is None.
        """

        if var_input is True:
            return dataclass()

        elif isinstance(var_input, dict):
            return dataclass(**var_input)

        elif isinstance(var_input, dataclass):
            return var_input

        else:
            return None

    def _over_function(self, x: pl.Expr) -> pl.Expr:
        """
        Apply a Polars expression over grouping columns if present.

        Parameters
        ----------
        x : pl.Expr
            The Polars expression to apply.
        Returns
        -------
        pl.Expr
            The expression.
        """
        return x

    @staticmethod
    def _is_array_like(obj: Any) -> bool:
        """
        Check if an object is array-like (has __len__ and __getitem__ but is not a string).

        Parameters
        ----------
        obj : any
            The object to check.

        Returns
        -------
        bool
            True if the object is array-like, False otherwise.
        """
        return (
            hasattr(obj, "__len__")
            and hasattr(obj, "__getitem__")
            and not isinstance(obj, str)
        )

    @staticmethod
    def _process_array_input(
        input_data: Any,
        col_prefix: str,
        df_dict: dict[str, Any],
    ) -> str:
        """
        Process array-like input and add to df_dict.

        Parameters
        ----------
        input_data : array-like
            The input array data.
        col_prefix : str
            Column name for the array (e.g., 'x', 'y', 'keys').
        df_dict : dict
            Dictionary to accumulate arrays for DataFrame creation.

        Returns
        -------
        str
            The column name assigned to this input.

        Raises
        ------
        ValueError
            If input is a string (column name) when data=None, or if not array-like.
        TypeError
            If input is a dict (not allowed) or contains complex numbers.
        """
        if isinstance(input_data, str):
            raise ValueError(
                f"Column name '{input_data}' provided but data=None. "
                "When data=None, provide array-like values, not column names."
            )

        if isinstance(input_data, dict):
            raise TypeError(
                f"Dictionary input is not supported for {col_prefix}. "
                "Provide array-like values (list, ndarray, Series) instead."
            )

        # Validate array using consolidated validation method
        # Convert TypeError to ValueError for API consistency
        try:
            BaseUtils._validate_single_array(
                input_data, col_prefix, allow_empty=False, numeric_only=True
            )
        except TypeError as e:
            # If it's not array-like, convert to ValueError to maintain API
            if "must be array-like" in str(e):
                raise ValueError(f"Input must be {str(e).split('must be')[1].strip()}")
            raise

        array = np.asarray(input_data)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(
                f"{col_prefix} must be univariate: a 1D array or one column"
            )
        df_dict[col_prefix] = array if np.asarray(input_data).ndim == 2 else input_data
        return col_prefix

    @staticmethod
    def _validate_input_types(x: Any, y: Any, data: Optional[Frame] = None) -> None:
        """
        Validate input types for x and y before array processing.

        Parameters
        ----------
        x : any
            The x input data.
        y : any
            The y input data.
        data : pl.DataFrame, optional
            DataFrame if provided.

        Raises
        ------
        ValueError
            If x/y are string column names but data=None.
        TypeError
            If x/y are dicts or other invalid types.
        """
        if data is None:
            # When data=None, x and y must be array-like, not column names
            if isinstance(x, str):
                raise ValueError(
                    f"Column name '{x}' provided but data=None. "
                    "When data=None, provide array-like values, not column names."
                )
            if isinstance(y, str):
                raise ValueError(
                    f"Column name '{y}' provided but data=None. "
                    "When data=None, provide array-like values, not column names."
                )

        # Reject dict inputs
        if isinstance(x, dict):
            raise TypeError(
                "Dictionary input is not supported for x. "
                "Provide array-like values (list, ndarray, Series) instead."
            )
        if isinstance(y, dict):
            raise TypeError(
                "Dictionary input is not supported for y. "
                "Provide array-like values (list, ndarray, Series) instead."
            )

    @staticmethod
    def _validate_arrays(
        x: Any, y: Any, array_name_x: str = "x", array_name_y: str = "y"
    ) -> tuple[Any, Any]:
        """
        Validate x and y arrays for non-empty and matching lengths.

        Parameters
        ----------
        x : array-like
            The x input data.
        y : array-like
            The y input data.
        array_name_x : str, optional
            Display name for x in error messages.
        array_name_y : str, optional
            Display name for y in error messages.

        Returns
        -------
        tuple
            (x, y) if validation passes

        Raises
        ------
        ValueError
            If arrays are empty or have mismatched lengths.
        """
        # Check for empty arrays
        try:
            len_x = len(x)
        except (TypeError, AttributeError):
            raise TypeError(f"{array_name_x} must be array-like with a length")

        try:
            len_y = len(y)
        except (TypeError, AttributeError):
            raise TypeError(f"{array_name_y} must be array-like with a length")

        if len_x == 0:
            raise ValueError(f"Cannot process empty {array_name_x} array")

        if len_y == 0:
            raise ValueError(f"Cannot process empty {array_name_y} array")

        # Check for mismatched lengths
        if len_x != len_y:
            raise ValueError(
                f"Length mismatch: {array_name_x} has length {len_x}, "
                f"but {array_name_y} has length {len_y}. Arrays must have equal length."
            )

        return x, y

    @staticmethod
    def _validate_single_array(
        arr: Any,
        array_name: str = "x",
        allow_empty: bool = False,
        numeric_only: bool = True,
    ) -> Any:
        """
        Validate a single array for array-like, non-empty, and optionally numeric content.

        Supports numpy arrays, Polars Series, and native Python sequences.
        Complex number check works with any numeric type that numpy can interpret.

        Parameters
        ----------
        arr : array-like
            The input array to validate.
        array_name : str, optional
            Display name for the array in error messages.
        allow_empty : bool, optional
            If True, allow empty arrays. Default False.
        numeric_only : bool, optional
            If True, validate that array contains only numeric values. Default True.

        Returns
        -------
        arr
            The validated array.

        Raises
        ------
        TypeError
            If array is not array-like, contains non-numeric values, or is invalid type.
        ValueError
            If array is empty (when allow_empty=False).
        """
        # Check if array-like
        if not BaseUtils._is_array_like(arr):
            raise TypeError(
                f"{array_name} must be array-like (e.g., list, ndarray, Series), "
                f"got {type(arr).__name__}"
            )

        # Check for empty
        try:
            len_arr = len(arr)
        except (TypeError, AttributeError):
            raise TypeError(f"{array_name} must support len() operation")

        if len_arr == 0 and not allow_empty:
            raise ValueError(f"Cannot process empty {array_name} array")

        # Check for numeric content (if not empty and numeric_only=True)
        if numeric_only and len_arr > 0:
            try:
                import numpy as np

                np_arr = np.asarray(arr)

                # Check for complex numbers
                if np.iscomplexobj(np_arr):
                    raise TypeError(
                        f"Complex numbers are not supported in {array_name}"
                    )

                # Check that dtype is numeric
                if not np.issubdtype(np_arr.dtype, np.number):
                    raise TypeError(
                        f"{array_name} contains non-numeric values (dtype: {np_arr.dtype}). "
                        "Only numeric arrays are supported."
                    )
            except TypeError:
                raise
            except ImportError:
                # If numpy not available, skip numeric validation
                pass
            except Exception:
                # If validation fails, let it pass to Polars
                pass

        return arr

    def _prepare_data(
        self,
        x: Input,
        y: Input,
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
    ) -> tuple[pl.LazyFrame, str, str]:
        """
        Prepare and normalize data for analysis (similar to seaborn API).

        Supports two usage patterns:
        1. DataFrame mode: Provide a DataFrame with x/y as column names
        2. Array mode: Provide x/y as array-like without a DataFrame

        Parameters
        ----------
        x : str or array-like
            Feature(s). Column name(s) if `data` provided, else array-like (list, ndarray, Series).
        y : str or array-like
            Target(s). Column name(s) if `data` provided, else array-like (list, ndarray, Series).
        data : pl.DataFrame, pl.LazyFrame, or None, optional
            Input data. If None, x/y must be array-like.
            If provided, x/y are treated as column names.

        Returns
        -------
        tuple
            (snapshot as LazyFrame, internal x column name, internal y column name)

        Examples
        --------
        >>> import polars as pl
        >>> import numpy as np
        >>> from rgram.base import BaseUtils
        >>>
        >>> utils = BaseUtils()
        >>>
        >>> # Pattern 1: DataFrame with column names (like seaborn)
        >>> df = pl.DataFrame({"feature": [1, 2, 3], "target": [4, 5, 6]})
        >>> lf, x, y = utils._prepare_data(data=df, x="feature", y="target")
        >>>
        >>> # Pattern 2: Raw arrays (like seaborn without data parameter)
        >>> x_arr = np.array([1, 2, 3])
        >>> y_arr = np.array([4, 5, 6])
        >>> lf, x, y = utils._prepare_data(x=x_arr, y=y_arr)
        """
        if data is None:
            df_dict = {}

            # Validate input types first (checks for invalid string column names, etc.)
            self._validate_input_types(x, y, data)

            # Then validate array lengths and non-empty
            x, y = self._validate_arrays(x, y, "x", "y")

            # Process and validate arrays (includes numeric validation)
            x = self._process_array_input(x, "x", df_dict)
            y = self._process_array_input(y, "y", df_dict)

            data = pl.DataFrame(df_dict)

        if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError("data must be a Polars DataFrame or LazyFrame")
        if not isinstance(x, str) or not isinstance(y, str):
            raise ValueError(
                "fit only supports univariate input: x and y must be single column names"
            )
        # Snapshot exactly the selected columns once; never filter or impute rows.
        frame = data.lazy().select(pl.col(x).alias("x"), pl.col(y).alias("y")).collect()
        for name in ("x", "y"):
            col = frame[name]
            if not col.dtype.is_numeric():
                raise TypeError(f"{name} must contain numeric values")
            if frame.height == 0:
                raise ValueError("Cannot process empty data")
            if col.null_count() or not np.isfinite(col.to_numpy()).all():
                raise ValueError(
                    f"{name} contains null, NaN, or infinite values; no rows were dropped"
                )
        return frame.lazy(), "x", "y"

    @staticmethod
    def _prediction_array(values: Input) -> Array:
        """Validate and flatten one finite numeric query/weight vector without reordering."""
        BaseUtils._validate_single_array(values, "x")
        array = np.asarray(values)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError("x must be univariate: a 1D array or one column")
        if not np.isfinite(array).all():
            raise ValueError("x contains NaN or infinite values; no rows were dropped")
        return array

    def __sklearn_is_fitted__(self) -> bool:
        """Report successful-fit state to sklearn, independently of partially assigned attributes."""
        return getattr(self, "_is_fitted", False)

    def _warn(self, message: str, category: Optional[type[Warning]] = None) -> None:
        """Emit a standard Python advisory with the supplied category and caller-facing stack level."""
        import warnings

        from rgram.warnings import RgramWarning

        # Warning presentation belongs to Python's warning filters, not model state.
        warnings.warn(message, category or RgramWarning, stacklevel=3)

    def _check_order(self, values: Input) -> None:
        """Warn on decreasing adjacent feature values, leaving all rows unchanged."""
        from rgram.warnings import UnsortedInputWarning

        if np.any(values[1:] < values[:-1]):
            self._warn(
                "X is not sorted. No rows were sorted or reordered; predictions retain "
                "input order. Sort paired values yourself if you need an ordered line plot.",
                UnsortedInputWarning,
            )

    @staticmethod
    def _feature_name(values: Input, data: Optional[Frame] = None) -> Optional[str]:
        """Extract a selected column, one-column frame, or nonempty Series name when available."""
        if data is not None:
            return values if isinstance(values, str) else None
        columns = getattr(values, "columns", None)
        if columns is not None and len(columns) == 1 and isinstance(columns[0], str):
            return columns[0]
        name = getattr(values, "name", None)
        return name if isinstance(name, str) and name else None

    def _record_feature_name(self, values: Input, data: Optional[Frame] = None) -> None:
        """Replace or clear the fitted feature name when training input changes."""
        self.__dict__.pop("feature_names_in_", None)
        name = self._feature_name(values, data)
        if name is not None:
            self.feature_names_in_ = np.asarray([name], dtype=object)

    def _prediction_features(self, values: Input) -> Array:
        """Validate one query feature and enforce/warn about fitted feature-name compatibility."""
        name = self._feature_name(values)
        if (
            name is not None
            and hasattr(self, "feature_names_in_")
            and name != self.feature_names_in_[0]
        ):
            raise ValueError(
                f"Feature name {name!r} does not match fitted feature {self.feature_names_in_[0]!r}"
            )
        fitted_name = hasattr(self, "feature_names_in_")
        if (name is not None) != fitted_name:
            import warnings

            warnings.warn(
                "X does not have valid feature names, but the estimator was fitted with feature names"
                if fitted_name
                else "X has feature names, but the estimator was fitted without feature names",
                UserWarning,
                stacklevel=3,
            )
        return self._prediction_array(values)

    @staticmethod
    def _float_array(values: Input, name: str) -> FloatArray:
        """Make an explicit float64 computation buffer; reject precision loss."""
        array = np.asarray(values)
        with np.errstate(over="ignore", invalid="ignore"):
            converted = array.astype(np.float64, copy=True)
        if not np.isfinite(converted).all():
            raise ValueError(
                f"{name} cannot be represented as finite float64; rescale explicitly"
            )
        if np.issubdtype(array.dtype, np.integer):
            large = np.abs(converted) >= 2**53
            if any(
                int(original) != int(value)
                for original, value in zip(array[large], converted[large])
            ):
                raise ValueError(
                    f"{name} would lose integer precision in float64; rescale explicitly"
                )
        elif np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 8:
            if not np.array_equal(converted.astype(array.dtype), array):
                raise ValueError(
                    f"{name} would lose precision in float64; convert explicitly"
                )
        return converted

    def _sample_weights(self, weights: Optional[ArrayLike], n_samples: int) -> Array:
        """Copy aligned finite nonnegative weights with positive total mass, or create unit weights."""
        if weights is None:
            return np.ones(n_samples, dtype=float)
        weights = self._prediction_array(weights).copy()
        self._float_array(weights, "sample_weight")
        if len(weights) != n_samples or (weights < 0).any() or not (weights > 0).any():
            raise ValueError(
                "sample_weight must match training rows, be nonnegative and have positive total mass"
            )
        return weights

    def _data_summary(self) -> dict[str, Union[int, float, str, bool]]:
        """Describe the fitted snapshots and explicit data-integrity/computation conventions."""
        return {
            "n_rows_received": self.n_samples_in_,
            "n_rows_retained": self.n_samples_in_,
            "rows_sorted": False,
            "rows_dropped": 0,
            "values_imputed": 0,
            "x_dtype": str(self.X_.dtype),
            "y_dtype": str(self.y_.dtype),
            "computation_dtype": "float64",
            "sample_weight_provided": self.sample_weight_provided_,
            "weights_modified_in_snapshot": False,
        }

    def regression_diagnostics(self, x: Input, y: Input) -> pl.DataFrame:
        """Inspect observed values, fitted values, and residuals in input order.

        Parameters
        ----------
        x : array-like of shape (n_samples,) or (n_samples, 1)
            One evaluation feature. Named inputs must match the fitted feature.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Observed response, aligned with x. May be training or held-out data.

        Returns
        -------
        diagnostics : polars.DataFrame
            Columns: x, observed, prediction, residual (observed minus prediction),
            supported (finite prediction). Unsupported rows remain present.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If no successful fit is available.
        ValueError
            For invalid, misaligned inputs or strict prediction policies.
        """
        check_is_fitted(self)
        values = self._prediction_features(x)
        target = self._prediction_array(y)
        self._validate_arrays(values, target)
        prediction = self.predict(x)
        return pl.DataFrame(
            {
                "x": values,
                "observed": target,
                "prediction": prediction,
                "residual": target - prediction,
                "supported": np.isfinite(prediction),
            }
        )

    def predict_interval(
        self,
        x: Input,
        *,
        confidence_level: float = 0.95,
        n_resamples: int = 200,
        random_state: Optional[int] = None,
        method: str = "percentile",
        resampling: str = "iid",
        groups: Optional[ArrayLike] = None,
        block_size: Optional[int] = None,
        min_valid_fraction: float = 1.0,
    ) -> pl.DataFrame:
        """Compute pointwise paired-bootstrap confidence intervals for the curve.

        Parameters
        ----------
        x : array-like of shape (n_queries,) or (n_queries, 1)
            Query locations; order and duplicates are preserved.
        confidence_level : float, default=0.95
            Nominal interval level strictly between zero and one.
        n_resamples : int, default=200
            Bootstrap refits, at least two. More draws improve Monte Carlo stability.
        random_state : int or None, default=None
            Seed for resampling, independent of any estimator CV seed.
        method : {'percentile', 'basic'}, default='percentile'
            Bootstrap quantiles, or quantiles reflected around the original estimate.
        resampling : {'iid', 'groups', 'blocks'}, default='iid'
            Resample paired independent rows, whole independent groups, or contiguous
            moving blocks in original training order. No sorting occurs.
        groups : array-like of shape (``n_samples_in_``,) or None, default=None
            Nonmissing group labels, required only for group resampling. At least
            two distinct groups are required.
        block_size : int or None, default=None
            Block length from 2 to ``n_samples_in_`` - 1, required only for blocks.
        min_valid_fraction : float, default=1.0
            Fraction of draws that must support each query, in (0, 1]. At least two
            valid draws are always required. Lower fractions condition on valid draws.

        Returns
        -------
        interval : polars.DataFrame
            Columns: x, prediction, lower, upper, n_valid, bootstrap_coverage.
            Insufficiently supported bounds are NaN. Queries are never dropped.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If no successful fit is available.
        ValueError
            For invalid options, fewer than two training rows, invalid group/block
            inputs, or errors from the estimator's strict fit/prediction policies.

        Notes
        -----
        These are curve intervals, not prediction intervals or simultaneous bands.
        IID draws assume independent random-design rows. Group and block draws require
        appropriate independent groups or ordered dependence. Smoothing settings are
        held fixed; bin boundaries are relearned. Selection uncertainty and smoothing
        bias are not corrected. Memory includes n_resamples * n_queries draw values.
        A draw with no positive observation weight counts as unsupported. Other fit
        errors propagate. Explicit neighbor-only algorithms can reject unsorted draws.
        """
        from rgram.uncertainty import bootstrap_interval

        return bootstrap_interval(
            self,
            x,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
            method=method,
            resampling=resampling,
            groups=groups,
            block_size=block_size,
            min_valid_fraction=min_valid_fraction,
        )
