from __future__ import annotations

import polars as pl

from typing import Sequence, Optional, Union, Any, Type


class BaseUtils:
    """
    BaseUtils

    Utility base class for DataFrame-related utilities, such as list conversion and group-over operations.

    Methods
    -------
    _to_list(item)
        Convert a string or sequence to a list, or return None.
    _over_function(x)
        Apply a Polars expression over grouping columns if present.
    """

    def __init__(
        self,
    ) -> None:
        pass

    @staticmethod
    def _to_list(item: Optional[Union[str, Sequence]]) -> Optional[list]:
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
        df_dict: dict,
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

        df_dict[col_prefix] = input_data
        return col_prefix

    @staticmethod
    def _validate_input_types(
        x: Any, y: Any, data: Optional[pl.DataFrame] = None
    ) -> None:
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
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
    ) -> tuple[
        pl.LazyFrame,
        Union[str, Sequence[str]],
        Union[str, Sequence[str]],
    ]:
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
            (data as LazyFrame, x_col_names, y_col_names, None)

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
        >>> lf, x, y, k = utils._prepare_data(data=df, x="feature", y="target")
        >>>
        >>> # Pattern 2: Raw arrays (like seaborn without data parameter)
        >>> x_arr = np.array([1, 2, 3])
        >>> y_arr = np.array([4, 5, 6])
        >>> lf, x, y, k = utils._prepare_data(x=x_arr, y=y_arr)
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

        return data.lazy(), x, y
