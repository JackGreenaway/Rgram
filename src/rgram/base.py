from __future__ import annotations

import polars as pl
import polars_ols as pls  # noqa: F401

from typing import Sequence, Optional, Union, Any, Type, List, cast


class BaseUtils:
    """
    BaseUtils

    Utility base class for DataFrame-related utilities, such as list conversion and group-over operations.

    Methods
    -------
    _to_list(item)
        Convert a string or sequence to a list, or return None.
    _over_function(x)
        Apply a Polars expression over the 'hue' columns if present.
    """

    def __init__(
        self,
        hue: Sequence[str] | None = None,
    ) -> None:
        self.hue: List[str] = cast(List[str], self._to_list(hue) or [])

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
            return dataclass

        else:
            return None

    def _over_function(self, x: pl.Expr) -> pl.Expr:
        """
        Apply a Polars expression over the 'hue' columns if present.

        Parameters
        ----------
        x : pl.Expr
            The Polars expression to apply.
        Returns
        -------
        pl.Expr
            The possibly grouped expression.
        """
        if hasattr(self, "hue") and self.hue:
            return x.over(self.hue)

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
        """
        if isinstance(input_data, str):
            raise ValueError(
                f"Column name '{input_data}' provided but data=None. "
                "When data=None, provide array-like values, not column names."
            )

        if not BaseUtils._is_array_like(input_data):
            raise ValueError(
                f"Input must be array-like (e.g., list, ndarray, Series), got {type(input_data).__name__}."
            )

        df_dict[col_prefix] = input_data
        return col_prefix

    def _prepare_data(
        self,
        x: Union[str, Sequence[Any]],
        y: Union[str, Sequence[Any]],
        data: Union[pl.DataFrame, pl.LazyFrame, None] = None,
        keys: Optional[Union[str, Sequence[Any]]] = None,
    ) -> tuple[
        pl.LazyFrame,
        Union[str, Sequence[str]],
        Union[str, Sequence[str]],
        Optional[Union[str, Sequence[str]]],
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
        keys : str or array-like, optional
            Optional grouping column(s). Column name if `data` provided, else array-like.

        Returns
        -------
        tuple
            (data as LazyFrame, x_col_names, y_col_names, keys_col_names)

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

            x = self._process_array_input(x, "x", df_dict)
            y = self._process_array_input(y, "y", df_dict)

            if keys is not None:
                keys = self._process_array_input(keys, "keys", df_dict)

            data = pl.DataFrame(df_dict)

        return data.lazy(), x, y, keys
