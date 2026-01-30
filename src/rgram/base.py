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
        input_data: Sequence[int, float],
        col_prefix: str,
        df_dict: dict,
    ) -> Union[str, Sequence[str]]:
        """
        Process a single array-like input and add to df_dict if needed.

        Parameters
        ----------
        input_data : str, sequence of str, or array-like
            The input to process.
        col_prefix : str
            Prefix for auto-generated column names (e.g., 'x', 'y', 'keys').
        df_dict : dict
            Dictionary to store arrays for DataFrame creation.

        Returns
        -------
        str or sequence of str
            Column name(s) for the processed input.
        """
        is_string = isinstance(input_data, str)
        is_array = BaseUtils._is_array_like(input_data) and not is_string

        if not is_array and not is_string:
            raise ValueError(
                f"Input must be a string (column name) or array-like, got {type(input_data)}."
            )

        if is_string:
            raise ValueError(
                "If data is None, input must be an array-like, not a column name string."
            )

        # Handle array-like input
        # input_list = input_data if isinstance(input_data, list) else [input_data]
        # input_list = [input_data]

        # col_names = []
        # for i, arr in enumerate(input_list):
        # col_name = col_prefix if len(input_list) == 1 else f"{col_prefix}_{i}"
        # df_dict[col_name] = arr
        # col_names.append(col_name)

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
        Prepare data by converting arrays to DataFrames if needed.

        If data is None; x, y, and keys are assumed to be array-like and will be
        converted to a Polars DataFrame with auto-generated column names.
        If data is provided, x, y, and keys are assumed to be column names.

        Parameters
        ----------
        data : pl.DataFrame, pl.LazyFrame, or None
            Input data. If None, x/y/keys are expected to be arrays.
        x : str, sequence of str, or array-like
            Feature(s). Column name(s) if data provided, else array(s).
        y : str, sequence of str, or array-like
            Target(s). Column name(s) if data provided, else array(s).
        keys : str, sequence of str, or array-like, optional
            Optional grouping variable(s).

        Returns
        -------
        tuple
            (data as LazyFrame, x, y, keys) where x, y, keys are column names.
        """
        if data is None:
            df_dict = {}

            x = self._process_array_input(x, "x", df_dict)
            y = self._process_array_input(y, "y", df_dict)

            if keys is not None:
                keys = self._process_array_input(keys, "keys", df_dict)

            data = pl.DataFrame(df_dict)

        return data.lazy(), x, y, keys
