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
        return (
            dataclass(**var_input)
            if isinstance(var_input, dict)
            else dataclass
            if isinstance(var_input, dataclass)
            else dataclass()
        )

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
