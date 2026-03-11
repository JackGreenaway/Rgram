import pytest
import polars as pl
import numpy as np
from rgram.base import BaseUtils


def test_to_list_with_none():
    assert BaseUtils._to_list(None) is None


def test_to_list_with_string():
    assert BaseUtils._to_list("col") == ["col"]


def test_to_list_with_list():
    assert BaseUtils._to_list([1, 2, 3]) == [1, 2, 3]


def test_to_list_with_tuple():
    assert BaseUtils._to_list((1, 2)) == [1, 2]


def test_to_list_with_numpy_array():
    arr = np.array([1, 2, 3])
    result = BaseUtils._to_list(arr)
    assert isinstance(result, list)
    assert len(result) == 3


def test_to_list_empty_list():
    assert BaseUtils._to_list([]) == []


def test_is_array_like_true():
    assert BaseUtils._is_array_like([1, 2, 3])
    assert BaseUtils._is_array_like((1, 2))
    assert BaseUtils._is_array_like(np.array([1, 2]))
    assert BaseUtils._is_array_like(pl.Series([1, 2]))


def test_is_array_like_false():
    assert not BaseUtils._is_array_like(123)
    assert not BaseUtils._is_array_like("string")
    assert not BaseUtils._is_array_like(None)
    assert not BaseUtils._is_array_like(3.14)


def test_is_array_like_empty_list():
    assert BaseUtils._is_array_like([])


def test_over_function_without_hue():
    utils = BaseUtils()
    expr = pl.col("y")
    result = utils._over_function(expr)

    # It should return the exact same Expr object
    assert result is expr


def test_process_array_input_with_list():
    utils = BaseUtils()
    df_dict = {}
    arr = [1, 2, 3]
    col_name = utils._process_array_input(arr, "x", df_dict)

    assert col_name == "x"
    assert df_dict["x"] == arr


def test_process_array_input_with_numpy_array():
    utils = BaseUtils()
    df_dict = {}
    arr = np.array([1, 2, 3])
    col_name = utils._process_array_input(arr, "features", df_dict)

    assert col_name == "features"
    assert np.array_equal(df_dict["features"], arr)


def test_process_array_input_with_polars_series():
    utils = BaseUtils()
    df_dict = {}
    arr = pl.Series([1, 2, 3])
    col_name = utils._process_array_input(arr, "data", df_dict)

    assert col_name == "data"


def test_process_array_input_invalid_type():
    utils = BaseUtils()
    df_dict = {}

    with pytest.raises(ValueError):
        utils._process_array_input(123, "x", df_dict)


def test_process_array_input_string_when_data_none():
    utils = BaseUtils()
    df_dict = {}

    with pytest.raises(ValueError):
        utils._process_array_input("col_name", "x", df_dict)


def test_process_array_input_empty_array():
    utils = BaseUtils()
    df_dict = {}
    arr = []
    col_name = utils._process_array_input(arr, "empty", df_dict)

    assert col_name == "empty"
    assert df_dict["empty"] == []


def test_prepare_data_with_arrays_no_keys():
    utils = BaseUtils()
    x = [1, 2, 3]
    y = [4, 5, 6]
    lf, x_col, y_col = utils._prepare_data(x, y, data=None)

    assert isinstance(lf, pl.LazyFrame)
    assert x_col == "x"
    assert y_col == "y"


def test_prepare_data_with_lazyframe():
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    lf_input = df.lazy()
    utils = BaseUtils()
    lf, x_col, y_col = utils._prepare_data("x", "y", data=lf_input)

    assert isinstance(lf, pl.LazyFrame)
    assert x_col == "x"
    assert y_col == "y"


def test_prepare_data_with_numpy_arrays():
    utils = BaseUtils()
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    lf, x_col, y_col = utils._prepare_data(x, y, data=None)

    assert isinstance(lf, pl.LazyFrame)
    assert x_col == "x"
    assert y_col == "y"


def test_prepare_data_collected_dataframe_content():
    """Test that prepared data contains correct values."""
    utils = BaseUtils()
    x = [1, 2, 3]
    y = [10, 20, 30]
    lf, x_col, y_col = utils._prepare_data(x, y, data=None)

    df = lf.collect()
    assert list(df[x_col]) == x
    assert list(df[y_col]) == y


def test_prepare_data_with_multiple_column_names():
    """Test that multiple column names are handled."""
    df = pl.DataFrame({"x1": [1, 2], "x2": [3, 4], "y1": [5, 6], "y2": [7, 8]})
    utils = BaseUtils()
    lf, x_cols, y_cols = utils._prepare_data(["x1", "x2"], ["y1", "y2"], data=df)

    assert isinstance(lf, pl.LazyFrame)
    assert x_cols == ["x1", "x2"]
    assert y_cols == ["y1", "y2"]
