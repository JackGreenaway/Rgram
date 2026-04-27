"""
Data validation and type checking tests.
Ensures proper handling of various data types and input validation.
Also includes BaseUtils utility function tests.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother
from rgram.base import BaseUtils


class TestBaseUtilsDataConversion:
    """Test BaseUtils data conversion helper functions."""

    def test_to_list_with_none(self):
        """Test _to_list with None input."""
        assert BaseUtils._to_list(None) is None

    def test_to_list_with_string(self):
        """Test _to_list with single string."""
        assert BaseUtils._to_list("single") == ["single"]

    def test_to_list_with_list(self):
        """Test _to_list preserves lists."""
        assert BaseUtils._to_list([1, 2, 3]) == [1, 2, 3]

    def test_to_list_with_tuple(self):
        """Test _to_list converts tuples to lists."""
        assert BaseUtils._to_list((1, 2, 3)) == [1, 2, 3]

    def test_to_list_with_numpy_array(self):
        """Test _to_list converts numpy arrays to lists."""
        result = BaseUtils._to_list(np.array([1, 2]))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_to_list_with_generator(self):
        """Test _to_list with generator."""
        gen = (x for x in [1, 2, 3])
        result = BaseUtils._to_list(gen)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_to_list_preserves_order(self):
        """Test that _to_list preserves element order."""
        original = [3, 1, 4, 1, 5, 9]
        result = BaseUtils._to_list(original)
        assert result == original

    def test_to_list_empty_list(self):
        """Test _to_list with empty list."""
        assert BaseUtils._to_list([]) == []


class TestBaseUtilsArrayLikeDetection:
    """Test BaseUtils array-like type detection."""

    def test_is_array_like_true_cases(self):
        """Test _is_array_like returns True for array-like objects."""
        assert BaseUtils._is_array_like([1, 2, 3])
        assert BaseUtils._is_array_like((1, 2, 3))
        assert BaseUtils._is_array_like(np.array([1, 2, 3]))
        assert BaseUtils._is_array_like(pl.Series([1, 2, 3]))

    def test_is_array_like_false_cases(self):
        """Test _is_array_like returns False for non-array objects."""
        assert not BaseUtils._is_array_like("string")
        assert not BaseUtils._is_array_like(123)
        assert not BaseUtils._is_array_like(None)
        assert not BaseUtils._is_array_like(3.14)
        assert not BaseUtils._is_array_like({1, 2, 3})

    def test_is_array_like_empty_collections(self):
        """Test _is_array_like with empty collections."""
        assert BaseUtils._is_array_like([])
        assert BaseUtils._is_array_like(())
        assert BaseUtils._is_array_like(np.array([]))


class TestBaseUtilsArrayInput:
    """Test BaseUtils _process_array_input method."""

    def test_process_array_input_with_list(self):
        """Test _process_array_input with list."""
        utils = BaseUtils()
        df_dict = {}
        col_name = utils._process_array_input([1, 2, 3], "test", df_dict)
        assert col_name == "test"
        assert df_dict["test"] == [1, 2, 3]

    def test_process_array_input_with_numpy_array(self):
        """Test _process_array_input with numpy array."""
        utils = BaseUtils()
        df_dict = {}
        arr = np.array([1, 2, 3])
        col_name = utils._process_array_input(arr, "numpy_col", df_dict)
        assert col_name == "numpy_col"
        assert np.array_equal(df_dict["numpy_col"], arr)

    def test_process_array_input_with_polars_series(self):
        """Test _process_array_input with Polars Series."""
        utils = BaseUtils()
        df_dict = {}
        arr = pl.Series([1, 2, 3])
        col_name = utils._process_array_input(arr, "series_col", df_dict)
        assert col_name == "series_col"

    def test_process_array_input_rejects_string_without_data(self):
        """Test that string column name is rejected when data=None."""
        utils = BaseUtils()
        df_dict = {}
        with pytest.raises(ValueError, match="Column name .* provided but data=None"):
            utils._process_array_input("column_name", "x", df_dict)

    def test_process_array_input_rejects_invalid_type(self):
        """Test that non-array-like objects are rejected."""
        utils = BaseUtils()
        df_dict = {}
        with pytest.raises(ValueError, match="Input must be array-like"):
            utils._process_array_input(123, "x", df_dict)


class TestBaseUtilsDataPreparation:
    """Test BaseUtils _prepare_data method."""

    def test_prepare_data_with_dataframe(self):
        """Test _prepare_data with DataFrame."""
        df = pl.DataFrame({"feature": [1, 2, 3], "target": [10, 20, 30]})
        utils = BaseUtils()
        lf, x_col, y_col = utils._prepare_data(data=df, x="feature", y="target")

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "feature"
        assert y_col == "target"

    def test_prepare_data_with_arrays(self):
        """Test _prepare_data with array inputs."""
        utils = BaseUtils()
        x = [1, 2, 3]
        y = [4, 5, 6]
        lf, x_col, y_col = utils._prepare_data(data=None, x=x, y=y)

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "x"
        assert y_col == "y"

    def test_prepare_data_creates_proper_dataframe(self):
        """Test that _prepare_data creates proper DataFrame content."""
        utils = BaseUtils()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        lf, _, _ = utils._prepare_data(data=None, x=x, y=y)

        df = lf.collect()
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) == 3

    def test_prepare_data_with_lazyframe_input(self):
        """Test _prepare_data with LazyFrame input."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        lf_input = df.lazy()
        utils = BaseUtils()
        lf, x_col, y_col = utils._prepare_data(data=lf_input, x="x", y="y")

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "x"
        assert y_col == "y"


class TestBaseUtilsDataValidation:
    """Test BaseUtils data processing and validation (legacy tests)."""

    def test_to_list_with_various_types(self):
        """Test to_list with different input types."""
        assert BaseUtils._to_list(None) is None
        assert BaseUtils._to_list("single") == ["single"]
        assert BaseUtils._to_list([1, 2, 3]) == [1, 2, 3]
        assert BaseUtils._to_list((1, 2)) == [1, 2]
        result = BaseUtils._to_list(np.array([1, 2]))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_is_array_like_detection(self):
        """Test is_array_like detection of various types."""
        assert BaseUtils._is_array_like([1, 2, 3])
        assert BaseUtils._is_array_like((1, 2, 3))
        assert BaseUtils._is_array_like(np.array([1, 2, 3]))
        assert BaseUtils._is_array_like(pl.Series([1, 2, 3]))
        assert not BaseUtils._is_array_like("string")
        assert not BaseUtils._is_array_like(123)
        assert not BaseUtils._is_array_like(None)
        assert not BaseUtils._is_array_like(3.14)
        assert not BaseUtils._is_array_like({1, 2, 3})

    def test_is_array_like_empty_collections(self):
        """Test is_array_like with empty collections."""
        assert BaseUtils._is_array_like([])
        assert BaseUtils._is_array_like(())
        assert BaseUtils._is_array_like(np.array([]))


class TestRegressogramDataValidation:
    """Test Regressogram data validation."""

    def test_fit_returns_self_for_chaining(self):
        """Test that fit returns self for method chaining."""
        rgram = Regressogram()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        result = rgram.fit(x=x, y=y)
        assert result is rgram

    def test_fit_predict_returns_numpy_array(self):
        """Test that fit_predict returns numpy array."""
        rgram = Regressogram()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)

    def test_fit_predict_with_ci_returns_tuple(self):
        """Test that fit_predict with CI returns tuple."""
        rgram = Regressogram()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = rgram.fit_predict(x=x, y=y, return_ci=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        y_pred, y_ci_low, y_ci_high = result
        assert isinstance(y_pred, np.ndarray)

    def test_predict_output_dtype(self):
        """Test that predict output is numeric numpy array."""
        rgram = Regressogram()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        rgram.fit(x=x, y=y)
        result = rgram.predict(x)

        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.number)

    def test_predict_returns_array(self):
        """Test that predict returns numeric array."""
        rgram = Regressogram()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    """Test BaseUtils data processing and validation."""

    def test_to_list_with_various_types(self):
        """Test to_list with different input types."""
        assert BaseUtils._to_list(None) is None
        assert BaseUtils._to_list("single") == ["single"]
        assert BaseUtils._to_list([1, 2, 3]) == [1, 2, 3]
        assert BaseUtils._to_list((1, 2)) == [1, 2]
        # numpy arrays are converted to list of their elements
        result = BaseUtils._to_list(np.array([1, 2]))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_to_list_with_generator(self):
        """Test to_list with generator."""
        gen = (x for x in [1, 2, 3])
        result = BaseUtils._to_list(gen)
        assert isinstance(result, list)

    def test_to_list_preserves_order(self):
        """Test that to_list preserves element order."""
        original = [3, 1, 4, 1, 5, 9]
        result = BaseUtils._to_list(original)
        assert result == original

    def test_is_array_like_detection(self):
        """Test is_array_like detection of various types."""
        assert BaseUtils._is_array_like([1, 2, 3])
        assert BaseUtils._is_array_like((1, 2, 3))
        assert BaseUtils._is_array_like(np.array([1, 2, 3]))
        assert BaseUtils._is_array_like(pl.Series([1, 2, 3]))
        assert not BaseUtils._is_array_like("string")
        assert not BaseUtils._is_array_like(123)
        assert not BaseUtils._is_array_like(None)
        assert not BaseUtils._is_array_like(3.14)
        assert not BaseUtils._is_array_like({1, 2, 3})  # Set is not array-like

    def test_is_array_like_empty_collections(self):
        """Test is_array_like with empty collections."""
        assert BaseUtils._is_array_like([])
        assert BaseUtils._is_array_like(())
        assert BaseUtils._is_array_like(np.array([]))

    def test_process_array_input_with_list(self):
        """Test process_array_input with list."""
        utils = BaseUtils()
        df_dict = {}
        col_name = utils._process_array_input([1, 2, 3], "test", df_dict)
        assert col_name == "test"
        assert df_dict["test"] == [1, 2, 3]

    def test_process_array_input_with_numpy(self):
        """Test process_array_input with numpy array."""
        utils = BaseUtils()
        df_dict = {}
        arr = np.array([1, 2, 3])
        col_name = utils._process_array_input(arr, "numpy_col", df_dict)
        assert col_name == "numpy_col"
        assert np.array_equal(df_dict["numpy_col"], arr)

    def test_process_array_input_rejects_string_without_data(self):
        """Test that string is rejected when data=None."""
        utils = BaseUtils()
        df_dict = {}
        with pytest.raises(ValueError, match="Column name .* provided but data=None"):
            utils._process_array_input("column_name", "x", df_dict)

    def test_process_array_input_rejects_non_array_like(self):
        """Test that non-array-like objects are rejected."""
        utils = BaseUtils()
        df_dict = {}
        with pytest.raises(ValueError, match="Input must be array-like"):
            utils._process_array_input(123, "x", df_dict)

    def test_prepare_data_with_dataframe_input(self):
        """Test prepare_data with DataFrame."""
        df = pl.DataFrame({"feature": [1, 2, 3], "target": [10, 20, 30]})
        utils = BaseUtils()
        lf, x_col, y_col = utils._prepare_data(data=df, x="feature", y="target")

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "feature"
        assert y_col == "target"

    def test_prepare_data_with_arrays(self):
        """Test prepare_data with array inputs."""
        utils = BaseUtils()
        x = [1, 2, 3]
        y = [4, 5, 6]
        lf, x_col, y_col = utils._prepare_data(data=None, x=x, y=y)

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "x"
        assert y_col == "y"

    def test_prepare_data_converts_to_dataframe_content(self):
        """Test that prepare_data actually creates proper DataFrame."""
        utils = BaseUtils()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        lf, _, _ = utils._prepare_data(data=None, x=x, y=y)

        df = lf.collect()
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) == 3

    def test_prepare_data_with_lazyframe_input(self):
        """Test prepare_data with LazyFrame input."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        lf_input = df.lazy()
        utils = BaseUtils()
        lf, x_col, y_col = utils._prepare_data(data=lf_input, x="x", y="y")

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "x"
        assert y_col == "y"


class TestKernelSmootherDataValidation:
    """Test KernelSmoother data validation."""

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        smoother = KernelSmoother()
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
        )

        result = smoother.fit(data=df, x="x", y="y")
        assert result is smoother

    def test_predict_output_dtype(self):
        """Test predict output is numeric array."""
        smoother = KernelSmoother()
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
        )

        smoother.fit(data=df, x="x", y="y")
        result = smoother.predict([5.0])

        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.number)

    def test_fit_predict_returns_array(self):
        """Test fit_predict returns numpy array."""
        smoother = KernelSmoother()
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
        )

        result = smoother.fit_predict(data=df, x="x", y="y")

        assert isinstance(result, np.ndarray)

    # def test_fit_predict_with_ci_returns_tuple(self):
    #     """Test fit_predict with return_ci returns tuple."""
    #     smoother = KernelSmoother()
    #     df = pl.DataFrame(
    #         {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
    #     )

    #     result = smoother.fit_predict(data=df, x="x", y="y", return_ci=True)

    #     assert isinstance(result, tuple)
    #     assert len(result) == 3


class TestInputTypeHandling:
    """Test handling of different input types."""

    def test_polars_series_as_x(self):
        """Test using Polars Series as x input."""
        rgram = Regressogram()
        x = pl.Series([1, 2, 3, 4, 5])
        y = [1, 2, 3, 4, 5]

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_polars_series_as_y(self):
        """Test using Polars Series as y input."""
        rgram = Regressogram()
        x = [1, 2, 3, 4, 5]
        y = pl.Series([1, 2, 3, 4, 5])

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_mixed_int_and_float_arrays(self):
        """Test with mixed integer and float in arrays."""
        rgram = Regressogram()
        x = np.array([1, 2.5, 3, 4.7, 5])
        y = np.array([1.5, 2, 3.2, 4, 5.1])

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_output_is_float_regardless_of_input_type(self):
        """Test that predictions are numeric even with int input."""
        rgram = Regressogram()
        x = np.array([1, 2, 3, 4, 5], dtype=int)
        y = np.array([10, 20, 30, 40, 50], dtype=int)

        result = rgram.fit_predict(x=x, y=y)

        # Predictions should be numeric
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.number)


class TestPandasCoercion:
    """Test coercion and handling of pandas DataFrames and Series."""

    def test_pandas_series_as_x_array_mode(self):
        """Test that pandas Series is accepted as x in array mode."""
        import pandas as pd

        rgram = Regressogram()
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_pandas_series_as_y_array_mode(self):
        """Test that pandas Series is accepted as y in array mode."""
        import pandas as pd

        rgram = Regressogram()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_both_pandas_series(self):
        """Test with both x and y as pandas Series."""
        import pandas as pd

        rgram = Regressogram()
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_pandas_dataframe_with_column_names(self):
        """Test that pandas DataFrame can be used with column names."""
        import pandas as pd

        rgram = Regressogram()
        pd_df = pd.DataFrame(
            {"feature": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [2.0, 4.0, 6.0, 8.0, 10.0]}
        )

        # Convert to polars for fit since the method expects polars
        pl_df = pl.from_pandas(pd_df)
        result = rgram.fit_predict(data=pl_df, x="feature", y="target")
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_pandas_series_with_non_default_index(self):
        """Test pandas Series with custom index is properly coerced."""
        import pandas as pd

        rgram = Regressogram()
        # Custom index (not default 0,1,2,...)
        x = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        y = pd.Series([2.0, 4.0, 6.0], index=["x", "y", "z"])

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_pandas_series_float32(self):
        """Test pandas Series with float32 dtype."""
        import pandas as pd

        rgram = Regressogram()
        x = pd.Series([1.0, 2.0, 3.0], dtype="float32")
        y = pd.Series([2.0, 4.0, 6.0], dtype="float32")

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_pandas_series_int64(self):
        """Test pandas Series with int64 dtype."""
        import pandas as pd

        rgram = Regressogram()
        x = pd.Series([1, 2, 3, 4, 5], dtype="int64")
        y = pd.Series([2, 4, 6, 8, 10], dtype="int64")

        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_pandas_series_fit_and_predict_separately(self):
        """Test pandas Series with separate fit and predict calls."""
        import pandas as pd

        rgram = Regressogram()
        x_train = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_train = pd.Series([1.0, 2.0, 3.0, 4.0])
        x_test = pd.Series([1.5, 2.5, 3.5])

        rgram.fit(x=x_train, y=y_train)
        result = rgram.predict(x=x_test)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_pandas_series_with_nan_detection(self):
        """Test that fit fails gracefully with NaN in pandas Series."""
        import pandas as pd

        rgram = Regressogram()
        x = pd.Series([1.0, 2.0, np.nan, 4.0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0])

        # Should raise an error due to NaN validation
        with pytest.raises((ValueError, TypeError)):
            rgram.fit_predict(x=x, y=y)

    def test_kernel_smoother_with_pandas_series(self):
        """Test KernelSmoother with pandas Series."""
        import pandas as pd

        ks = KernelSmoother()
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = ks.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5


class TestDuplicatePreservation:
    """Test that duplicate values are preserved (not dropped)."""

    def test_duplicate_x_values_preserved_in_output(self):
        """Test that duplicate x values result in aggregated y values, not dropped."""
        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        result = rgram.fit_predict(x=x, y=y)
        # fit_predict returns predictions for each input point
        # x=1: mean of [10, 20, 30] = 20
        # x=2: mean of [40, 50] = 45
        # x=3: mean of [60] = 60
        assert len(result) == 6  # One prediction per input point
        # All three x=1 values should get the same aggregated prediction
        assert np.isclose(result[0], 20.0)  # x=1
        assert np.isclose(result[1], 20.0)  # x=1
        assert np.isclose(result[2], 20.0)  # x=1
        # x=2 values get their aggregation
        assert np.isclose(result[3], 45.0)  # x=2
        assert np.isclose(result[4], 45.0)  # x=2
        # x=3 value
        assert np.isclose(result[5], 60.0)  # x=3

    def test_all_duplicate_x_values(self):
        """Test with all x values being identical."""
        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        result = rgram.fit_predict(x=x, y=y)
        # All duplicates should aggregate to mean of all y values
        expected_mean = np.mean([10.0, 20.0, 30.0, 40.0, 50.0])
        assert np.isclose(result[0], expected_mean)

    def test_duplicate_y_values_preserved(self):
        """Test that duplicate y values are preserved in aggregation."""
        rgram = Regressogram(binning="int", agg=lambda s: s.count())
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
        y = np.array([5.0, 5.0, 5.0, 8.0, 8.0])  # Duplicate y values

        result = rgram.fit_predict(x=x, y=y)
        # Count should be 3 for x=1 and 2 for x=2, not 1
        assert np.isclose(result[0], 3.0)  # Three values at x=1
        assert np.isclose(result[3], 2.0)  # Two values at x=2

    def test_duplicate_pairs_preserved(self):
        """Test that exact duplicate (x, y) pairs are all counted."""
        rgram = Regressogram(binning="int", agg=lambda s: s.count())
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])  # Identical values

        result = rgram.fit_predict(x=x, y=y)
        # Should count all 4 values, not drop duplicates
        assert np.isclose(result[0], 4.0)

    def test_duplicates_with_sum_aggregation(self):
        """Test that duplicates are not dropped with sum aggregation."""
        rgram = Regressogram(binning="int", agg=lambda s: s.sum())
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([10.0, 20.0, 30.0])

        result = rgram.fit_predict(x=x, y=y)
        # Sum of [10, 20, 30] = 60
        assert np.isclose(result[0], 60.0)

    def test_duplicates_with_min_aggregation(self):
        """Test that duplicates are considered with min aggregation."""
        rgram = Regressogram(binning="int", agg=lambda s: s.min())
        x = np.array([1.0, 1.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0, 50.0])

        result = rgram.fit_predict(x=x, y=y)
        # Min of [10, 20, 30] = 10
        assert np.isclose(result[0], 10.0)
        # Min of [50] = 50
        assert np.isclose(result[-1], 50.0)

    def test_duplicates_with_max_aggregation(self):
        """Test that duplicates are considered with max aggregation."""
        rgram = Regressogram(binning="int", agg=lambda s: s.max())
        x = np.array([1.0, 1.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0, 50.0])

        result = rgram.fit_predict(x=x, y=y)
        # Max of [10, 20, 30] = 30
        assert np.isclose(result[0], 30.0)
        # Max of [50] = 50
        assert np.isclose(result[-1], 50.0)

    def test_fit_predict_preserves_duplicates_same_as_separate_fit_predict(self):
        """Test that duplicates are handled consistently in fit_predict vs fit then predict."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        # fit_predict
        rgram1 = Regressogram(binning="int", agg=lambda s: s.mean())
        result_fit_predict = rgram1.fit_predict(x=x, y=y)

        # fit then predict
        rgram2 = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram2.fit(x=x, y=y)
        result_separate = rgram2.predict(x=x)

        # Results should be identical
        assert np.allclose(result_fit_predict, result_separate)

    def test_duplicates_in_dataframe_mode(self):
        """Test that duplicates are preserved in DataFrame mode."""
        df = pl.DataFrame(
            {"x": [1.0, 1.0, 1.0, 2.0, 2.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}
        )

        rgram = Regressogram(binning="int", agg=lambda s: s.sum())
        result = rgram.fit_predict(data=df, x="x", y="y")

        # Sum for x=1: 10+20+30 = 60
        assert np.isclose(result[0], 60.0)
        # Sum for x=2: 40+50 = 90
        assert np.isclose(result[3], 90.0)

    def test_kernel_smoother_preserves_duplicates(self):
        """Test that KernelSmoother preserves duplicate x values."""
        ks = KernelSmoother()
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        result = ks.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        # Should have output for all 6 input points
        assert len(result) == 6
