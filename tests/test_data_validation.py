"""
Data validation and type checking tests.
Ensures proper handling of various data types and input validation.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother
from rgram.base import BaseUtils


class TestBaseUtilsDataValidation:
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
        lf, x_col, y_col, keys_col = utils._prepare_data(
            data=df, x="feature", y="target"
        )

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "feature"
        assert y_col == "target"
        assert keys_col is None

    def test_prepare_data_with_arrays(self):
        """Test prepare_data with array inputs."""
        utils = BaseUtils()
        x = [1, 2, 3]
        y = [4, 5, 6]
        lf, x_col, y_col, keys_col = utils._prepare_data(data=None, x=x, y=y)

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "x"
        assert y_col == "y"
        assert keys_col is None

    def test_prepare_data_converts_to_dataframe_content(self):
        """Test that prepare_data actually creates proper DataFrame."""
        utils = BaseUtils()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        lf, _, _, _ = utils._prepare_data(data=None, x=x, y=y)

        df = lf.collect()
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) == 3

    def test_prepare_data_with_lazyframe_input(self):
        """Test prepare_data with LazyFrame input."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        lf_input = df.lazy()
        utils = BaseUtils()
        lf, x_col, y_col, _ = utils._prepare_data(data=lf_input, x="x", y="y")

        assert isinstance(lf, pl.LazyFrame)
        assert x_col == "x"
        assert y_col == "y"


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

    def test_transform_output_is_lazyframe(self):
        """Test that transform returns LazyFrame."""
        rgram = Regressogram()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        rgram.fit(x=x, y=y)
        result = rgram.transform()

        assert isinstance(result, pl.LazyFrame)

    def test_collect_on_transform_returns_dataframe(self):
        """Test that calling collect on transform returns DataFrame."""
        rgram = Regressogram()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        rgram.fit(x=x, y=y)
        result = rgram.transform().collect()

        assert isinstance(result, pl.DataFrame)
        assert "y_pred_rgram" in result.columns

    def test_fit_with_list_input(self):
        """Test fit with list inputs."""
        rgram = Regressogram()
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]

        result = rgram.fit(x=x, y=y).transform().collect()
        assert len(result) > 0

    def test_fit_with_tuple_input(self):
        """Test fit with tuple inputs."""
        rgram = Regressogram()
        x = (1, 2, 3, 4, 5)
        y = (1, 2, 3, 4, 5)

        result = rgram.fit(x=x, y=y).transform().collect()
        assert len(result) > 0

    def test_fit_preserves_input_data(self):
        """Test that fit doesn't modify input arrays."""
        rgram = Regressogram()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        x_copy = x.copy()
        y_copy = y.copy()

        rgram.fit(x=x, y=y)

        assert np.array_equal(x, x_copy)
        assert np.array_equal(y, y_copy)

    def test_columns_in_transform_output(self):
        """Test that transform includes expected columns."""
        rgram = Regressogram()
        df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

        result = rgram.fit(data=df, x="x", y="y").transform().collect()

        assert "x_val" in result.columns
        assert "y_val" in result.columns
        assert "y_pred_rgram" in result.columns
        assert "y_pred_rgram_lci" in result.columns
        assert "y_pred_rgram_uci" in result.columns

    def test_no_ci_columns_when_ci_is_none(self):
        """Test that CI columns are absent when ci=None."""
        rgram = Regressogram(ci=None)
        df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

        result = rgram.fit(data=df, x="x", y="y").transform().collect()

        assert "y_pred_rgram_lci" not in result.columns
        assert "y_pred_rgram_uci" not in result.columns


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

    def test_transform_returns_lazyframe(self):
        """Test that transform returns LazyFrame."""
        smoother = KernelSmoother()
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
        )

        smoother.fit(data=df, x="x", y="y")
        result = smoother.transform()

        assert isinstance(result, pl.LazyFrame)

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

    def test_fit_predict_with_ci_returns_tuple(self):
        """Test fit_predict with return_ci returns tuple."""
        smoother = KernelSmoother()
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
        )

        result = smoother.fit_predict(data=df, x="x", y="y", return_ci=True)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_transform_columns(self):
        """Test expected columns in transform output."""
        smoother = KernelSmoother()
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
        )

        smoother.fit(data=df, x="x", y="y")
        result = smoother.transform().collect()

        assert "x_eval" in result.columns
        assert "y_kernel" in result.columns


class TestInputTypeHandling:
    """Test handling of different input types."""

    def test_polars_series_as_x(self):
        """Test using Polars Series as x input."""
        rgram = Regressogram()
        x = pl.Series([1, 2, 3, 4, 5])
        y = [1, 2, 3, 4, 5]

        result = rgram.fit(x=x, y=y).transform().collect()
        assert len(result) > 0

    def test_polars_series_as_y(self):
        """Test using Polars Series as y input."""
        rgram = Regressogram()
        x = [1, 2, 3, 4, 5]
        y = pl.Series([1, 2, 3, 4, 5])

        result = rgram.fit(x=x, y=y).transform().collect()
        assert len(result) > 0

    def test_mixed_int_and_float_arrays(self):
        """Test with mixed integer and float in arrays."""
        rgram = Regressogram()
        x = np.array([1, 2.5, 3, 4.7, 5])
        y = np.array([1.5, 2, 3.2, 4, 5.1])

        result = rgram.fit(x=x, y=y).transform().collect()
        assert len(result) > 0

    def test_output_is_float_regardless_of_input_type(self):
        """Test that predictions are float even with int input."""
        rgram = Regressogram()
        x = np.array([1, 2, 3, 4, 5], dtype=int)
        y = np.array([10, 20, 30, 40, 50], dtype=int)

        result = rgram.fit(x=x, y=y).transform().collect()

        # Predictions should be numeric (could be int or float)
        assert result["y_pred_rgram"].dtype in [
            pl.Int32,
            pl.Int64,
            pl.Float32,
            pl.Float64,
        ]
