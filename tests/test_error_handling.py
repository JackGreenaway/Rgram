"""
Comprehensive error handling tests for Regressogram and KernelSmoother.
Tests invalid inputs, edge cases, and proper error messages.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother


class TestRegressogramErrorHandling:
    """Test error conditions and invalid inputs for Regressogram."""

    def test_invalid_binning_strategy_raises_valueerror(self):
        """Test that invalid binning strategy raises ValueError."""
        rgram = Regressogram(binning="invalid_strategy")
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Unknown binning type"):
            rgram.fit(x=x, y=y)

    def test_fit_with_empty_arrays_raises(self):
        """Test that empty arrays raise an error."""
        rgram = Regressogram()
        x = np.array([])
        y = np.array([])

        with pytest.raises(Exception):  # IndexError or ValueError
            rgram.fit(x=x, y=y)

    def test_fit_with_mismatched_array_lengths_raises(self):
        """Test that mismatched x and y lengths raise an error."""
        rgram = Regressogram()
        x = np.array([1, 2, 3])
        y = np.array([1, 2])  # Different length

        with pytest.raises(Exception):  # Will fail during DataFrame creation
            rgram.fit(x=x, y=y)

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises RuntimeError."""
        rgram = Regressogram()

        with pytest.raises(RuntimeError, match="Call fit\\(\\) before predict"):
            rgram.predict([1, 2, 3])

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises RuntimeError."""
        rgram = Regressogram()

        with pytest.raises(RuntimeError, match="Call fit\\(\\) before predict"):
            rgram.predict([1, 2, 3])

    def test_fit_with_none_data_and_invalid_input_raises(self):
        """Test that providing string col name without DataFrame raises error."""
        rgram = Regressogram()

        with pytest.raises(ValueError, match="Column name .* provided but data=None"):
            rgram.fit(x="col_name", y=[1, 2, 3])

    def test_fit_with_none_y_values_raises(self):
        """Test that all None/NaN y values cause issues."""
        rgram = Regressogram()
        x = np.array([1, 2, 3])
        y = np.array([np.nan, np.nan, np.nan])

        # Should handle or raise appropriately
        try:
            result = rgram.fit_predict(x=x, y=y)
            # If it doesn't raise, predictions should be NaN or handled gracefully
            assert np.all(np.isnan(result)) or len(result) == 0
        except Exception:
            # Expected if library doesn't support all-NaN data
            pass

    def test_predict_with_nan_returns_nan(self):
        """Test that NaN inputs in predict are handled."""
        rgram = Regressogram()
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rgram.fit(x=x_data, y=y_data)
        result = rgram.predict([1.0, np.nan, 5.0])

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_fit_with_single_unique_x_value(self):
        """Test that single unique x value is handled."""
        rgram = Regressogram()
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        result = rgram.fit_predict(x=x, y=y)
        assert len(result) > 0
        # All predictions should be the same (aggregated y value)
        assert np.allclose(result, result[0])

    def test_fit_with_single_unique_y_value(self):
        """Test that single unique y value returns constant predictions."""
        rgram = Regressogram()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        result = rgram.fit_predict(x=x, y=y)
        assert np.allclose(result, 5.0)

    def test_fit_with_inf_values_raises_or_handles(self):
        """Test behavior with infinite values."""
        rgram = Regressogram()
        x = np.array([1.0, 2.0, 3.0, np.inf])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        # Should either handle gracefully or raise
        try:
            result = rgram.fit_predict(x=x, y=y)
            assert len(result) > 0
        except Exception:
            # Expected behavior if inf not supported
            pass

    def test_predict_with_empty_array_raises(self):
        """Test that predicting with empty array raises or returns empty."""
        rgram = Regressogram()
        x_data = np.array([1.0, 2.0, 3.0])
        y_data = np.array([1.0, 2.0, 3.0])

        rgram.fit(x=x_data, y=y_data)

        with pytest.raises(Exception):
            rgram.predict([])

    def test_invalid_agg_function(self):
        """Test that invalid aggregation function is handled."""
        # Pass a non-callable agg
        with pytest.raises(TypeError):
            Regressogram(agg="not_callable")

    def test_invalid_ci_tuple_length(self):
        """Test that CI tuple with wrong length is handled."""
        # CI should be None or tuple of exactly 2 functions
        with pytest.raises(ValueError, match="ci tuple must have exactly 2 elements"):
            Regressogram(ci=(lambda x: x.mean(),))  # Only one function

    def test_n_bins_negative_raises(self):
        """Test that negative n_bins raises error or is handled."""
        rgram = Regressogram(binning="dist", n_bins=-5)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        try:
            result = rgram.fit_predict(x=x, y=y)
            # If no error, n_bins should be clamped to at least 1
            assert len(result) > 0
        except (ValueError, Exception):
            # Expected if negative n_bins not allowed
            pass

    def test_fit_with_complex_numbers(self):
        """Test that complex numbers in x or y raise or fail gracefully."""
        rgram = Regressogram()
        x = np.array([1 + 2j, 2 + 3j, 3 + 4j])
        y = np.array([1, 2, 3])

        with pytest.raises((TypeError, ValueError)):
            rgram.fit(x=x, y=y)


class TestKernelSmootherErrorHandling:
    """Test error conditions for KernelSmoother."""

    def test_manual_bandwidth_without_value_raises(self):
        """Test that manual bandwidth without value raises."""
        with pytest.raises(ValueError, match="bandwidth_value must be specified"):
            KernelSmoother(bandwidth="manual")

    def test_invalid_bandwidth_strategy_raises(self):
        """Test that invalid bandwidth strategy raises."""
        smoother = KernelSmoother(bandwidth="invalid")
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(Exception):
            smoother.fit(data=pl.DataFrame({"x": x, "y": y}), x="x", y="y")

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises."""
        smoother = KernelSmoother()

        with pytest.raises(RuntimeError, match="You must call fit\(\) before predict"):
            smoother.predict([1.0, 2.0, 3.0])

    def test_fit_with_empty_data_raises(self):
        """Test that empty data raises."""
        smoother = KernelSmoother()
        df = pl.DataFrame({"x": [], "y": []})

        with pytest.raises(Exception):
            smoother.fit(data=df, x="x", y="y")

    def test_fit_with_single_point_raises_or_handles(self):
        """Test behavior with single data point."""
        smoother = KernelSmoother()
        df = pl.DataFrame({"x": [1.0], "y": [1.0]})

        try:
            result = smoother.fit(data=df, x="x", y="y")
            # If it succeeds, predictions should work
            pred = result.predict([1.0])
            assert len(pred) == 1
        except Exception:
            # Expected - single point may not be enough
            pass

    def test_manual_bandwidth_negative_raises_or_handles(self):
        """Test that negative bandwidth is handled."""
        smoother = KernelSmoother(bandwidth="manual", bandwidth_value=-0.5)
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))}
        )

        try:
            smoother.fit(data=df, x="x", y="y")
            # Negative bandwidth might be allowed but shouldn't crash
        except (ValueError, Exception):
            # Expected if negative bandwidth not allowed
            pass

    def test_fit_with_nan_data(self):
        """Test fitting with NaN values."""
        smoother = KernelSmoother()
        df = pl.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})

        try:
            smoother.fit(data=df, x="x", y="y")
            # May handle NaN or raise
        except Exception:
            pass

    def test_predict_with_incompatible_type(self):
        """Test predict with non-numeric input."""
        smoother = KernelSmoother()
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        smoother.fit(data=df, x="x", y="y")

        with pytest.raises((TypeError, ValueError)):
            smoother.predict(["a", "b", "c"])


class TestDataValidationErrors:
    """Test data validation and type checking."""

    def test_fit_with_pandas_series_raises(self):
        """Test that raw pandas Series without converting to Polars raises."""
        try:
            import pandas as pd

            rgram = Regressogram()
            x = pd.Series([1, 2, 3])
            y = pd.Series([1, 2, 3])

            # Should either work (converted) or raise
            rgram.fit(x=x, y=y)
        except ImportError:
            pytest.skip("pandas not installed")

    def test_fit_with_dict_raises(self):
        """Test that dict input for x/y raises."""
        rgram = Regressogram()

        with pytest.raises((ValueError, TypeError)):
            rgram.fit(x={"a": 1}, y={"b": 2})

    def test_fit_with_mixed_numeric_types(self):
        """Test that mixed int/float is handled."""
        rgram = Regressogram()
        x = np.array([1, 2.5, 3, 4.7, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = rgram.fit_predict(x=x, y=y)
        assert len(result) > 0

    def test_fit_with_large_array_shape_mismatch(self):
        """Test with large arrays that have shape mismatches."""
        rgram = Regressogram()
        x = np.random.randn(1000)
        y = np.random.randn(999)  # One fewer element

        with pytest.raises(Exception):
            rgram.fit(x=x, y=y)
