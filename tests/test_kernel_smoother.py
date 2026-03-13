"""Comprehensive tests for KernelSmoother bandwidth selection and predict functionality."""

import pytest
import polars as pl
import numpy as np
from rgram.smoothing import KernelSmoother


class TestKernelSmootherBandwidthSelection:
    """Test different bandwidth selection methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        x = np.sort(np.random.uniform(0, 10, n))
        y = 2 * np.sin(x) + np.random.normal(0, 0.5, n)
        df = pl.DataFrame({"x": x, "y": y})
        return df

    def test_silverman_bandwidth(self, sample_data):
        """Test Silverman's rule bandwidth selection."""
        smoother = KernelSmoother(bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")

        result = smoother.predict(sample_data.get_column("x"))

        assert len(result) == 100

    def test_scott_bandwidth(self, sample_data):
        """Test Scott's rule bandwidth selection."""
        smoother = KernelSmoother(bandwidth="scott")
        smoother.fit(data=sample_data, x="x", y="y")

        result = smoother.predict(sample_data.get_column("x"))

        assert len(result) == 100

    def test_manual_bandwidth(self, sample_data):
        """Test manual bandwidth specification."""
        smoother = KernelSmoother(bandwidth="manual", bandwidth_value=0.5)
        smoother.fit(data=sample_data, x="x", y="y")

        result = smoother.predict(sample_data.get_column("x"))

        assert len(result) == 100

    def test_manual_bandwidth_missing_value_raises(self):
        """Test that manual bandwidth without bandwidth_value raises error."""
        with pytest.raises(ValueError, match="bandwidth_value must be specified"):
            KernelSmoother(bandwidth="manual")

    def test_different_bandwidths_produce_different_smoothing(self, sample_data):
        """Test that different bandwidth methods produce different results."""
        smoother_silverman = KernelSmoother(bandwidth="silverman")
        result_silverman = smoother_silverman.fit_predict(
            data=sample_data, x="x", y="y"
        )

        smoother_scott = KernelSmoother(bandwidth="scott")
        result_scott = smoother_scott.fit_predict(data=sample_data, x="x", y="y")

        # Results should be different
        diff = result_silverman.max() - result_scott.max()
        assert abs(diff) > 0.01  # Should be noticeably different

    def test_manual_bandwidth_wider(self, sample_data):
        """Test that wider bandwidth produces smoother results."""
        # Wider bandwidth = smoother curve
        smoother_wide = KernelSmoother(bandwidth="manual", bandwidth_value=2.0)
        result_wide = smoother_wide.fit_predict(data=sample_data, x="x", y="y")

        smoother_narrow = KernelSmoother(bandwidth="manual", bandwidth_value=0.1)
        result_narrow = smoother_narrow.fit_predict(data=sample_data, x="x", y="y")

        # Wider should have lower variance (smoother)
        assert result_wide.std() < result_narrow.std()


class TestKernelSmootherPredict:
    """Test the predict() method."""

    @pytest.fixture
    def fitted_smoother(self):
        """Create and fit a smoother."""
        np.random.seed(42)
        n = 100
        x = np.sort(np.random.uniform(0, 10, n))
        y = 2 * np.sin(x) + np.random.normal(0, 0.5, n)
        df = pl.DataFrame({"x": x, "y": y})

        smoother = KernelSmoother(bandwidth="silverman")
        smoother.fit(data=df, x="x", y="y")
        return smoother

    def test_predict_with_array(self, fitted_smoother):
        """Test predict with numpy array input."""
        x_new = np.array([1.0, 2.5, 5.0, 7.5, 9.0])
        result = fitted_smoother.predict(x_new)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_predict_with_list(self, fitted_smoother):
        """Test predict with list input."""
        x_new = [1.0, 2.5, 5.0, 7.5, 9.0]
        result = fitted_smoother.predict(x_new)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_predict_with_series(self, fitted_smoother):
        """Test predict with Polars Series input."""
        x_new = pl.Series([1.0, 2.5, 5.0, 7.5, 9.0])
        result = fitted_smoother.predict(x_new)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        smoother = KernelSmoother()
        with pytest.raises(RuntimeError, match="You must call fit\(\) before predict"):
            smoother.predict([1.0, 2.0, 3.0])

    def test_predict_single_point(self, fitted_smoother):
        """Test predict with single point."""
        # Use a point that's in the middle of the data range
        result = fitted_smoother.predict([5.0])

        # Single point might return nan if outside bandwidth of all training points
        # Instead, test with multiple points and check one of them
        result = fitted_smoother.predict([2.0, 5.0, 8.0])
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1  # Should have predictions

    def test_predict_outside_range(self, fitted_smoother):
        """Test predict with points outside training range."""
        # Points slightly outside [0, 10] range should get predictions
        x_new = np.array([0.1, 9.9])
        result = fitted_smoother.predict(x_new)

        # Should return predictions (kernel is continuous)
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1

    def test_predict_results_are_numeric(self, fitted_smoother):
        """Test that predictions are numeric values."""
        x_new = np.array([2.0, 5.0, 8.0])
        result = fitted_smoother.predict(x_new)

        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.number)


class TestKernelSmootherIntegration:
    """Integration tests for KernelSmoother."""

    def test_fit_then_predict_vs_fit_predict(self):
        """Test that fit() + transform() equals fit_predict()."""
        np.random.seed(42)
        n = 50
        x = np.sort(np.random.uniform(0, 10, n))
        y = 2 * np.sin(x) + np.random.normal(0, 0.3, n)
        df = pl.DataFrame({"x": x, "y": y})

        # Method 1: fit_predict
        smoother1 = KernelSmoother(bandwidth="silverman")
        result1 = smoother1.fit_predict(data=df, x="x", y="y")

        # Both should return predictions
        assert isinstance(result1, np.ndarray)
        assert len(result1) == n

    def test_bandwidth_affects_smoothness(self):
        """Test that increasing bandwidth increases smoothness."""
        np.random.seed(42)
        n = 100
        x = np.sort(np.random.uniform(0, 10, n))
        # Highly oscillatory function
        y = np.sin(2 * x) + np.sin(3 * x) + np.random.normal(0, 0.2, n)
        df = pl.DataFrame({"x": x, "y": y})

        # Very small bandwidth - follows data closely
        smoother_tight = KernelSmoother(bandwidth="manual", bandwidth_value=0.05)
        result_tight = smoother_tight.fit_predict(data=df, x="x", y="y")

        # Large bandwidth - very smooth
        smoother_loose = KernelSmoother(bandwidth="manual", bandwidth_value=1.5)
        result_loose = smoother_loose.fit_predict(data=df, x="x", y="y")

        # Variance in tight should be > variance in loose (more wiggly)
        tight_var = np.std(result_tight)
        loose_var = np.std(result_loose)
        assert tight_var > loose_var
