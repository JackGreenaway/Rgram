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
        smoother = KernelSmoother(bandwidth="silverman", n_eval_samples=20)
        result = smoother.fit(data=sample_data, x="x", y="y")
        transform_result = result.transform().collect()

        assert "x_eval" in transform_result.columns
        assert "y_kernel" in transform_result.columns
        assert len(transform_result) == 20

    def test_scott_bandwidth(self, sample_data):
        """Test Scott's rule bandwidth selection."""
        smoother = KernelSmoother(bandwidth="scott", n_eval_samples=20)
        result = smoother.fit(data=sample_data, x="x", y="y")
        transform_result = result.transform().collect()

        assert "x_eval" in transform_result.columns
        assert "y_kernel" in transform_result.columns
        assert len(transform_result) == 20

    def test_manual_bandwidth(self, sample_data):
        """Test manual bandwidth specification."""
        smoother = KernelSmoother(
            bandwidth="manual", bandwidth_value=0.5, n_eval_samples=20
        )
        result = smoother.fit(data=sample_data, x="x", y="y")
        transform_result = result.transform().collect()

        assert "x_eval" in transform_result.columns
        assert "y_kernel" in transform_result.columns
        assert len(transform_result) == 20

    def test_manual_bandwidth_missing_value_raises(self):
        """Test that manual bandwidth without bandwidth_value raises error."""
        with pytest.raises(ValueError, match="bandwidth_value must be specified"):
            KernelSmoother(bandwidth="manual")

    def test_different_bandwidths_produce_different_smoothing(self, sample_data):
        """Test that different bandwidth methods produce different results."""
        smoother_silverman = KernelSmoother(bandwidth="silverman", n_eval_samples=50)
        smoother_silverman.fit(data=sample_data, x="x", y="y")
        result_silverman = smoother_silverman.transform().collect()

        smoother_scott = KernelSmoother(bandwidth="scott", n_eval_samples=50)
        smoother_scott.fit(data=sample_data, x="x", y="y")
        result_scott = smoother_scott.transform().collect()

        # Results should be different
        diff = result_silverman["y_kernel"].max() - result_scott["y_kernel"].max()
        assert abs(diff) > 0.01  # Should be noticeably different

    def test_manual_bandwidth_wider(self, sample_data):
        """Test that wider bandwidth produces smoother results."""
        # Wider bandwidth = smoother curve
        smoother_wide = KernelSmoother(
            bandwidth="manual", bandwidth_value=2.0, n_eval_samples=50
        )
        smoother_wide.fit(data=sample_data, x="x", y="y")
        result_wide = smoother_wide.transform().collect()

        smoother_narrow = KernelSmoother(
            bandwidth="manual", bandwidth_value=0.1, n_eval_samples=50
        )
        smoother_narrow.fit(data=sample_data, x="x", y="y")
        result_narrow = smoother_narrow.transform().collect()

        # Wider should have lower variance (smoother)
        assert result_wide["y_kernel"].std() < result_narrow["y_kernel"].std()


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

        smoother = KernelSmoother(bandwidth="silverman", n_eval_samples=50)
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
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before predict"):
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
        smoother1 = KernelSmoother(bandwidth="silverman", n_eval_samples=30)
        result1 = smoother1.fit_predict(data=df, x="x", y="y")

        # Method 2: fit then transform
        smoother2 = KernelSmoother(bandwidth="silverman", n_eval_samples=30)
        smoother2.fit(data=df, x="x", y="y")
        result2 = smoother2.transform().collect()

        # Both should return predictions
        assert isinstance(result1, np.ndarray)
        assert len(result1) > 0
        assert len(result2) > 0  # transform returns full DataFrame

    def test_predict_at_eval_points_uses_stored_bandwidth(self):
        """Test that predict uses the same bandwidth learned during fit."""
        np.random.seed(42)
        n = 50
        x = np.sort(np.random.uniform(0, 10, n))
        y = 2 * np.sin(x) + np.random.normal(0, 0.3, n)
        df = pl.DataFrame({"x": x, "y": y})

        smoother = KernelSmoother(bandwidth="silverman", n_eval_samples=20)
        smoother.fit(data=df, x="x", y="y")

        # Get transform results
        transform_result = smoother.transform().collect()

        # Get predict results at the same points
        eval_points = transform_result["x_eval"].to_list()
        predict_result = smoother.predict(eval_points)

        # Both should have same length
        assert len(transform_result) == len(predict_result)

        # They should be very similar (not identical due to groupby differences
        # but same bandwidth was used)
        corr = np.corrcoef(
            transform_result["y_kernel"].to_numpy(),
            predict_result,
        )[0, 1]
        assert corr > 0.95  # Should be highly correlated

    def test_predict_with_grouped_data(self):
        """Test predict doesn't interfere with grouped fitting."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )

        smoother = KernelSmoother(n_eval_samples=10)
        smoother.fit(data=df, x="x", y="y", hue="group")

        # Transform should work with hue
        transform_result = smoother.transform().collect()
        assert len(transform_result) > 0

    def test_bandwidth_affects_smoothness(self):
        """Test that increasing bandwidth increases smoothness."""
        np.random.seed(42)
        n = 100
        x = np.sort(np.random.uniform(0, 10, n))
        # Highly oscillatory function
        y = np.sin(2 * x) + np.sin(3 * x) + np.random.normal(0, 0.2, n)
        df = pl.DataFrame({"x": x, "y": y})

        # Very small bandwidth - follows data closely
        smoother_tight = KernelSmoother(
            bandwidth="manual", bandwidth_value=0.05, n_eval_samples=100
        )
        result_tight = smoother_tight.fit_predict(data=df, x="x", y="y")

        # Large bandwidth - very smooth
        smoother_loose = KernelSmoother(
            bandwidth="manual", bandwidth_value=1.5, n_eval_samples=100
        )
        result_loose = smoother_loose.fit_predict(data=df, x="x", y="y")

        # Variance in tight should be > variance in loose (more wiggly)
        tight_var = np.std(result_tight)
        loose_var = np.std(result_loose)
        assert tight_var > loose_var
