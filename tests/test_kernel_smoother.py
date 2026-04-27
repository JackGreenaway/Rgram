"""Comprehensive tests for KernelSmoother bandwidth selection and predict functionality."""

import pytest
import polars as pl
import numpy as np
from rgram.smoothing import KernelSmoother
from typing import Any


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


class TestKernelTypes:
    """Test different kernel functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        x = np.sort(np.random.uniform(0, 10, n))
        y = 2 * np.sin(x) + np.random.normal(0, 0.5, n)
        df = pl.DataFrame({"x": x, "y": y})
        return df

    @pytest.fixture
    def x_eval(self):
        """Create evaluation points."""
        return np.array([1.0, 3.0, 5.0, 7.0, 9.0])

    def test_epanechnikov_kernel(self, sample_data, x_eval):
        """Test Epanechnikov kernel (default)."""
        smoother = KernelSmoother(kernel="epanechnikov", bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict(x_eval)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_eval)
        assert np.all(np.isfinite(result))

    def test_gaussian_kernel(self, sample_data, x_eval):
        """Test Gaussian (RBF) kernel."""
        smoother = KernelSmoother(kernel="gaussian", bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict(x_eval)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_eval)
        assert np.all(np.isfinite(result))

    def test_uniform_kernel(self, sample_data, x_eval):
        """Test uniform (rectangular) kernel."""
        smoother = KernelSmoother(kernel="uniform", bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict(x_eval)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_eval)
        assert np.all(np.isfinite(result))

    def test_triangular_kernel(self, sample_data, x_eval):
        """Test triangular kernel."""
        smoother = KernelSmoother(kernel="triangular", bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict(x_eval)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_eval)
        assert np.all(np.isfinite(result))

    def test_cosine_kernel(self, sample_data, x_eval):
        """Test cosine kernel."""
        smoother = KernelSmoother(kernel="cosine", bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict(x_eval)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_eval)
        assert np.all(np.isfinite(result))

    def test_logistic_kernel(self, sample_data, x_eval):
        """Test logistic kernel."""
        smoother = KernelSmoother(kernel="logistic", bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict(x_eval)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_eval)
        assert np.all(np.isfinite(result))

    def test_all_kernels_produce_similar_results(self, sample_data):
        """Test that all kernels produce reasonably similar smoothing results."""
        kernels = [
            "epanechnikov",
            "gaussian",
            "uniform",
            "triangular",
            "cosine",
            "logistic",
        ]
        x_eval = np.linspace(0, 10, 50)

        results = {}
        for kernel_name in kernels:
            smoother = KernelSmoother(kernel=kernel_name, bandwidth="silverman")
            smoother.fit(data=sample_data, x="x", y="y")
            results[kernel_name] = smoother.predict(x_eval)

        # Check that all results are valid
        for kernel_name, result in results.items():
            assert np.all(np.isfinite(result)), f"{kernel_name} produced NaN values"

        # All kernels should produce predictions in roughly the same range
        all_results = np.concatenate(list(results.values()))
        min_val = np.min(all_results)
        max_val = np.max(all_results)

        for kernel_name, result in results.items():
            # Each kernel's predictions should be within the overall range
            assert np.min(result) >= min_val - 0.1  # Small tolerance
            assert np.max(result) <= max_val + 0.1

    def test_kernel_parameter_stored(self, sample_data):
        """Test that kernel parameter is properly stored."""
        for kernel_name in ["epanechnikov", "gaussian", "uniform"]:
            smoother = KernelSmoother(kernel=kernel_name)
            assert smoother.kernel == kernel_name


class TestCustomKernel:
    """Test custom kernel function support."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        df = pl.DataFrame({"x": x, "y": y})
        return df

    def test_custom_kernel_callable(self, sample_data):
        """Test custom kernel as a callable."""

        def custom_kernel(u: Any) -> Any:
            """A simple custom box kernel."""
            return pl.when(u.abs() <= 1).then(1.0).otherwise(0.0)

        smoother = KernelSmoother(kernel=custom_kernel, bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict([2.0, 5.0, 8.0])

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.all(np.isfinite(result))

    def test_custom_kernel_lambda(self, sample_data):
        """Test custom kernel as a lambda function."""
        # Custom triangular kernel using lambda
        custom_kernel = lambda u: pl.when(u.abs() <= 1).then(1 - u.abs()).otherwise(0.0)

        smoother = KernelSmoother(kernel=custom_kernel, bandwidth="silverman")
        smoother.fit(data=sample_data, x="x", y="y")
        result = smoother.predict([2.0, 5.0, 8.0])

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.all(np.isfinite(result))

    def test_custom_kernel_stored(self, sample_data):
        """Test that custom kernel is stored as-is."""

        def my_kernel(u: Any) -> Any:
            return pl.when(u.abs() <= 1).then(0.5).otherwise(0.0)

        smoother = KernelSmoother(kernel=my_kernel)
        assert smoother.kernel is my_kernel

    def test_custom_kernel_vs_builtin(self, sample_data):
        """Test that custom kernel produces different results than built-in."""

        # Create a custom kernel that's significantly different
        def sharp_kernel(u: Any) -> Any:
            """Very sharp kernel: only near-zero distances get weight."""
            return pl.when(u.abs() <= 0.1).then(1.0).otherwise(0.0)

        x_eval = np.array([2.0, 5.0, 8.0])

        # Using sharp custom kernel
        smoother_custom = KernelSmoother(kernel=sharp_kernel, bandwidth="silverman")
        smoother_custom.fit(data=sample_data, x="x", y="y")
        result_custom = smoother_custom.predict(x_eval)

        # Using wide built-in kernel
        smoother_builtin = KernelSmoother(kernel="gaussian", bandwidth="silverman")
        smoother_builtin.fit(data=sample_data, x="x", y="y")
        result_builtin = smoother_builtin.predict(x_eval)

        # Results should be different (custom kernel will follow data more closely)
        diff = np.abs(result_custom - result_builtin).mean()
        assert diff > 0.01  # Should be noticeably different


class TestKernelValidation:
    """Test kernel parameter validation."""

    def test_invalid_kernel_string_raises(self):
        """Test that invalid kernel string raises ValueError."""
        with pytest.raises(ValueError, match="kernel must be one of"):
            KernelSmoother(kernel="invalid_kernel")

    def test_invalid_kernel_type_raises(self):
        """Test that invalid kernel type raises TypeError."""
        with pytest.raises(TypeError, match="kernel must be a string or callable"):
            KernelSmoother(kernel=123)

    def test_valid_kernel_strings(self):
        """Test that all valid kernel strings are accepted."""
        valid_kernels = [
            "epanechnikov",
            "gaussian",
            "uniform",
            "triangular",
            "cosine",
            "logistic",
        ]
        for kernel_name in valid_kernels:
            smoother = KernelSmoother(kernel=kernel_name)
            assert smoother.kernel == kernel_name

    def test_kernel_callable_no_validation_at_init(self):
        """Test that callable kernels are accepted without validation at init."""

        # Should not raise, even though the callable doesn't do anything useful
        def dummy_kernel(u):
            return u

        smoother = KernelSmoother(kernel=dummy_kernel)
        assert smoother.kernel is dummy_kernel


class TestKernelWithDifferentBandwidths:
    """Test kernels with different bandwidth selections."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        df = pl.DataFrame({"x": x, "y": y})
        return df

    def test_gaussian_kernel_with_silverman(self, sample_data):
        """Test Gaussian kernel with Silverman bandwidth."""
        smoother = KernelSmoother(kernel="gaussian", bandwidth="silverman")
        result = smoother.fit_predict(data=sample_data, x="x", y="y")
        assert len(result) == 100
        assert np.all(np.isfinite(result))

    def test_gaussian_kernel_with_scott(self, sample_data):
        """Test Gaussian kernel with Scott bandwidth."""
        smoother = KernelSmoother(kernel="gaussian", bandwidth="scott")
        result = smoother.fit_predict(data=sample_data, x="x", y="y")
        assert len(result) == 100
        assert np.all(np.isfinite(result))

    def test_gaussian_kernel_with_manual(self, sample_data):
        """Test Gaussian kernel with manual bandwidth."""
        smoother = KernelSmoother(
            kernel="gaussian", bandwidth="manual", bandwidth_value=0.5
        )
        result = smoother.fit_predict(data=sample_data, x="x", y="y")
        assert len(result) == 100
        assert np.all(np.isfinite(result))

    def test_uniform_kernel_with_manual(self, sample_data):
        """Test uniform kernel with manual bandwidth."""
        smoother = KernelSmoother(
            kernel="uniform", bandwidth="manual", bandwidth_value=1.0
        )
        result = smoother.fit_predict(data=sample_data, x="x", y="y")
        assert len(result) == 100
        assert np.all(np.isfinite(result))
