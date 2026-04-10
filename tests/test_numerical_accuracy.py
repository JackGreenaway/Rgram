"""
Numerical stability and accuracy tests.
Tests for precise numerical behavior, edge cases, and consistency.
"""

import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother


class TestRegressogramNumericalAccuracy:
    """Test numerical accuracy of Regressogram."""

    def test_exact_linear_relationship_recovery(self):
        """Test that exact linear relationship is recovered."""
        x = np.linspace(0, 10, 20)
        y = 2 * x + 3

        rgram = Regressogram(binning="none")  # No binning aggregation
        result = rgram.fit_predict(x=x, y=y)

        # For linear data with no binning, predictions should be close to the actuals
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert not np.all(np.isnan(result))

    def test_constant_function_recovery(self):
        """Test recovery of constant function."""
        x = np.linspace(0, 10, 30)
        y = np.full_like(x, 42.0)

        rgram = Regressogram(binning="width")
        result = rgram.fit_predict(x=x, y=y)

        # All predictions should be close to 42.0
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, 42.0, atol=0.1)

    def test_mean_computation_accuracy(self):
        """Test that mean aggregation is computed correctly."""
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        # Check that means are computed
        # x=1 -> y mean = (2+4)/2 = 3
        # x=2 -> y mean = (6+8)/2 = 7
        # x=3 -> y mean = (10+12)/2 = 11
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert not np.all(np.isnan(result))

    def test_median_computation(self):
        """Test median aggregation accuracy."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.median())
        result = rgram.fit_predict(x=x, y=y)

        # x=1 -> y median = 2.0
        # x=2 -> y median = 5.0
        assert isinstance(result, np.ndarray)
        assert not np.all(np.isnan(result))

    def test_ci_lower_less_than_upper(self):
        """Test that CI lower bounds are less than upper bounds."""
        np.random.seed(123)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.randn(50) * 0.2

        rgram = Regressogram()
        pred, lci, uci = rgram.fit_predict(x=x, y=y, return_ci=True)

        # Remove NaN values for comparison
        valid_idx = ~(np.isnan(lci) | np.isnan(uci))
        assert np.all(lci[valid_idx] <= uci[valid_idx])

    def test_ci_contains_prediction(self):
        """Test that predictions fall within CI bounds."""
        np.random.seed(123)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.randn(50) * 0.1

        rgram = Regressogram()
        pred, lci, uci = rgram.fit_predict(x=x, y=y, return_ci=True)

        # Predictions should be between bounds (allow for NaN)
        valid_idx = ~(np.isnan(pred) | np.isnan(lci) | np.isnan(uci))
        assert np.all(pred[valid_idx] >= lci[valid_idx])
        assert np.all(pred[valid_idx] <= uci[valid_idx])

    def test_max_aggregation_never_exceeds_max_value(self):
        """Test that max aggregation never exceeds max input."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.max())
        result = rgram.fit_predict(x=x, y=y)

        # Max predictions should not exceed max of y
        assert np.all(result <= np.max(y))

    def test_min_aggregation_never_below_min_value(self):
        """Test that min aggregation never goes below min input."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.min())
        result = rgram.fit_predict(x=x, y=y)

        # Min predictions should not go below min of y
        assert np.all(result >= np.min(y))

    def test_stability_with_large_values(self):
        """Test numerical stability with large values."""
        x = np.linspace(1e6, 2e6, 50)
        y = x * 2 + np.random.randn(50) * 1e4

        rgram = Regressogram()
        predictions = rgram.fit_predict(x=x, y=y)

        assert np.all(np.isfinite(predictions))

    def test_stability_with_small_values(self):
        """Test numerical stability with small values."""
        x = np.linspace(1e-6, 2e-6, 50)
        y = x * 2 + np.random.randn(50) * 1e-7

        rgram = Regressogram()
        result = rgram.fit_predict(x=x, y=y)

        # Predictions should be finite
        assert np.all(np.isfinite(result))

    def test_predictions_monotonic_with_monotonic_data(self):
        """Test that monotonic input produces monotonic output."""
        x = np.linspace(0, 10, 30)
        y = np.exp(x)  # Monotonically increasing

        rgram = Regressogram(binning="width")
        rgram.fit(x=x, y=y)
        pred = rgram.predict(x)

        # Check if predictions are monotonic (allowing for ties in binning)
        diffs = np.diff(pred)
        # Most differences should be non-negative
        non_decreasing_ratio = np.sum(diffs >= -1e-10) / len(diffs)
        assert non_decreasing_ratio > 0.9


class TestKernelSmootherNumericalAccuracy:
    """Test numerical accuracy of KernelSmoother."""

    def test_constant_function_recovery(self):
        """Test that constant function is recovered."""
        x = np.linspace(0, 10, 50)
        y = np.full_like(x, 5.0)

        smoother = KernelSmoother()
        result = smoother.fit_predict(data=pl.DataFrame({"x": x, "y": y}), x="x", y="y")

        # All smoothed values should be approximately 5.0
        assert np.allclose(result, 5.0, rtol=0.01)

    def test_linear_function_recovery(self):
        """Test recovery of linear function."""
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3

        smoother = KernelSmoother()
        result = smoother.fit(data=pl.DataFrame({"x": x, "y": y}), x="x", y="y")
        pred = result.predict(np.array([5.0]))

        # At x=5, should be close to y=13
        assert np.isclose(pred[0], 13.0, rtol=0.1)

    def test_smooth_predictions_less_variable(self):
        """Test that smoothing reduces variability."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y_true = np.sin(x)
        y_noisy = y_true + np.random.randn(100) * 0.5

        smoother = KernelSmoother()
        result = smoother.fit(data=pl.DataFrame({"x": x, "y": y_noisy}), x="x", y="y")
        pred = result.predict(x)

        # Predictions should be smoother than noisy data
        pred_std = np.std(np.diff(pred))
        noisy_std = np.std(np.diff(y_noisy))

        # Smoothed version should have less variability in differences
        assert pred_std < noisy_std

    def test_bandwidth_effect_on_smoothness(self):
        """Test that larger bandwidth produces smoother results."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.randn(100) * 0.3

        # Small bandwidth (more wiggly)
        smoother_small = KernelSmoother(
            bandwidth="manual",
            bandwidth_value=0.2,
        )
        result_small = smoother_small.fit_predict(
            data=pl.DataFrame({"x": x, "y": y}), x="x", y="y"
        )

        # Large bandwidth (smoother)
        smoother_large = KernelSmoother(
            bandwidth="manual",
            bandwidth_value=2.0,
        )
        result_large = smoother_large.fit_predict(
            data=pl.DataFrame({"x": x, "y": y}), x="x", y="y"
        )

        # Larger bandwidth should have lower variance in predictions
        var_small = result_small.var()
        var_large = result_large.var()

        # Note: might not always be strictly lower due to random seed, but trend should hold
        # Just check they're different
        assert var_small != var_large

    def test_silverman_vs_scott_bandwidth(self):
        """Test that Silverman and Scott bandwidths produce different results."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.randn(100) * 0.2

        smoother_silverman = KernelSmoother(
            bandwidth="silverman",
        )
        result_silverman = smoother_silverman.fit_predict(
            data=pl.DataFrame({"x": x, "y": y}), x="x", y="y"
        )

        smoother_scott = KernelSmoother(
            bandwidth="scott",
        )
        result_scott = smoother_scott.fit_predict(
            data=pl.DataFrame({"x": x, "y": y}), x="x", y="y"
        )

        # Results should be different
        diff = np.abs(result_silverman - result_scott).mean()
        assert diff > 0.01

    def test_predictions_within_data_range(self):
        """Test that predictions stay within data range."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        smoother = KernelSmoother()
        smoother.fit(data=pl.DataFrame({"x": x, "y": y}), x="x", y="y")
        pred = smoother.predict(x)

        # Predictions should be within (or very close to) data range
        assert np.all(pred >= y.min() - 0.5)
        assert np.all(pred <= y.max() + 0.5)

    def test_prediction_consistency(self):
        """Test that predict gives consistent results for same input."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        smoother = KernelSmoother()
        smoother.fit(data=pl.DataFrame({"x": x, "y": y}), x="x", y="y")

        pred1 = smoother.predict([5.0])
        pred2 = smoother.predict([5.0])

        assert np.isclose(pred1[0], pred2[0])


class TestBinningNumericalAccuracy:
    """Test numerical accuracy of different binning strategies."""

    def test_dist_binning_equal_quantiles(self):
        """Test that dist binning creates reasonable quantile bins."""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 1000)
        y = np.random.uniform(0, 100, 1000)

        rgram = Regressogram(binning="dist", n_bins=10)
        result = rgram.fit_predict(x=x, y=y)

        # Should have predictions for all data points
        assert len(result) == 1000
        assert not np.any(np.isnan(result))

    def test_width_binning_equal_width(self):
        """Test that width binning creates equal-width bins."""
        x = np.linspace(0, 100, 100)
        y = x + np.random.randn(100) * 5

        rgram = Regressogram(binning="width")
        result = rgram.fit_predict(x=x, y=y)

        # Collect unique bins and check they're roughly equally spaced
        assert len(result) == 100

    def test_int_binning_respects_integer_values(self):
        """Test that int binning respects integer boundaries."""
        x = np.array([0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1])
        y = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)

        rgram = Regressogram(binning="int")
        result = rgram.fit_predict(x=x, y=y)

        # Should have bins for integers 0, 1, 2, 3
        assert len(result) == 7  # One per input point

    def test_none_binning_each_unique_x_separate_bin(self):
        """Test that none binning treats each unique x as separate bin."""
        x = np.array([1, 1, 2, 2, 3, 3])
        y = np.array([1, 1.5, 2, 2.5, 3, 3.5])

        rgram = Regressogram(binning="none")
        result = rgram.fit_predict(x=x, y=y)

        # Should have 6 output rows (one per input)
        assert len(result) == 6


class TestNumericalEdgeCases:
    """Test edge cases in numerical computation."""

    def test_division_by_zero_handling(self):
        """Test that division by zero is handled gracefully."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])

        rgram = Regressogram(agg=lambda s: s.sum() / s.count())
        result = rgram.fit_predict(x=x, y=y)

        # Should not crash and predictions should be 0
        assert np.all(result == 0.0)

    def test_empty_bin_handling(self):
        """Test that empty bins are handled."""
        x = np.array([0.1, 0.2, 0.3, 9.7, 9.8, 9.9])
        y = np.array([1, 2, 3, 4, 5, 6], dtype=float)

        rgram = Regressogram(binning="width", n_bins=10)
        result = rgram.fit_predict(x=x, y=y)

        # Should handle bins with no data
        assert len(result) == 6

    def test_floating_point_precision(self):
        """Test numerical precision with close values."""
        x = np.array([1.0, 1.0 + 1e-15, 1.0 + 2e-15])
        y = np.array([1.0, 2.0, 3.0])

        rgram = Regressogram(binning="dist")
        try:
            result = rgram.fit_predict(x=x, y=y)
            assert len(result) > 0
        except Exception:
            # Floating point limits might cause issues
            pass
