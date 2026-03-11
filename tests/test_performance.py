"""
Performance and stress tests.
Tests for scaling, memory efficiency, and robustness under load.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother
import time


class TestRegressogramPerformance:
    """Test Regressogram performance with various dataset sizes."""

    def test_fit_small_dataset(self):
        """Test fit performance on small dataset."""
        x = np.random.randn(10)
        y = np.random.randn(10)

        rgram = Regressogram()
        start = time.time()
        rgram.fit(x=x, y=y)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be very fast

    def test_fit_medium_dataset(self):
        """Test fit performance on medium dataset."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        rgram = Regressogram()
        start = time.time()
        rgram.fit(x=x, y=y)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should be reasonably fast

    def test_fit_large_dataset(self):
        """Test fit performance on large dataset."""
        x = np.random.randn(10000)
        y = np.random.randn(10000)

        rgram = Regressogram()
        start = time.time()
        rgram.fit(x=x, y=y)
        elapsed = time.time() - start

        assert elapsed < 30.0  # Should complete in reasonable time

    def test_predict_small_dataset(self):
        """Test predict performance."""
        x_train = np.random.randn(100)
        y_train = np.random.randn(100)

        x_test = np.random.randn(50)

        rgram = Regressogram()
        rgram.fit(x=x_train, y=y_train)

        start = time.time()
        pred = rgram.predict(x_test)
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert len(pred) == len(x_test)

    def test_batch_predictions(self):
        """Test making multiple batch predictions."""
        x_train = np.random.randn(500)
        y_train = np.random.randn(500)

        rgram = Regressogram()
        rgram.fit(x=x_train, y=y_train)

        batch_sizes = [10, 100, 1000]
        for batch_size in batch_sizes:
            x_test = np.random.randn(batch_size)
            pred = rgram.predict(x_test)
            assert len(pred) == batch_size

    def test_scaling_with_data_size(self):
        """Test that fitting time scales reasonably with data size."""
        sizes = [100, 500, 1000]
        times = []

        for size in sizes:
            x = np.random.randn(size)
            y = np.random.randn(size)

            rgram = Regressogram()
            start = time.time()
            rgram.fit(x=x, y=y)
            elapsed = time.time() - start
            times.append(elapsed)

        # Time should not increase exponentially
        ratio = times[-1] / times[0]
        assert ratio < 100  # Should be modest scaling


class TestKernelSmootherPerformance:
    """Test KernelSmoother performance."""

    def test_fit_small_dataset(self):
        """Test fit on small dataset."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 50), "y": np.sin(np.linspace(0, 10, 50))}
        )

        smoother = KernelSmoother(n_eval_samples=20)
        start = time.time()
        smoother.fit(data=df, x="x", y="y")
        elapsed = time.time() - start

        assert elapsed < 1.0

    def test_fit_large_dataset(self):
        """Test fit on large dataset."""
        df = pl.DataFrame({"x": np.random.randn(5000), "y": np.random.randn(5000)})

        smoother = KernelSmoother(n_eval_samples=100)
        start = time.time()
        smoother.fit(data=df, x="x", y="y")
        elapsed = time.time() - start

        assert elapsed < 30.0

    def test_predict_performance(self):
        """Test predict performance."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 200), "y": np.sin(np.linspace(0, 10, 200))}
        )

        smoother = KernelSmoother()
        smoother.fit(data=df, x="x", y="y")

        x_test = np.linspace(0, 10, 100)

        start = time.time()
        pred = smoother.predict(x_test)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert len(pred) == len(x_test)

    def test_many_eval_samples(self):
        """Test with many evaluation samples."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 100), "y": np.sin(np.linspace(0, 10, 100))}
        )

        for n_eval in [50, 100, 500]:
            smoother = KernelSmoother(n_eval_samples=n_eval)
            smoother.fit(data=df, x="x", y="y")
            result = smoother.transform().collect()
            assert len(result) == n_eval


class TestMemoryEfficiency:
    """Test memory efficiency and proper resource handling."""

    def test_lazy_evaluation(self):
        """Test that LazyFrames are actually lazy."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        rgram = Regressogram()
        rgram.fit(x=x, y=y)

        # transform returns LazyFrame
        lazy_result = rgram.transform()
        assert isinstance(lazy_result, pl.LazyFrame)

        # Only collect when needed
        result = lazy_result.collect()
        assert isinstance(result, pl.DataFrame)

    def test_no_memory_leak_repeated_fits(self):
        """Test that repeated fits don't accumulate memory."""
        x = np.random.randn(500)
        y = np.random.randn(500)

        # Fit multiple times
        for _ in range(10):
            rgram = Regressogram()
            rgram.fit(x=x, y=y)

        # Should complete without memory issues

    def test_large_output_handling(self):
        """Test handling of large output DataFrames."""
        x = np.linspace(0, 10, 5000)
        y = np.sin(x)

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 5000
        assert result.shape[0] == 5000


class TestConcurrentUsage:
    """Test models in concurrent-like scenarios."""

    def test_independent_models_independent_results(self):
        """Test that independent model instances don't interfere."""
        x = np.linspace(0, 10, 100)

        rgram1 = Regressogram(binning="dist", n_bins=10)
        rgram2 = Regressogram(binning="width")

        y1 = x**2
        y2 = x**3

        rgram1.fit(x=x, y=y1)
        rgram2.fit(x=x, y=y2)

        pred1 = rgram1.predict([5.0])
        pred2 = rgram2.predict([5.0])

        # Predictions should be different
        assert pred1[0] != pred2[0]

    def test_model_isolation(self):
        """Test that fitting one model doesn't affect another."""
        x1 = np.linspace(0, 5, 50)
        y1 = x1

        x2 = np.linspace(5, 10, 50)
        y2 = -x2

        rgram1 = Regressogram()
        rgram1.fit(x=x1, y=y1)
        pred1_before = rgram1.predict([2.5])

        # Fit a different model
        rgram2 = Regressogram()
        rgram2.fit(x=x2, y=y2)

        # First model's predictions should be unchanged
        pred1_after = rgram1.predict([2.5])
        assert np.isclose(pred1_before[0], pred1_after[0])


class TestStressConditions:
    """Test under stress conditions."""

    def test_extreme_value_ranges(self):
        """Test with extreme value ranges."""
        x = np.linspace(1e6, 1e7, 100)
        y = np.sin(x / 1e6)

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 100
        assert result["y_pred_rgram"].dtype in [pl.Float32, pl.Float64]

    def test_many_duplicate_values(self):
        """Test with many duplicate x values."""
        x = np.repeat(np.linspace(0, 10, 20), 50)  # 1000 points, 20 unique x
        y = np.tile(np.sin(np.linspace(0, 10, 20)), 50) + np.random.randn(1000) * 0.1

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 1000

    def test_highly_clustered_data(self):
        """Test with highly clustered data."""
        # Most points clustered at edges
        x = np.concatenate(
            [
                np.random.normal(0, 0.1, 400),
                np.random.normal(10, 0.1, 400),
                np.random.uniform(0, 10, 200),
            ]
        )
        y = np.sin(x)

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 1000

    def test_sparse_vs_dense_regions(self):
        """Test with sparse and dense regions."""
        x = np.concatenate(
            [
                np.linspace(0, 1, 100),  # Dense
                np.linspace(1.1, 9, 10),  # Sparse
                np.linspace(9.1, 10, 100),  # Dense
            ]
        )
        y = np.sin(x)

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 210

    def test_repeated_fit_predict_cycles(self):
        """Test repeated fit-predict cycles."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        for _ in range(5):
            rgram = Regressogram()
            rgram.fit(x=x, y=y)
            pred = rgram.predict(x)
            assert len(pred) == len(x)

    def test_multiple_sequential_fits(self):
        """Test multiple sequential fits on same instance."""
        rgram = Regressogram()

        for i in range(5):
            x = np.linspace(0, 10, 50 + i * 10)
            y = np.sin(x)

            rgram.fit(x=x, y=y)
            result = rgram.transform().collect()
            assert len(result) > 0


class TestDatasetVariations:
    """Test with various dataset characteristics."""

    def test_monotonically_increasing(self):
        """Test with monotonically increasing data."""
        x = np.linspace(0, 100, 100)
        y = x + np.random.randn(100) * 1

        rgram = Regressogram()
        pred = rgram.fit_predict(x=x, y=y)
        assert len(pred) > 0

    def test_monotonically_decreasing(self):
        """Test with monotonically decreasing data."""
        x = np.linspace(0, 100, 100)
        y = 100 - x + np.random.randn(100) * 1

        rgram = Regressogram()
        pred = rgram.fit_predict(x=x, y=y)
        assert len(pred) > 0

    def test_periodic_data(self):
        """Test with periodic data."""
        x = np.linspace(0, 20, 200)
        y = np.sin(x) + np.cos(2 * x)

        rgram = Regressogram()
        pred = rgram.fit_predict(x=x, y=y)
        assert len(pred) > 0

    def test_noisy_data(self):
        """Test with very noisy data."""
        x = np.linspace(0, 10, 100)
        y = x + np.random.randn(100) * 100  # Very large noise

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()
        assert len(result) == 100

    def test_sparse_data(self):
        """Test with sparse data."""
        x = np.array([0, 0.1, 5, 5.2, 9.8, 10])
        y = np.array([0, 0.1, 25, 27, 96, 100], dtype=float)

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()
        assert len(result) == 6


class TestScalingBehavior:
    """Test scaling behavior as dimensions change."""

    def test_scaling_with_bins(self):
        """Test that increasing n_bins affects computation reasonably."""
        x = np.linspace(0, 100, 1000)
        y = x**2

        for n_bins in [5, 10, 20, 50]:
            rgram = Regressogram(binning="dist", n_bins=n_bins)
            result = rgram.fit(x=x, y=y).transform().collect()
            assert len(result) == 1000

    def test_scaling_with_eval_samples(self):
        """Test KernelSmoother scaling with eval samples."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 200), "y": np.sin(np.linspace(0, 10, 200))}
        )

        for n_eval in [10, 50, 100, 500]:
            smoother = KernelSmoother(n_eval_samples=n_eval)
            result = smoother.fit(data=df, x="x", y="y").transform().collect()
            assert len(result) == n_eval
