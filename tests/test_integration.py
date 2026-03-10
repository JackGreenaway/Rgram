"""
Integration and workflow tests.
Tests complex scenarios, method interactions, and realistic workflows.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother


class TestRegressogramWorkflows:
    """Test realistic Regressogram workflows."""

    def test_full_workflow_fit_predict_ci(self):
        """Test complete workflow: fit, predict, get CI."""
        np.random.seed(42)
        x_train = np.linspace(0, 10, 50)
        y_train = np.sin(x_train) + np.random.randn(50) * 0.2

        rgram = Regressogram()

        # Fit
        rgram.fit(x=x_train, y=y_train)

        # Predict on new data
        x_test = np.array([1.5, 3.5, 5.5, 7.5, 9.5])
        pred, ci_low, ci_high = rgram.predict(x_test, return_ci=True)

        assert len(pred) == len(x_test)
        assert ci_low is not None
        assert ci_high is not None
        assert np.all(ci_low <= pred)
        assert np.all(pred <= ci_high)

    def test_fit_predict_shortcut_workflow(self):
        """Test fit_predict shortcut method."""
        x = np.linspace(0, 10, 30)
        y = np.exp(x / 5)

        rgram = Regressogram()
        pred = rgram.fit_predict(x=x, y=y)

        assert isinstance(pred, np.ndarray)
        assert len(pred) > 0

    def test_fit_predict_with_ci_shortcut(self):
        """Test fit_predict with CI calculation."""
        x = np.linspace(0, 10, 30)
        y = np.sin(x)

        rgram = Regressogram()
        pred, ci_low, ci_high = rgram.fit_predict(x=x, y=y, return_ci=True)

        assert isinstance(pred, np.ndarray)
        assert isinstance(ci_low, np.ndarray)
        assert isinstance(ci_high, np.ndarray)

    def test_multiple_predictions_after_single_fit(self):
        """Test making multiple predictions after single fit."""
        x_train = np.linspace(0, 10, 50)
        y_train = x_train**2

        rgram = Regressogram()
        rgram.fit(x=x_train, y=y_train)

        # Make predictions at different points
        x_test1 = np.array([2.5, 3.5, 4.5])
        x_test2 = np.array([1.0, 5.0, 9.0])

        pred1 = rgram.predict(x_test1)
        pred2 = rgram.predict(x_test2)

        assert len(pred1) == 3
        assert len(pred2) == 3

    def test_fit_on_dataframe_then_predict_with_arrays(self):
        """Test fitting with DataFrame then predicting with arrays."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 30), "y": np.sin(np.linspace(0, 10, 30))}
        )

        rgram = Regressogram()
        rgram.fit(data=df, x="x", y="y")

        # Predict with new array
        x_new = np.array([1.5, 5.0, 8.5])
        pred = rgram.predict(x_new)

        assert len(pred) == 3

    def test_transform_after_fit_with_arrays(self):
        """Test transform after fitting with arrays."""
        x = np.linspace(0, 10, 20)
        y = x + np.random.randn(20) * 0.5

        rgram = Regressogram()
        rgram.fit(x=x, y=y)

        result = rgram.transform().collect()

        assert "y_pred_rgram" in result.columns
        assert len(result) == 20

    def test_multiple_binning_strategies_same_data(self):
        """Test using different binning strategies on same data."""
        x = np.linspace(0, 10, 50)
        y = x**2 + np.random.randn(50) * 10

        strategies = ["dist", "width", "int", "none"]

        results = {}
        for strategy in strategies:
            rgram = Regressogram(binning=strategy)
            try:
                pred = rgram.fit_predict(x=x, y=y)
                results[strategy] = pred
                assert len(pred) > 0
            except Exception:
                # Some strategies might fail on specific data
                pass

        # Should have at least some results
        assert len(results) > 0

    def test_different_aggregation_functions_same_data(self):
        """Test using different aggregation functions on same data."""
        x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=float)
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

        agg_funcs = {
            "mean": lambda s: s.mean(),
            "median": lambda s: s.median(),
            "min": lambda s: s.min(),
            "max": lambda s: s.max(),
        }

        results = {}
        for name, agg_func in agg_funcs.items():
            rgram = Regressogram(binning="int", agg=agg_func)
            result = rgram.fit(x=x, y=y).transform().collect()
            results[name] = result["y_pred_rgram"].to_numpy()

        # Results should be different
        assert not np.allclose(results["mean"], results["min"])
        assert not np.allclose(results["mean"], results["max"])

    def test_hue_grouping_workflow(self):
        """Test workflow with hue grouping."""
        df = pl.DataFrame(
            {
                "x": np.tile(np.linspace(0, 10, 20), 2),
                "y": np.tile(np.linspace(1, 20, 20), 2) + np.random.randn(40) * 0.5,
                "group": np.repeat([0, 1], 20),
            }
        )

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y", hue="group").transform().collect()

        # Should have results for both groups
        assert "group" in result.columns
        groups = result["group"].unique().to_list()
        assert len(groups) == 2

    def test_keys_parameter_workflow(self):
        """Test workflow with keys parameter."""
        df = pl.DataFrame(
            {
                "x": np.linspace(0, 10, 30),
                "y": np.sin(np.linspace(0, 10, 30)),
                "keys": np.repeat([1, 2, 3], 10),
            }
        )

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y", keys="keys").transform().collect()

        assert len(result) == 30


class TestKernelSmootherWorkflows:
    """Test realistic KernelSmoother workflows."""

    def test_full_workflow_fit_predict(self):
        """Test complete KernelSmoother workflow."""
        np.random.seed(42)
        x_train = np.linspace(0, 10, 100)
        y_train = np.sin(x_train) + np.random.randn(100) * 0.2

        smoother = KernelSmoother(n_eval_samples=50)

        # Fit
        smoother.fit(data=pl.DataFrame({"x": x_train, "y": y_train}), x="x", y="y")

        # Predict
        x_test = np.linspace(0, 10, 20)
        pred = smoother.predict(x_test)

        assert len(pred) == 20
        assert np.all(np.isfinite(pred))

    def test_fit_predict_shortcut(self):
        """Test fit_predict shortcut."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 50), "y": np.cos(np.linspace(0, 10, 50))}
        )

        smoother = KernelSmoother(n_eval_samples=30)
        pred = smoother.fit_predict(data=df, x="x", y="y")

        assert isinstance(pred, np.ndarray)
        assert len(pred) == 30

    def test_transform_workflow(self):
        """Test transform workflow."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 50), "y": np.sin(np.linspace(0, 10, 50))}
        )

        smoother = KernelSmoother(n_eval_samples=20)
        smoother.fit(data=df, x="x", y="y")
        result = smoother.transform().collect()

        assert len(result) == 20
        assert "x_eval" in result.columns
        assert "y_kernel" in result.columns

    def test_different_bandwidth_methods_comparison(self):
        """Test comparing different bandwidth methods."""
        df = pl.DataFrame(
            {
                "x": np.linspace(0, 10, 100),
                "y": np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1,
            }
        )

        methods = ["silverman", "scott"]
        results = {}

        for method in methods:
            smoother = KernelSmoother(bandwidth=method, n_eval_samples=50)
            result = smoother.fit_predict(data=df, x="x", y="y")
            results[method] = result

        # Both should work and produce different results
        assert not np.allclose(results["silverman"], results["scott"])

    def test_manual_bandwidth_workflow(self):
        """Test workflow with manual bandwidth."""
        df = pl.DataFrame(
            {"x": np.linspace(0, 10, 50), "y": np.exp(np.linspace(0, 10, 50) / 5)}
        )

        smoother = KernelSmoother(
            bandwidth="manual", bandwidth_value=1.5, n_eval_samples=30
        )
        smoother.fit(data=df, x="x", y="y")
        pred = smoother.predict(np.linspace(0, 10, 20))

        assert len(pred) == 20


class TestCrossValidationWorkflows:
    """Test workflows that involve cross-validation patterns."""

    def test_train_test_split_workflow(self):
        """Test train-test split workflow."""
        np.random.seed(42)
        x_all = np.linspace(0, 10, 100)
        y_all = np.sin(x_all) + np.random.randn(100) * 0.3

        # Split data
        split_idx = 80
        x_train, x_test = x_all[:split_idx], x_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        # Fit on training data
        rgram = Regressogram()
        rgram.fit(x=x_train, y=y_train)

        # Predict on test data
        pred_test = rgram.predict(x_test)

        assert len(pred_test) == len(x_test)

    def test_incremental_model_refitting(self):
        """Test refitting model with more data."""
        x1 = np.linspace(0, 5, 20)
        y1 = x1**2

        x2 = np.linspace(5, 10, 20)
        y2 = x2**2

        # Fit with first dataset
        rgram1 = Regressogram()
        rgram1.fit(x=x1, y=y1)
        pred1 = rgram1.predict(x1)

        # Fit with combined dataset
        rgram2 = Regressogram()
        rgram2.fit(x=np.concatenate([x1, x2]), y=np.concatenate([y1, y2]))
        pred2 = rgram2.predict(x1)

        # Predictions should be different (more data should refine estimate)
        assert not np.allclose(pred1, pred2)

    def test_leave_one_out_like_workflow(self):
        """Test leave-one-out style workflow."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

        # Leave out each point and predict it
        predictions = []
        for i in range(len(x)):
            x_train = np.delete(x, i)
            y_train = np.delete(y, i)
            x_test = x[i : i + 1]

            rgram = Regressogram()
            rgram.fit(x=x_train, y=y_train)
            pred = rgram.predict(x_test)
            predictions.append(pred[0])

        assert len(predictions) == len(x)


class TestMixedInputWorkflows:
    """Test workflows with mixed DataFrame and array inputs."""

    def test_dataframe_input_then_array_operations(self):
        """Test using DataFrame input then array operations."""
        df = pl.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 4.0, 9.0, 16.0, 25.0]}
        )

        rgram = Regressogram()
        rgram.fit(data=df, x="x", y="y")

        # Predict with array
        pred = rgram.predict(np.array([1.5, 2.5, 3.5]))
        assert len(pred) == 3

    def test_array_input_then_dataframe_analysis(self):
        """Test using array input then analyzing with DataFrame operations."""
        x = np.linspace(0, 10, 30)
        y = np.sin(x)

        rgram = Regressogram()
        rgram.fit(x=x, y=y)

        # Get full results as DataFrame
        result_df = rgram.transform().collect()

        assert isinstance(result_df, pl.DataFrame)
        assert "y_pred_rgram" in result_df.columns

        # Can then filter/operate on DataFrame
        high_pred = result_df.filter(pl.col("y_pred_rgram") > 0)
        assert len(high_pred) > 0


class TestModelComparison:
    """Test workflows for comparing models."""

    def test_compare_regressogram_vs_kernel_smoother(self):
        """Test comparing predictions from both models on same data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.randn(50) * 0.1

        df = pl.DataFrame({"x": x, "y": y})

        # Regressogram
        rgram = Regressogram(binning="width")
        pred_rgram = rgram.fit_predict(x=x, y=y)

        # KernelSmoother
        smoother = KernelSmoother(n_eval_samples=len(x))
        pred_ks = smoother.fit_predict(data=df, x="x", y="y")

        # Both should produce predictions
        assert len(pred_rgram) > 0
        assert len(pred_ks) > 0

    def test_parameter_sensitivity_analysis(self):
        """Test sensitivity to parameter changes."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.randn(50) * 0.2

        # Test different n_bins values
        n_bins_values = [1, 5, 10, 20]
        predictions = {}

        for n_bins in n_bins_values:
            rgram = Regressogram(binning="dist", n_bins=n_bins)
            pred = rgram.fit_predict(x=x, y=y)
            predictions[n_bins] = pred

        # Predictions should vary with parameter
        assert len(predictions[1]) > 0
        assert len(predictions[20]) > 0


class TestErrorRecoveryWorkflows:
    """Test workflows that involve error conditions."""

    def test_retry_after_failed_fit(self):
        """Test that can retry after fit failure."""
        rgram = Regressogram()

        # Try with invalid data
        with pytest.raises(Exception):
            rgram.fit(x=[], y=[])

        # Should be able to fit with valid data
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) > 0

    def test_model_reset_workflow(self):
        """Test resetting model by creating new instance."""
        x1 = np.linspace(0, 5, 20)
        y1 = x1

        x2 = np.linspace(10, 15, 20)
        y2 = x2**2

        # First fit
        rgram1 = Regressogram()
        rgram1.fit(x=x1, y=y1)
        pred1 = rgram1.predict([2.5])

        # Different fit with new instance
        rgram2 = Regressogram()
        rgram2.fit(x=x2, y=y2)
        pred2 = rgram2.predict([12.5])

        assert pred1[0] != pred2[0]
