"""
Comprehensive workflow tests.
Ensures all major workflows are tested and interact correctly.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother


class TestRegressogramComprehensiveWorkflows:
    """Test comprehensive Regressogram workflows."""

    def test_full_workflow_fit_predict_then_predict_again(self):
        """Test complete workflow: fit, predict, predict again."""
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram.fit(x=x_train, y=y_train)

        # First prediction
        x_test1 = np.array([1.5, 2.5, 3.5])
        pred1 = rgram.predict(x=x_test1)

        # Second prediction
        x_test2 = np.array([2.0, 4.0])
        pred2 = rgram.predict(x=x_test2)

        assert len(pred1) == 3
        assert len(pred2) == 2

    def test_workflow_with_all_binning_strategies(self):
        """Test workflow with all binning strategies."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        strategies = ["int", "none", "width", "dist"]

        for strategy in strategies:
            rgram = Regressogram(binning=strategy, agg=lambda s: s.mean())
            pred_fit = rgram.fit_predict(x=x, y=y)

            # Test predictions on same data
            pred = rgram.predict(x=x)

            assert isinstance(pred_fit, np.ndarray)
            assert isinstance(pred, np.ndarray)
            assert len(pred) == len(x)

    def test_workflow_with_all_agg_functions(self):
        """Test workflow with different aggregation functions."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        agg_functions = [
            ("mean", lambda s: s.mean()),
            ("median", lambda s: s.median()),
            ("min", lambda s: s.min()),
            ("max", lambda s: s.max()),
            ("sum", lambda s: s.sum()),
            ("count", lambda s: s.count()),
            ("std", lambda s: s.std()),
        ]

        for name, agg_func in agg_functions:
            rgram = Regressogram(binning="int", agg=agg_func)
            result = rgram.fit_predict(x=x, y=y)

            assert isinstance(result, np.ndarray)
            assert len(result) > 0

    def test_workflow_dataframe_input_multiple_xy(self):
        """Test workflow with DataFrame input and multiple x/y columns."""
        df = pl.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
                "y1": [10.0, 20.0, 30.0, 40.0, 50.0],
                "y2": [5.0, 10.0, 15.0, 20.0, 25.0],
            }
        )

        # Single x, single y
        rgram1 = Regressogram()
        result1 = rgram1.fit_predict(data=df, x="x1", y="y1")
        assert isinstance(result1, np.ndarray)

        # Single x, multiple y
        rgram3 = Regressogram()
        result3 = rgram3.fit_predict(data=df, x="x1", y=["y1", "y2"])
        assert isinstance(result3, np.ndarray)

    def test_workflow_with_ci_various_definitions(self):
        """Test workflow with different CI definitions."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        ci_definitions = [
            (lambda s: s.mean() - s.std(), lambda s: s.mean() + s.std(), "mean ± std"),
            (lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "IQR"),
            (lambda s: s.min(), lambda s: s.max(), "min-max"),
        ]

        for ci_lower, ci_upper, name in ci_definitions:
            rgram = Regressogram(
                binning="int", agg=lambda s: s.mean(), ci=(ci_lower, ci_upper)
            )
            pred, lci, uci = rgram.fit_predict(x=x, y=y, return_ci=True)

            assert isinstance(pred, np.ndarray)
            assert isinstance(lci, np.ndarray)
            assert isinstance(uci, np.ndarray)

    def test_workflow_array_vs_dataframe_consistency(self):
        """Test that array and DataFrame inputs produce consistent results."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        df = pl.DataFrame({"x": x, "y": y})

        rgram_array = Regressogram(binning="int", agg=lambda s: s.mean())
        result_array = rgram_array.fit_predict(x=x, y=y)

        rgram_df = Regressogram(binning="int", agg=lambda s: s.mean())
        result_df = rgram_df.fit_predict(data=df, x="x", y="y")

        # Should produce same length results
        assert len(result_array) == len(result_df)

    def test_workflow_fit_persist_across_calls(self):
        """Test that fit parameters persist across multiple predict calls."""
        x_train = np.array([1.0, 2.0, 3.0, 4.0])
        y_train = np.array([10.0, 20.0, 30.0, 40.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram.fit(x=x_train, y=y_train)

        # Multiple predictions
        pred1 = rgram.predict(np.array([1.5]))
        pred2 = rgram.predict(np.array([2.5]))
        pred3 = rgram.predict(np.array([3.5]))

        assert isinstance(pred1, np.ndarray)
        assert isinstance(pred2, np.ndarray)
        assert isinstance(pred3, np.ndarray)

    def test_workflow_large_dataset(self):
        """Test workflow with large dataset."""
        np.random.seed(42)
        x = np.random.uniform(0, 10, 5000)
        y = np.sin(x) + np.random.normal(0, 0.1, 5000)

        rgram = Regressogram(binning="dist", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_workflow_narrow_range_x(self):
        """Test workflow with very narrow x range."""
        x = np.array([1.0, 1.001, 1.002, 1.003, 1.004])
        y = np.array([10.0, 10.1, 10.2, 10.3, 10.4])

        rgram = Regressogram(binning="width", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)

    def test_workflow_wide_range_x(self):
        """Test workflow with very wide x range."""
        x = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rgram = Regressogram(binning="dist", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)


class TestComprehensiveIntegration:
    """Test comprehensive integration scenarios."""

    def test_alternating_fit_predict_workflow(self):
        """Test alternating fit/predict with different datasets."""
        x1 = np.array([1.0, 2.0, 3.0])
        y1 = np.array([10.0, 20.0, 30.0])

        x2 = np.array([4.0, 5.0, 6.0])
        y2 = np.array([40.0, 50.0, 60.0])

        rgram1 = Regressogram(binning="int")
        rgram1.fit(x=x1, y=y1)
        pred1 = rgram1.predict(x1)

        rgram2 = Regressogram(binning="int")
        rgram2.fit(x=x2, y=y2)
        pred2 = rgram2.predict(x2)

        # Both should work independently
        assert len(pred1) == 3
        assert len(pred2) == 3

    def test_sequential_refitting_scenario(self):
        """Test refitting model with different data."""
        rgram = Regressogram(binning="int", agg=lambda s: s.mean())

        # First fit
        x1 = np.array([1.0, 2.0, 3.0])
        y1 = np.array([10.0, 20.0, 30.0])
        rgram.fit(x=x1, y=y1)
        pred1 = rgram.predict(np.array([2.0]))

        # Refit with new data
        x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y2 = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        rgram.fit(x=x2, y=y2)
        pred2 = rgram.predict(np.array([2.0]))

        assert isinstance(pred1, np.ndarray)
        assert isinstance(pred2, np.ndarray)

    def test_complex_multi_step_workflow(self):
        """Test complex multi-step workflow with various operations."""
        # Create initial data
        x_init = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_init = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Create another dataset
        df = pl.DataFrame({"x": [1.5, 2.5, 3.5, 4.5], "y": [15.0, 25.0, 35.0, 45.0]})

        # Use first dataset
        rgram1 = Regressogram(binning="int")
        pred1 = rgram1.fit_predict(x=x_init, y=y_init)

        # Use DataFrame
        rgram2 = Regressogram(binning="width")
        pred2 = rgram2.fit_predict(data=df, x="x", y="y")

        # Combine predictions
        assert len(pred1) == 5
        assert len(pred2) == 4

    def test_workflow_with_edge_case_combinations(self):
        """Test workflow with combinations of edge cases."""
        # Duplicates + small range
        x = np.array([1.0, 1.0, 1.0, 1.001, 1.001, 1.001])
        y = np.array([10.0, 20.0, 30.0, 11.0, 21.0, 31.0])

        rgram = Regressogram(binning="width", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_workflow_prediction_at_training_points(self):
        """Test prediction at exact training points."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram.fit(x=x, y=y)

        # Predict at training points
        pred_at_train = rgram.predict(x=x)

        # Predictions should be defined at training points
        assert len(pred_at_train) == len(x)

    def test_workflow_prediction_outside_training_range(self):
        """Test prediction outside training data range."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram.fit(x=x, y=y)

        # Predict outside range
        x_test = np.array([0.5, 5.5, 10.0])
        pred = rgram.predict(x=x_test)

        assert len(pred) == 3

    def test_workflow_mixed_array_series_input(self):
        """Test workflow with mixed array and Series input."""
        x_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_series = pl.Series([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int")
        result = rgram.fit_predict(x=x_array, y=y_series)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestWorkflowRobustness:
    """Test workflow robustness under various conditions."""

    def test_workflow_stable_with_noisy_data(self):
        """Test workflow robustness with noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 1, 100)  # High noise

        rgram = Regressogram(binning="dist", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_workflow_with_extreme_outliers(self):
        """Test workflow with extreme outliers."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 1e10, 40.0, 50.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)

    def test_workflow_reproducibility(self):
        """Test that workflow produces reproducible results."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Run workflow twice
        rgram1 = Regressogram(binning="int", agg=lambda s: s.mean())
        result1 = rgram1.fit_predict(x=x, y=y)

        rgram2 = Regressogram(binning="int", agg=lambda s: s.mean())
        result2 = rgram2.fit_predict(x=x, y=y)

        # Results should be identical
        assert np.allclose(result1, result2)

    def test_workflow_with_repeated_identical_rows(self):
        """Test workflow when data contains repeated identical rows."""
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        y = np.array([10.0, 10.0, 20.0, 20.0, 30.0, 30.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
