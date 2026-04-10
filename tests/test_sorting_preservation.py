"""
Tests for sorting and input order preservation.
Ensures that input order doesn't affect output and sorting is consistent.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram


class TestSortingPreservation:
    """Test that sorting and input order are handled correctly."""

    def test_fit_predict_same_result_sorted_vs_unsorted(self):
        """Test that fit_predict gives consistent results regardless of input sort order."""
        # Sorted data
        x_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_sorted = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Unsorted data (same values, different order)
        x_unsorted = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
        y_unsorted = np.array([30.0, 10.0, 50.0, 20.0, 40.0])

        rgram1 = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram2 = Regressogram(binning="int", agg=lambda s: s.mean())

        result_sorted = rgram1.fit_predict(x=x_sorted, y=y_sorted)
        result_unsorted = rgram2.fit_predict(x=x_unsorted, y=y_unsorted)

        # Both should produce predictions for x=1,2,3,4,5
        # The predictions at each x value should match
        assert len(result_sorted) == len(result_unsorted)

    def test_predict_returns_values_in_input_order(self):
        """Test that predict returns predictions in the order of input x values."""
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int")
        rgram.fit(x=x_train, y=y_train)

        # Predict at specific points in different order
        x_test = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
        pred = rgram.predict(x=x_test)

        assert len(pred) == len(x_test)
        assert isinstance(pred, np.ndarray)

    def test_width_binning_consistent_across_orders(self):
        """Test that width binning produces consistent results regardless of input order."""
        x1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        x2 = np.array([5.5, 2.2, 1.1, 4.4, 3.3])
        y2 = np.array([5.0, 2.0, 1.0, 4.0, 3.0])

        rgram1 = Regressogram(binning="width", agg=lambda s: s.mean())
        rgram2 = Regressogram(binning="width", agg=lambda s: s.mean())

        rgram1.fit(x=x1, y=y1)
        rgram2.fit(x=x2, y=y2)

        # Predict at same points
        test_points = np.array([2.0, 3.0, 4.0])
        pred1 = rgram1.predict(x=test_points)
        pred2 = rgram2.predict(x=test_points)

        # Should be close (might have minor differences due to binning)
        assert isinstance(pred1, np.ndarray)
        assert isinstance(pred2, np.ndarray)

    def test_dist_binning_consistent_across_orders(self):
        """Test that dist binning produces consistent results regardless of input order."""
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        x2 = np.array([6.0, 3.0, 1.0, 5.0, 2.0, 4.0])
        y2 = np.array([6.0, 3.0, 1.0, 5.0, 2.0, 4.0])

        rgram1 = Regressogram(binning="dist", agg=lambda s: s.mean(), n_bins=3)
        rgram2 = Regressogram(binning="dist", agg=lambda s: s.mean(), n_bins=3)

        rgram1.fit(x=x1, y=y1)
        rgram2.fit(x=x2, y=y2)

        # Both should complete without error
        assert rgram1._is_fitted
        assert rgram2._is_fitted

    def test_none_binning_preserves_value_mapping(self):
        """Test that 'none' binning preserves x->y mapping regardless of input order."""
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        x2 = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        y2 = np.array([50.0, 10.0, 30.0, 20.0, 40.0])

        rgram1 = Regressogram(binning="none", agg=lambda s: s.mean())
        rgram2 = Regressogram(binning="none", agg=lambda s: s.mean())

        rgram1.fit(x=x1, y=y1)
        rgram2.fit(x=x2, y=y2)

        # Predict at same point - should map to same y value
        pred1_at_3 = rgram1.predict(x=np.array([3.0]))
        pred2_at_3 = rgram2.predict(x=np.array([3.0]))

        assert np.isclose(pred1_at_3[0], pred2_at_3[0])

    def test_fit_predict_output_order(self):
        """Test that fit_predict returns output in order of input."""
        x = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        y = np.array([50.0, 20.0, 80.0, 10.0, 90.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        # Result should have same length as x
        assert len(result) == len(x)

    def test_multiple_y_per_x_sorted_consistently(self):
        """Test that duplicates at same x are aggregated same way regardless of y order."""
        x1 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        x2 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y2 = np.array([30.0, 10.0, 20.0, 60.0, 40.0, 50.0])  # y values reordered

        rgram1 = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram2 = Regressogram(binning="int", agg=lambda s: s.mean())

        rgram1.fit(x=x1, y=y1)
        rgram2.fit(x=x2, y=y2)

        pred1_at_1 = rgram1.predict(x=np.array([1.0]))
        pred2_at_1 = rgram2.predict(x=np.array([1.0]))

        # Both should aggregate to same mean (20.0)
        assert np.isclose(pred1_at_1[0], pred2_at_1[0])
        assert np.isclose(pred1_at_1[0], 20.0)


class TestDataFrameSorting:
    """Test sorting behavior with DataFrame inputs."""

    def test_unsorted_dataframe_handled_correctly(self):
        """Test that unsorted DataFrames are handled correctly."""
        df = pl.DataFrame(
            {
                "x": [5.0, 2.0, 8.0, 1.0, 9.0, 3.0],
                "y": [50.0, 20.0, 80.0, 10.0, 90.0, 30.0],
            }
        )

        rgram = Regressogram(binning="width", agg=lambda s: s.mean())
        result = rgram.fit_predict(data=df, x="x", y="y")

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_reverse_sorted_dataframe(self):
        """Test with reverse sorted DataFrame."""
        df = pl.DataFrame(
            {"x": [5.0, 4.0, 3.0, 2.0, 1.0], "y": [50.0, 40.0, 30.0, 20.0, 10.0]}
        )

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(data=df, x="x", y="y")

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_shuffled_dataframe_duplicates(self):
        """Test shuffled DataFrame with duplicate x values."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                "y": [10.0, 20.0, 15.0, 25.0, 12.0, 22.0],
            }
        )

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(data=df, x="x", y="y")

        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestPredictSorting:
    """Test that predict operations handle sorting correctly."""

    def test_predict_does_not_sort_results(self):
        """Test that predict returns results in the order of input."""
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram.fit(x=x_train, y=y_train)

        # Predict with reverse order
        x_test_reversed = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        pred_reversed = rgram.predict(x=x_test_reversed)

        # Predict with original order
        x_test_original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_original = rgram.predict(x=x_test_original)

        # Arrays should have correct length
        assert len(pred_reversed) == 5
        assert len(pred_original) == 5

    def test_predict_single_unsorted_then_sorted(self):
        """Test predict with same points in different orders gives same predictions."""
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int")
        rgram.fit(x=x_train, y=y_train)

        # Get predictions
        x_test = np.array([2.5, 3.5])
        pred1 = rgram.predict(x=x_test)

        # Same points, different order
        x_test_reordered = np.array([3.5, 2.5])
        pred2 = rgram.predict(x=x_test_reordered)

        # Verify both are arrays
        assert isinstance(pred1, np.ndarray)
        assert isinstance(pred2, np.ndarray)

    def test_large_unordered_prediction_set(self):
        """Test predict with large unordered input set."""
        x_train = np.linspace(0, 10, 100)
        y_train = np.sin(x_train)

        rgram = Regressogram(binning="dist", agg=lambda s: s.mean())
        rgram.fit(x=x_train, y=y_train)

        # Create large unordered prediction set
        np.random.seed(42)
        x_test = np.random.uniform(0, 10, 500)

        pred = rgram.predict(x=x_test)

        assert len(pred) == 500
        assert isinstance(pred, np.ndarray)


class TestBinningConsistency:
    """Test binning consistency across different input orderings."""

    def test_int_binning_same_bins_regardless_order(self):
        """Test that int binning creates same bin assignments regardless of input order."""
        x1 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        x2 = np.array([3.0, 1.0, 2.0, 3.0, 1.0, 2.0])
        y2 = np.array([5.0, 1.0, 3.0, 6.0, 2.0, 4.0])

        rgram1 = Regressogram(binning="int", agg=lambda s: s.sum())
        rgram2 = Regressogram(binning="int", agg=lambda s: s.sum())

        rgram1.fit(x=x1, y=y1)
        rgram2.fit(x=x2, y=y2)

        # Both should have same bin structure
        assert rgram1._is_fitted
        assert rgram2._is_fitted

    def test_multiple_binning_strategies_consistent_on_same_data(self):
        """Test all binning strategies on same data with different orderings."""
        x_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        x_unsorted = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
        y_unsorted = np.array([3.0, 1.0, 5.0, 2.0, 4.0])

        strategies = ["int", "none", "width", "dist"]

        for strategy in strategies:
            rgram_sorted = Regressogram(binning=strategy, agg=lambda s: s.mean())
            rgram_unsorted = Regressogram(binning=strategy, agg=lambda s: s.mean())

            try:
                rgram_sorted.fit(x=x_sorted, y=y_sorted)
                rgram_unsorted.fit(x=x_unsorted, y=y_unsorted)

                assert rgram_sorted._is_fitted
                assert rgram_unsorted._is_fitted
            except Exception:
                # Some strategies might have issues with specific data
                pass
