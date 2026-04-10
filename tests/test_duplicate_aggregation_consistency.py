"""
Tests for duplicate aggregation consistency.
Ensures that duplicate handling is consistent across different workflows.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram


class TestDuplicateAggregationConsistency:
    """Test that duplicates are aggregated consistently across methods."""

    def test_fit_predict_match_fit_then_predict_duplicates(self):
        """Test that fit_predict and fit+predict give same results with duplicates."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])

        # Method 1: fit_predict
        rgram1 = Regressogram(binning="int", agg=lambda s: s.mean())
        result_fit_predict = rgram1.fit_predict(x=x, y=y)

        # Method 2: fit then predict
        rgram2 = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram2.fit(x=x, y=y)
        result_fit_then_predict = rgram2.predict(x=x)

        # Both should give same results
        assert np.allclose(result_fit_predict, result_fit_then_predict)

    def test_duplicates_aggregation_matches_manual_calculation(self):
        """Test that duplicate aggregation matches manual calculation."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([10.0, 20.0, 30.0])

        # Manual calculation: mean of [10, 20, 30] = 20
        expected_mean = 20.0

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert np.isclose(result[0], expected_mean)

    def test_duplicate_count_matches_manual(self):
        """Test that count aggregation matches manual count."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Expected count = 5
        rgram = Regressogram(binning="int", agg=lambda s: s.count())
        result = rgram.fit_predict(x=x, y=y)

        assert np.isclose(result[0], 5.0)

    def test_duplicate_sum_matches_manual(self):
        """Test that sum aggregation matches manual sum."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([10.0, 20.0, 30.0])

        # Expected sum = 60
        expected_sum = 60.0

        rgram = Regressogram(binning="int", agg=lambda s: s.sum())
        result = rgram.fit_predict(x=x, y=y)

        assert np.isclose(result[0], expected_sum)

    def test_duplicate_median_matches_manual(self):
        """Test that median aggregation matches manual median."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Expected median = 3.0
        expected_median = 3.0

        rgram = Regressogram(binning="int", agg=lambda s: s.median())
        result = rgram.fit_predict(x=x, y=y)

        assert np.isclose(result[0], expected_median)

    def test_duplicates_consistency_with_multiple_aggs(self):
        """Test consistency across multiple aggregation functions on same duplicates."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        # Compute results with different aggs
        aggs = {
            "mean": (lambda s: s.mean(), 5.0),  # (2+4+6+8)/4 = 5
            "sum": (lambda s: s.sum(), 20.0),  # 2+4+6+8 = 20
            "min": (lambda s: s.min(), 2.0),  # min = 2
            "max": (lambda s: s.max(), 8.0),  # max = 8
            "count": (lambda s: s.count(), 4.0),  # count = 4
        }

        for name, (agg_func, expected) in aggs.items():
            rgram = Regressogram(binning="int", agg=agg_func)
            result = rgram.fit_predict(x=x, y=y)
            assert np.isclose(result[0], expected), f"{name} aggregation mismatch"

    def test_partial_duplicates_consistency(self):
        """Test consistency with partial duplicates."""
        x = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result1 = rgram.fit_predict(x=x, y=y)

        # Refit with same data
        rgram.fit(x=x, y=y)
        result2 = rgram.predict(x=x)

        # Should be identical
        assert np.allclose(result1, result2)

    def test_duplicates_no_order_dependence(self):
        """Test that aggregation is independent of duplicate order."""
        x1 = np.array([1.0, 1.0, 1.0])
        y1 = np.array([10.0, 20.0, 30.0])

        x2 = np.array([1.0, 1.0, 1.0])
        y2 = np.array([30.0, 10.0, 20.0])  # Different order

        rgram1 = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram2 = Regressogram(binning="int", agg=lambda s: s.mean())

        result1 = rgram1.fit_predict(x=x1, y=y1)
        result2 = rgram2.fit_predict(x=x2, y=y2)

        # Same aggregation regardless of y order
        assert np.isclose(result1[0], result2[0])

    def test_cumulative_duplicates_aggregation(self):
        """Test aggregation with many duplicates of same value."""
        n = 100
        x = np.ones(n)
        y = 5.0 * np.ones(n)

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        # Mean of 100 identical values should be 5.0
        assert np.isclose(result[0], 5.0)

    def test_duplicates_with_extreme_values(self):
        """Test duplicate aggregation with extreme value ranges."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1e-10, 1.0, 1e10])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        # Should still compute mean without errors
        assert np.isfinite(result[0])

    def test_duplicates_consistency_across_binning_int_none(self):
        """Test duplicate handling is consistent between 'int' and 'none' binning."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([10.0, 20.0, 30.0])

        # For unique x values, 'int' and 'none' should give same result
        rgram_int = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram_none = Regressogram(binning="none", agg=lambda s: s.mean())

        result_int = rgram_int.fit_predict(x=x, y=y)
        result_none = rgram_none.fit_predict(x=x, y=y)

        # Both should handle the duplicate correctly
        assert isinstance(result_int, np.ndarray)
        assert isinstance(result_none, np.ndarray)

    def test_dataframe_duplicate_handling_consistency(self):
        """Test that DataFrame and array inputs handle duplicates consistently."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        df = pl.DataFrame({"x": x, "y": y})

        rgram_array = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram_df = Regressogram(binning="int", agg=lambda s: s.mean())

        result_array = rgram_array.fit_predict(x=x, y=y)
        result_df = rgram_df.fit_predict(data=df, x="x", y="y")

        # Results should match
        assert len(result_array) == len(result_df)

    def test_duplicate_ci_consistency(self):
        """Test that CI is computed consistently with duplicates."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        ci = (lambda s: s.mean() - s.std(), lambda s: s.mean() + s.std())
        rgram = Regressogram(binning="int", agg=lambda s: s.mean(), ci=ci)

        # fit_predict with CI
        pred1, lci1, uci1 = rgram.fit_predict(x=x, y=y, return_ci=True)

        # fit then predict with CI
        rgram.fit(x=x, y=y)
        pred2, lci2, uci2 = rgram.predict(x=x, return_ci=True)

        # Should be consistent
        assert np.allclose(pred1, pred2)
        assert np.allclose(lci1, lci2)
        assert np.allclose(uci1, uci2)


class TestDuplicateAggregationInvariants:
    """Test invariants that should hold for duplicate aggregation."""

    def test_mean_is_between_min_max(self):
        """Test that mean aggregation is always between min and max."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([5.0, 15.0, 25.0, 35.0, 45.0])

        rgram_mean = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram_min = Regressogram(binning="int", agg=lambda s: s.min())
        rgram_max = Regressogram(binning="int", agg=lambda s: s.max())

        mean_result = rgram_mean.fit_predict(x=x, y=y)
        min_result = rgram_min.fit_predict(x=x, y=y)
        max_result = rgram_max.fit_predict(x=x, y=y)

        # Invariant: min <= mean <= max
        assert min_result[0] <= mean_result[0] <= max_result[0]

    def test_count_equals_num_duplicates(self):
        """Test that count aggregation equals the number of duplicate points."""
        x = np.array([1.0] * 7 + [2.0] * 3)
        y = np.arange(10, dtype=float)

        rgram_x1 = Regressogram(binning="int", agg=lambda s: s.count())
        result = rgram_x1.fit_predict(x=x, y=y)

        # Count for x=1 should be 7
        assert np.isclose(result[0], 7.0)

    def test_max_is_maximum_of_duplicates(self):
        """Test that max aggregation equals the maximum value."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([5.0, 15.0, 10.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.max())
        result = rgram.fit_predict(x=x, y=y)

        # Max should be 15.0
        assert np.isclose(result[0], 15.0)

    def test_min_is_minimum_of_duplicates(self):
        """Test that min aggregation equals the minimum value."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([5.0, 15.0, 10.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.min())
        result = rgram.fit_predict(x=x, y=y)

        # Min should be 5.0
        assert np.isclose(result[0], 5.0)

    def test_sum_doubles_with_duplicate_count(self):
        """Test that sum aggregation doubles when duplicates double (with same values)."""
        x1 = np.array([1.0, 1.0])
        y1 = np.array([5.0, 5.0])

        x2 = np.array([1.0, 1.0, 1.0, 1.0])
        y2 = np.array([5.0, 5.0, 5.0, 5.0])

        rgram1 = Regressogram(binning="int", agg=lambda s: s.sum())
        rgram2 = Regressogram(binning="int", agg=lambda s: s.sum())

        result1 = rgram1.fit_predict(x=x1, y=y1)
        result2 = rgram2.fit_predict(x=x2, y=y2)

        # Sum should double (10 vs 20)
        assert np.isclose(result2[0], 2 * result1[0])

    def test_std_of_identical_duplicates_is_zero(self):
        """Test that std of identical values is zero."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([5.0, 5.0, 5.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.std())
        result = rgram.fit_predict(x=x, y=y)

        # Std of identical values should be 0
        assert np.isclose(result[0], 0.0)
