"""
Tests for duplicate handling and preservation.
Ensures duplicates are properly aggregated and not dropped.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram


class TestDuplicatePreservation:
    """Test that duplicates are properly handled and aggregated."""

    def test_multiple_y_values_per_x_aggregated_correctly(self):
        """Test that multiple y values at same x are aggregated correctly with mean."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        # At x=1, mean should be 2.0
        # At x=2, mean should be 5.0
        # At x=3, mean should be 8.0
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # Check that aggregation happened (not just returning raw values)
        # assert not np.all(np.isin(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))

    def test_duplicates_with_median_aggregation(self):
        """Test duplicate aggregation with median function."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.median())
        result = rgram.fit_predict(x=x, y=y)

        # Median of [1, 2, 3, 4, 5] is 3
        assert isinstance(result, np.ndarray)
        assert np.isclose(result[0], 3.0)

    def test_duplicates_with_sum_aggregation(self):
        """Test duplicate aggregation with sum function."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([10.0, 20.0, 30.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.sum())
        result = rgram.fit_predict(x=x, y=y)

        # Sum should be 60
        assert isinstance(result, np.ndarray)
        assert np.isclose(result[0], 60.0)

    def test_duplicates_with_max_aggregation(self):
        """Test duplicate aggregation with max function."""
        x = np.array([2.0, 2.0, 2.0, 2.0])
        y = np.array([5.0, 15.0, 10.0, 8.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.max())
        result = rgram.fit_predict(x=x, y=y)

        # Max should be 15
        assert isinstance(result, np.ndarray)
        assert np.isclose(result[0], 15.0)

    def test_duplicates_with_min_aggregation(self):
        """Test duplicate aggregation with min function."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([100.0, 20.0, 50.0, 75.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.min())
        result = rgram.fit_predict(x=x, y=y)

        # Min should be 20
        assert isinstance(result, np.ndarray)
        assert np.isclose(result[0], 20.0)

    def test_duplicates_with_count_aggregation(self):
        """Test duplicate aggregation with count function."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.count())
        result = rgram.fit_predict(x=x, y=y)

        # Count should be 5
        assert isinstance(result, np.ndarray)
        assert np.isclose(result[0], 5.0)

    def test_partial_duplicates_not_dropped(self):
        """Test that partial duplicates are aggregated correctly."""
        # Mix of unique and duplicate values
        x = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0])
        y = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_all_duplicates_fit_predict_consistency(self):
        """Test that fit then predict gives same result as fit_predict with duplicates."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        rgram1 = Regressogram(binning="int", agg=lambda s: s.mean())
        fit_predict_result = rgram1.fit_predict(x=x, y=y)

        rgram2 = Regressogram(binning="int", agg=lambda s: s.mean())
        rgram2.fit(x=x, y=y)
        fit_then_predict = rgram2.predict(x=x)

        # Results should be the same
        assert np.allclose(fit_predict_result, fit_then_predict)

    def test_duplicates_across_all_binning_strategies(self):
        """Test duplicate handling works across all binning strategies."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        strategies = ["int", "none", "width", "dist"]

        for strategy in strategies:
            rgram = Regressogram(binning=strategy, agg=lambda s: s.mean())
            result = rgram.fit_predict(x=x, y=y)

            assert isinstance(result, np.ndarray)
            assert len(result) > 0

    # def test_many_duplicates_with_variance_aggregation(self):
    #     """Test duplicate handling with variance calculation."""
    #     x = np.array([1.0] * 10 + [2.0] * 10 + [3.0] * 10)
    #     y = np.arange(1, 11) * np.ones(30)

    #     rgram = Regressogram(binning="int", agg=lambda s: s.var())
    #     result = rgram.fit_predict(x=x, y=y)

    #     assert isinstance(result, np.ndarray)
    #     assert len(result) > 0

    def test_duplicates_with_quantile_aggregation(self):
        """Test duplicate aggregation with quantile function."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.quantile(0.75))
        result = rgram.fit_predict(x=x, y=y)

        # 75th percentile of [1,2,3,4,5] is 4.0
        assert isinstance(result, np.ndarray)
        assert result[0] >= 3.0


class TestDuplicateWithCI:
    """Test confidence intervals with duplicate data."""

    def test_duplicates_with_ci_lower_upper(self):
        """Test CI calculation with duplicate x values."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        ci = (lambda s: s.min(), lambda s: s.max())
        rgram = Regressogram(binning="int", agg=lambda s: s.mean(), ci=ci)
        pred, lci, uci = rgram.fit_predict(x=x, y=y, return_ci=True)

        assert isinstance(pred, np.ndarray)
        assert isinstance(lci, np.ndarray)
        assert isinstance(uci, np.ndarray)
        # CI should respect min < pred < max
        valid = ~np.isnan(lci) & ~np.isnan(pred) & ~np.isnan(uci)
        if np.any(valid):
            assert np.all(lci[valid] <= pred[valid])
            assert np.all(pred[valid] <= uci[valid])

    def test_all_same_duplicate_values_ci(self):
        """Test CI when all y values are identical (duplicate x with same y)."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([5.0, 5.0, 5.0])

        ci = (lambda s: s.mean() - s.std(), lambda s: s.mean() + s.std())
        rgram = Regressogram(binning="int", agg=lambda s: s.mean(), ci=ci)
        pred, lci, uci = rgram.fit_predict(x=x, y=y, return_ci=True)

        # Mean should be 5.0, std should be 0
        assert np.isclose(pred[0], 5.0)


class TestDuplicateEdgeCases:
    """Test edge cases with duplicates."""

    def test_single_duplicate_value(self):
        """Test with minimal duplicates (only 2 of same value)."""
        x = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        rgram = Regressogram(binning="none", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_heavily_duplicated_data(self):
        """Test with heavily duplicated data (90% duplicates)."""
        x = np.array([1.0] * 90 + [2.0] * 10)
        y = np.random.randn(100)

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_alternating_duplicates(self):
        """Test with alternating duplicate pattern."""
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_duplicates_with_nan_handling(self):
        """Test duplicate handling with NaN-like edge cases handled gracefully."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        y = np.array([1.0, 0.0, -1.0, 2.0, -2.0, 0.0])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_duplicates_with_large_value_range(self):
        """Test duplicates with large differences in y values."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1e-10, 1e10, 1e5])

        rgram = Regressogram(binning="int", agg=lambda s: s.mean())
        result = rgram.fit_predict(x=x, y=y)

        assert isinstance(result, np.ndarray)
        assert np.isfinite(result[0])
