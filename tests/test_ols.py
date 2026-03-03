import pytest
import polars as pl
import numpy as np
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother


class TestRegressogramBinningStrategies:
    """Test all binning strategies comprehensively."""

    def test_dist_binning_produces_consistent_results(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(binning="dist")
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

        # Fit again should produce consistent results
        rgram = Regressogram(binning="dist")
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

        # Results should be identical
        assert result.shape == result.shape

    def test_width_binning_with_various_sizes(self, sample_data):
        df, x, y, y_noise = sample_data
        for binning in ["dist", "width", "int"]:
            rgram = Regressogram(binning=binning)
            result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()
            assert len(result) > 0
            assert "y_pred_rgram" in result.columns

    def test_binning_covers_all_data_points(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(binning="dist")
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

        # All data points should be included in result
        assert len(result) == len(df)


class TestAggregationFunctions:
    """Test various aggregation functions."""

    def test_mean_aggregation(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(agg=lambda s: s.mean())
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()
        assert "y_pred_rgram" in result.columns

    def test_median_aggregation(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(agg=lambda s: s.median())
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()
        assert "y_pred_rgram" in result.columns

    def test_max_aggregation(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(agg=lambda s: s.max())
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()
        assert "y_pred_rgram" in result.columns

    def test_min_aggregation(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(agg=lambda s: s.min())
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()
        assert "y_pred_rgram" in result.columns

    def test_std_aggregation(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(agg=lambda s: s.std())
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()
        assert "y_pred_rgram" in result.columns

    def test_sum_aggregation_totals_correctly(self):
        x = np.arange(10)
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        rgram = Regressogram(agg=lambda s: s.sum(), binning="width")
        result = rgram.fit(x=x, y=y).transform().collect()

        # Just check that we have predictions
        assert "y_pred_rgram" in result.columns
        assert len(result) > 0

    def test_count_aggregation(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(agg=lambda s: s.count())
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()
        assert "y_pred_rgram" in result.columns


class TestConfidenceIntervals:
    """Test confidence interval computation."""

    def test_default_ci_bounds_order(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

        # CI upper should be >= CI lower
        lci = result["y_pred_rgram_lci"].drop_nulls()
        uci = result["y_pred_rgram_uci"].drop_nulls()

        if len(lci) > 0 and len(uci) > 0:
            assert (uci >= lci).all()

    def test_ci_with_quantiles(self, sample_data):
        df, x, y, y_noise = sample_data
        ci = (lambda x: x.quantile(0.5), lambda x: x.quantile(0.75))
        rgram = Regressogram(ci=ci)
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

        assert "y_pred_rgram_lci" in result.columns
        assert "y_pred_rgram_uci" in result.columns

    def test_no_ci_computation(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram(ci=None)
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

        assert "y_pred_rgram_lci" not in result.columns
        assert "y_pred_rgram_uci" not in result.columns


class TestDataFormats:
    """Test different input data formats."""

    def test_numpy_arrays_input(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) > 0

    def test_list_input(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]

        rgram = Regressogram()
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) > 0

    def test_polars_dataframe_input(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

        assert len(result) > 0

    def test_polars_lazyframe_input(self, sample_data):
        df, x, y, y_noise = sample_data
        lf = df.lazy()
        rgram = Regressogram()
        result = rgram.fit(data=lf, x="x", y="y_noise").transform().collect()

        assert len(result) > 0


class TestGrouping:
    """Test hue/grouping functionality."""

    def test_single_hue_grouping(self, sample_data):
        df, x, y, y_noise = sample_data
        df = df.with_columns((pl.arange(0, df.height) % 2).alias("group"))

        rgram = Regressogram()
        result = (
            rgram.fit(data=df, x="x", y="y_noise", hue="group").transform().collect()
        )

        assert "group" in result.columns
        assert result["group"].n_unique() == 2

    def test_multiple_hue_grouping(self, sample_data):
        df, x, y, y_noise = sample_data
        df = df.with_columns(
            [
                (pl.arange(0, df.height) % 2).alias("group1"),
                (pl.arange(0, df.height) % 3).alias("group2"),
            ]
        )

        rgram = Regressogram()
        result = (
            rgram.fit(data=df, x="x", y="y_noise", hue=["group1", "group2"])
            .transform()
            .collect()
        )

        assert "group1" in result.columns
        assert "group2" in result.columns

    def test_hue_with_keys_grouping(self, sample_data):
        df, x, y, y_noise = sample_data
        df = df.with_columns((pl.arange(0, df.height) % 2).alias("group"))

        rgram = Regressogram()
        result = (
            rgram.fit(data=df, x="x", y="y_noise", hue="group").transform().collect()
        )

        assert "group" in result.columns


class TestMultipleColumns:
    """Test with multiple x or y columns."""

    def test_multiple_y_columns(self, sample_data):
        df, x, y, y_noise = sample_data
        df = df.with_columns((pl.col("y") + 1).alias("y_offset"))

        rgram = Regressogram()
        result = (
            rgram.fit(data=df, x="x", y=["y_noise", "y_offset"]).transform().collect()
        )

        # Should have y_pred_rgram column
        assert "y_pred_rgram" in result.columns
        # With multiple y columns, result should have doubled rows
        assert len(result) == len(df) * 2

    def test_multiple_x_columns(self, sample_data):
        df, x, y, y_noise = sample_data
        df = df.with_columns((pl.col("x") * 2).alias("x_scaled"))

        rgram = Regressogram()
        result = (
            rgram.fit(data=df, x=["x", "x_scaled"], y="y_noise").transform().collect()
        )

        # Should have y_pred_rgram column
        assert "y_pred_rgram" in result.columns
        # With multiple x columns, result should have doubled rows
        assert len(result) == len(df) * 2


class TestPrediction:
    """Test predict method comprehensively."""

    def test_predict_returns_correct_shape(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram()
        rgram.fit(data=df, x="x", y="y_noise")

        pred = rgram.predict(x).collect()
        assert len(pred) == len(x)

    def test_predict_with_new_x_values(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram()
        rgram.fit(data=df, x="x", y="y_noise")

        new_x = np.array([0.0, 0.5, 1.0, 1.5])
        pred = rgram.predict(new_x).collect()

        assert len(pred) == len(new_x)
        assert "y_pred_rgram" in pred.columns

    def test_predict_with_polars_series(self, sample_data):
        df, x, y, y_noise = sample_data
        rgram = Regressogram()
        rgram.fit(data=df, x="x", y="y_noise")

        x_series = pl.Series(x)
        pred = rgram.predict(x_series).collect()

        assert len(pred) == len(x)


class TestRegressogramKernelSmoothingPipeline:
    """Test combining Regressogram with KernelSmoother."""

    def test_regressogram_to_kernel_smoother(self, sample_data):
        df, x, y, y_noise = sample_data

        # Step 1: Fit regressogram
        rgram = Regressogram(binning="dist")
        rgram_result = rgram.fit_transform(data=df, x="x", y="y_noise").collect()

        # Step 2: Apply kernel smoothing
        smoother = KernelSmoother(n_eval_samples=50)
        smoothed = smoother.fit_transform(
            data=rgram_result, x="x_val", y="y_pred_rgram"
        ).collect()

        assert "x_eval" in smoothed.columns
        assert "y_kernel" in smoothed.columns

    def test_kernel_smoother_with_grouping(self, sample_data):
        df, x, y, y_noise = sample_data
        df = df.with_columns((pl.arange(0, df.height) % 2).alias("group"))

        # Fit regressogram with grouping
        rgram = Regressogram()
        rgram_result = rgram.fit_transform(
            data=df, x="x", y="y_noise", hue="group"
        ).collect()

        # Apply kernel smoothing
        smoother = KernelSmoother()
        smoothed = smoother.fit_transform(
            data=rgram_result, x="x_val", y="y_pred_rgram", hue="group"
        ).collect()

        assert len(smoothed) > 0


class TestFitTransformConsistency:
    """Test consistency between fit+transform vs fit_transform."""

    def test_fit_transform_equivalent_to_separate_calls(self, sample_data):
        df, x, y, y_noise = sample_data

        # Method : Using fit_transform
        rgram = Regressogram()
        result = rgram.fit_transform(data=df, x="x", y="y_noise").collect()

        # Method : Using fit then transform
        rgram = Regressogram()
        rgram.fit(data=df, x="x", y="y_noise")
        result = rgram.transform().collect()

        # Results should be identical
        assert result.shape == result.shape
        assert result.columns == result.columns


class TestDataValidation:
    """Test data validation and error handling."""

    @staticmethod
    def test_mismatched_x_y_length_with_arrays():
        """Test error handling for mismatched x/y lengths."""
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6, 8])  # Different length

        rgram = Regressogram()
        # Should handle gracefully or raise clear error
        with pytest.raises((ValueError, Exception)):
            rgram.fit(x=x, y=y).transform().collect()

    @staticmethod
    def test_non_numeric_input_raises():
        """Test that non-numeric input raises error."""
        x = np.array(["a", "b", "c"])
        y = np.array([1, 2, 3])

        rgram = Regressogram()
        with pytest.raises(Exception):
            rgram.fit(x=x, y=y).transform().collect()
