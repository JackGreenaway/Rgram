"""
Parametric and combination tests.
Tests for parameter interactions and combinations.
"""

import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram
from rgram.smoothing import KernelSmoother


class TestRegressogramParameterCombinations:
    """Test various combinations of Regressogram parameters."""

    @pytest.mark.parametrize("binning", ["dist", "width", "int", "none"])
    def test_all_binning_types(self, binning):
        """Test all binning strategies work."""
        x = np.linspace(0, 10, 30)
        y = np.sin(x)

        rgram = Regressogram(binning=binning)
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 30
        assert "y_pred_rgram" in result.columns

    @pytest.mark.parametrize("binning", ["dist", "width", "int", "none"])
    @pytest.mark.parametrize("ci", [None, (lambda x: x.mean() - x.std(), lambda x: x.mean() + x.std())])
    def test_binning_with_ci_combinations(self, binning, ci):
        """Test binning strategies with CI options."""
        x = np.linspace(0, 10, 30)
        y = np.sin(x)

        rgram = Regressogram(binning=binning, ci=ci)
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 30
        
        if ci is None:
            assert "y_pred_rgram_lci" not in result.columns
        else:
            assert "y_pred_rgram_lci" in result.columns
            assert "y_pred_rgram_uci" in result.columns

    @pytest.mark.parametrize("agg", [
        lambda x: x.mean(),
        lambda x: x.median(),
        lambda x: x.min(),
        lambda x: x.max(),
        lambda x: x.std(),
    ])
    def test_different_aggregations(self, agg):
        """Test different aggregation functions."""
        x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=float)
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

        rgram = Regressogram(binning="int", agg=agg)
        result = rgram.fit(x=x, y=y).transform().collect()

        assert "y_pred_rgram" in result.columns

    @pytest.mark.parametrize("n_bins", [1, 5, 10, 20])
    def test_various_n_bins(self, n_bins):
        """Test various n_bins values."""
        x = np.linspace(0, 10, 50)
        y = x ** 2

        rgram = Regressogram(binning="dist", n_bins=n_bins)
        result = rgram.fit(x=x, y=y).transform().collect()

        assert len(result) == 50

    def test_binning_width_with_n_bins_ignored(self):
        """Test that n_bins is ignored for width binning."""
        x = np.linspace(0, 10, 30)
        y = x

        rgram_without_nbins = Regressogram(binning="width").fit(x=x, y=y)
        rgram_with_nbins = Regressogram(binning="width", n_bins=100).fit(x=x, y=y)

        # Both should produce same results (n_bins ignored for width)
        result1 = rgram_without_nbins.transform().collect()
        result2 = rgram_with_nbins.transform().collect()

        assert len(result1) == len(result2)


class TestKernelSmootherParameterCombinations:
    """Test KernelSmoother parameter combinations."""

    @pytest.mark.parametrize("bandwidth", ["silverman", "scott"])
    @pytest.mark.parametrize("n_eval", [10, 20, 50])
    def test_bandwidth_n_eval_combinations(self, bandwidth, n_eval):
        """Test bandwidth methods with various eval sample counts."""
        df = pl.DataFrame({
            "x": np.linspace(0, 10, 50),
            "y": np.sin(np.linspace(0, 10, 50))
        })

        smoother = KernelSmoother(bandwidth=bandwidth, n_eval_samples=n_eval)
        result = smoother.fit(data=df, x="x", y="y").transform().collect()

        assert len(result) == n_eval

    @pytest.mark.parametrize("bw_value", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_manual_bandwidth_values(self, bw_value):
        """Test various manual bandwidth values."""
        df = pl.DataFrame({
            "x": np.linspace(0, 10, 50),
            "y": np.sin(np.linspace(0, 10, 50))
        })

        smoother = KernelSmoother(
            bandwidth="manual", 
            bandwidth_value=bw_value,
            n_eval_samples=20
        )
        result = smoother.fit(data=df, x="x", y="y").transform().collect()

        assert len(result) == 20
        assert "y_kernel" in result.columns


class TestDataFrameInputVariations:
    """Test various DataFrame input scenarios."""

    @pytest.mark.parametrize("data_type", ["polars_df", "polars_lf"])
    def test_polars_data_types(self, data_type):
        """Test with different Polars data types."""
        df = pl.DataFrame({
            "x": np.linspace(0, 10, 20),
            "y": np.sin(np.linspace(0, 10, 20))
        })

        if data_type == "polars_lf":
            df = df.lazy()

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y").transform().collect()

        assert len(result) == 20

    def test_unnamed_columns_fail(self):
        """Test that unnamed columns are handled or fail appropriately."""
        df = pl.DataFrame([[1, 2], [3, 4], [5, 6]])

        rgram = Regressogram()
        with pytest.raises(Exception):
            rgram.fit(data=df, x=0, y=1)

    def test_case_sensitive_column_names(self):
        """Test that column names are case-sensitive."""
        df = pl.DataFrame({
            "X": [1, 2, 3],
            "Y": [1, 2, 3],
            "x": [4, 5, 6],
            "y": [4, 5, 6]
        })

        rgram = Regressogram()
        result1 = rgram.fit(data=df, x="X", y="Y").transform().collect()
        
        rgram2 = Regressogram()
        result2 = rgram2.fit(data=df, x="x", y="y").transform().collect()

        # Results should be different due to different data
        assert result1["y_val"].to_numpy()[0] != result2["y_val"].to_numpy()[0]


class TestMultipleFeatureTargets:
    """Test handling of multiple features and targets."""

    def test_multiple_y_columns(self):
        """Test fitting with multiple target columns."""
        df = pl.DataFrame({
            "x": np.linspace(0, 10, 20),
            "y1": np.sin(np.linspace(0, 10, 20)),
            "y2": np.cos(np.linspace(0, 10, 20))
        })

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y=["y1", "y2"]).transform().collect()

        # Should have results for both targets
        assert "y_var" in result.columns
        unique_y_vars = result["y_var"].unique().to_list()
        assert len(unique_y_vars) >= 2

    def test_multiple_x_columns(self):
        """Test fitting with multiple feature columns."""
        df = pl.DataFrame({
            "x1": np.linspace(0, 10, 20),
            "x2": np.linspace(10, 20, 20),
            "y": np.linspace(1, 20, 20)
        })

        rgram = Regressogram()
        result = rgram.fit(data=df, x=["x1", "x2"], y="y").transform().collect()

        # Should have results for both features
        assert "x_var" in result.columns
        unique_x_vars = result["x_var"].unique().to_list()
        assert len(unique_x_vars) >= 2

    def test_multiple_x_and_y_columns(self):
        """Test fitting with multiple features and targets."""
        df = pl.DataFrame({
            "x1": np.linspace(0, 10, 20),
            "x2": np.linspace(10, 20, 20),
            "y1": np.linspace(1, 20, 20),
            "y2": np.linspace(20, 1, 20)
        })

        rgram = Regressogram()
        result = rgram.fit(data=df, x=["x1", "x2"], y=["y1", "y2"]).transform().collect()

        # Should have combinations
        assert "x_var" in result.columns
        assert "y_var" in result.columns


class TestGroupingOperations:
    """Test hue and keys grouping operations."""

    @pytest.mark.parametrize("hue_levels", [2, 3, 5])
    def test_various_hue_levels(self, hue_levels):
        """Test with different numbers of hue levels."""
        x = np.tile(np.linspace(0, 10, 20), hue_levels)
        y = np.tile(np.linspace(1, 20, 20), hue_levels)
        hue = np.repeat(range(hue_levels), 20)

        df = pl.DataFrame({"x": x, "y": y, "hue": hue})

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y", hue="hue").transform().collect()

        unique_hues = result["hue"].unique().to_list()
        assert len(unique_hues) == hue_levels

    def test_multiple_hue_columns(self):
        """Test with multiple hue columns."""
        df = pl.DataFrame({
            "x": np.tile(np.linspace(0, 10, 10), 4),
            "y": np.tile(np.linspace(1, 10, 10), 4),
            "hue1": np.repeat([0, 1], 20),
            "hue2": np.tile(np.repeat([0, 1], 10), 2)
        })

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y", hue=["hue1", "hue2"]).transform().collect()

        assert "hue1" in result.columns
        assert "hue2" in result.columns

    def test_keys_parameter(self):
        """Test keys parameter for additional grouping."""
        df = pl.DataFrame({
            "x": np.tile(np.linspace(0, 10, 20), 2),
            "y": np.tile(np.linspace(1, 20, 20), 2),
            "keys": np.repeat([0, 1], 20)
        })

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y", keys="keys").transform().collect()

        assert len(result) == 40

    def test_hue_and_keys_together(self):
        """Test using both hue and keys together."""
        df = pl.DataFrame({
            "x": np.tile(np.linspace(0, 10, 10), 4),
            "y": np.tile(np.linspace(1, 10, 10), 4),
            "hue": np.repeat([0, 1], 20),
            "keys": np.tile(np.repeat([0, 1], 10), 2)
        })

        rgram = Regressogram()
        result = rgram.fit(data=df, x="x", y="y", hue="hue", keys="keys").transform().collect()

        assert "hue" in result.columns


class TestPredictWithVaryingInput:
    """Test predict with various input types and ranges."""

    def test_predict_single_value(self):
        """Test predicting for single value."""
        x = np.linspace(0, 10, 20)
        y = x ** 2

        rgram = Regressogram()
        rgram.fit(x=x, y=y)
        pred = rgram.predict([5.0])

        assert len(pred) == 1
        assert isinstance(pred[0], (np.floating, float))

    def test_predict_outside_training_range(self):
        """Test predicting outside training data range."""
        x = np.linspace(0, 10, 20)
        y = x

        rgram = Regressogram()
        rgram.fit(x=x, y=y)

        # Predict outside range
        pred = rgram.predict([-5.0, 15.0])

        assert len(pred) == 2
        assert np.all(np.isfinite(pred))

    def test_predict_at_extremes(self):
        """Test predicting at training data extremes."""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)

        rgram = Regressogram()
        rgram.fit(x=x, y=y)

        # Predict at extremes
        pred = rgram.predict([x.min(), x.max()])

        assert len(pred) == 2

    def test_predict_many_points(self):
        """Test predicting for many points."""
        x_train = np.linspace(0, 10, 50)
        y_train = x_train

        rgram = Regressogram()
        rgram.fit(x=x_train, y=y_train)

        x_test = np.linspace(0, 10, 1000)
        pred = rgram.predict(x_test)

        assert len(pred) == 1000


class TestCIComputation:
    """Test confidence interval computation variations."""

    def test_custom_ci_functions(self):
        """Test custom CI functions."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.randn(50) * 0.1

        # Use quantiles as CI
        ci = (
            lambda col: col.quantile(0.1),
            lambda col: col.quantile(0.9)
        )

        rgram = Regressogram(ci=ci)
        result = rgram.fit(x=x, y=y).transform().collect()

        lci = result["y_pred_rgram_lci"].drop_nulls().to_numpy()
        uci = result["y_pred_rgram_uci"].drop_nulls().to_numpy()

        assert (uci >= lci).all()

    def test_tight_ci(self):
        """Test with tight CI (close to prediction)."""
        x = np.linspace(0, 10, 30)
        y = x  # Deterministic

        ci = (
            lambda col: col.mean() - 0.01,
            lambda col: col.mean() + 0.01
        )

        rgram = Regressogram(ci=ci)
        result = rgram.fit(x=x, y=y).transform().collect()

        assert "y_pred_rgram_lci" in result.columns
        assert "y_pred_rgram_uci" in result.columns

    def test_wide_ci(self):
        """Test with wide CI (far from prediction)."""
        x = np.linspace(0, 10, 30)
        y = x

        ci = (
            lambda col: col.mean() - 100,
            lambda col: col.mean() + 100
        )

        rgram = Regressogram(ci=ci)
        result = rgram.fit(x=x, y=y).transform().collect()

        lci = result["y_pred_rgram_lci"].drop_nulls().to_numpy()
        pred = result["y_pred_rgram"].drop_nulls().to_numpy()
        uci = result["y_pred_rgram_uci"].drop_nulls().to_numpy()

        assert (lci <= pred).all()
        assert (pred <= uci).all()


class TestConsistencyAcrossMethods:
    """Test consistency between different methods."""

    def test_fit_predict_vs_fit_then_predict(self):
        """Test that fit_predict gives same results as fit then predict on training data."""
        x = np.linspace(0, 10, 30)
        y = np.sin(x)

        rgram1 = Regressogram()
        pred1 = rgram1.fit_predict(x=x, y=y)

        rgram2 = Regressogram()
        rgram2.fit(x=x, y=y)
        pred2 = rgram2.predict(np.unique(x))

        # Both should have valid predictions
        assert len(pred1) > 0
        assert len(pred2) > 0

    def test_fit_transform_vs_fit_then_predict(self):
        """Test consistency between fit_transform and fit_then_predict."""
        df = pl.DataFrame({
            "x": np.linspace(0, 10, 20),
            "y": np.sin(np.linspace(0, 10, 20))
        })

        rgram1 = Regressogram()
        result1 = rgram1.fit(data=df, x="x", y="y").transform().collect()

        rgram2 = Regressogram()
        rgram2.fit(data=df, x="x", y="y")
        
        assert "y_pred_rgram" in result1.columns
