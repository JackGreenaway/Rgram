import pytest
import polars as pl
import numpy as np
from rgram.rgram import Regressogram


def test_transform_before_fit_raises():
    rgram = Regressogram()
    with pytest.raises(RuntimeError):
        rgram.transform()


def test_predict_before_fit_raises():
    rgram = Regressogram()
    with pytest.raises(RuntimeError):
        rgram.predict([1, 2, 3])


def test_empty_data_raises():
    """Test that empty DataFrame raises an error."""
    rgram = Regressogram()
    empty_df = pl.DataFrame({"x": [], "y": []})

    with pytest.raises(Exception):
        rgram.fit(data=empty_df, x="x", y="y").transform().collect()


def test_unknown_binning_raises(sample_data):
    """Test that unknown binning strategy raises ValueError."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="unknown_strategy")

    with pytest.raises(ValueError):
        rgram.fit(data=df, x="x", y="y_noise").transform().collect()


def test_unique_binning_type(sample_data):
    """Test 'none' binning strategy uses each unique x value as a bin."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="none")
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_int_binning_type(sample_data):
    """Test 'int' binning strategy bins by integer values."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="int")
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_ci_none_no_ci_columns(sample_data):
    """Test that ci=None produces no confidence interval columns."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram(ci=None)
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    assert "y_pred_rgram_lci" not in result.columns
    assert "y_pred_rgram_uci" not in result.columns


def test_custom_ci_functions(sample_data):
    """Test custom confidence interval functions."""
    df, x, y, y_noise = sample_data

    # Use quantiles as CI bounds
    ci_lower = lambda x: x.quantile(0.05)  # noqa: E731
    ci_upper = lambda x: x.quantile(0.95)  # noqa: E731

    rgram = Regressogram(ci=(ci_lower, ci_upper))
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram_lci" in result.columns
    assert "y_pred_rgram_uci" in result.columns

    # CI upper should generally be >= CI lower
    lci = result["y_pred_rgram_lci"].drop_nulls()
    uci = result["y_pred_rgram_uci"].drop_nulls()
    if len(lci) > 0 and len(uci) > 0:
        assert (uci >= lci).all()


def test_single_point_per_bin(sample_data):
    """Test behavior when bins have single data points."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="int")
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    # Should not crash and should have predictions
    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_single_value_input():
    """Test behavior with single value input."""
    x = np.array([5.0])
    y = np.array([10.0])

    rgram = Regressogram(binning="dist")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert len(result) >= 1
    assert "y_pred_rgram" in result.columns


def test_very_large_values():
    """Test with very large numeric values."""
    x = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    y = np.array([1e8, 2e8, 3e8, 4e8, 5e8])

    rgram = Regressogram(binning="width")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_very_small_values():
    """Test with very small numeric values."""
    x = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
    y = np.array([1e-8, 2e-8, 3e-8, 4e-8, 5e-8])

    rgram = Regressogram(binning="width")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_negative_x_values():
    """Test with negative x values."""
    x = np.array([-5, -3, -1, 0, 1, 3, 5])
    y = np.array([1, 2, 3, 4, 5, 6, 7])

    rgram = Regressogram(binning="dist")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_all_same_x_values():
    """Test behavior when all x values are identical."""
    x = np.array([5.0, 5.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    rgram = Regressogram(binning="dist")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns


def test_all_same_y_values():
    """Test behavior when all y values are identical."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([5.0, 5.0, 5.0, 5.0])

    rgram = Regressogram()
    result = rgram.fit(x=x, y=y).transform().collect()

    # All predictions should be 5.0
    assert "y_pred_rgram" in result.columns
    assert (result["y_pred_rgram"] == 5.0).all()


def test_mixed_positive_negative_y():
    """Test with mixed positive and negative y values."""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([-2, -1, 1, 2, -3, 4, -5, 6])

    rgram = Regressogram(binning="width")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_dist_binning_with_duplicate_x_values():
    """Test dist binning with duplicate x values (common edge case)."""
    x = np.array(
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0]
    )
    y = np.array(
        [1.0, 1.1, 0.9, 2.0, 2.1, 1.9, 3.0, 3.1, 2.9, 4.0, 4.1, 3.9, 5.0, 5.1, 4.9]
    )

    rgram = Regressogram(binning="dist")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0
    # Check that predictions are reasonable
    assert result["y_pred_rgram"].min() >= y.min() - 1
    assert result["y_pred_rgram"].max() <= y.max() + 1


def test_dist_binning_with_many_duplicate_clusters():
    """Test dist binning with multiple clusters of identical x values."""
    # Create data with many duplicates clustered in ranges
    x = np.array([1.0] * 10 + [2.0] * 10 + [3.0] * 10 + [4.0] * 10 + [5.0] * 10)
    y = np.array(list(range(1, 11)) * 5)  # 1-10 repeated 5 times

    rgram = Regressogram(binning="dist")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_dist_binning_with_mostly_duplicates():
    """Test dist binning when most values are duplicates with few unique values."""
    x = np.array([1.0] * 20 + [2.0] * 5 + [3.0] * 20 + [4.0] * 5)
    y = np.array([1.0] * 20 + [2.0] * 5 + [3.0] * 20 + [4.0] * 5)

    rgram = Regressogram(binning="dist")
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_dist_binning_duplicates_with_custom_agg():
    """Test dist binning with duplicates and custom aggregation function."""
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    rgram = Regressogram(binning="dist", agg=lambda s: s.median())
    result = rgram.fit(x=x, y=y).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0
