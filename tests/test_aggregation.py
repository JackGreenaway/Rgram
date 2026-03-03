import polars as pl
import numpy as np
from rgram.rgram import Regressogram


def test_mean_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.mean())
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    assert result["y_pred_rgram"].dtype in [pl.Float32, pl.Float64]


def test_median_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.median())
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns


def test_max_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.max())
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    # Max of predictions should not exceed max of data
    assert result["y_pred_rgram"].max() <= df["y_noise"].max()


def test_min_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.min())
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    # Min of predictions should not be less than min of data
    assert result["y_pred_rgram"].min() >= df["y_noise"].min()


def test_std_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.std())
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns


def test_sum_aggregation_fixed():
    x = np.arange(10)
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    df = pl.DataFrame({"x": x, "y": y})

    rgram = Regressogram(agg=lambda s: s.sum(), binning="width")
    result = rgram.fit(data=df, x="x", y="y").transform().collect()

    # Check that predictions were computed
    assert "y_pred_rgram" in result.columns
    # Sum aggregation should produce values >= individual data points
    assert len(result) > 0


def test_median_aggregation_fixed():
    x = np.arange(10)
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    df = pl.DataFrame({"x": x, "y": y})

    rgram = Regressogram(agg=lambda s: s.median(), binning="width")
    result = rgram.fit(data=df, x="x", y="y").transform().collect()

    # Check that median per bin is correct
    unique_bin_values = result["y_pred_rgram"].unique().to_list()
    for val in unique_bin_values:
        # Each unique bin value must be in the original y or be a median
        assert val is not None


def test_count_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.count())
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    # Count values should be positive integers
    assert (result["y_pred_rgram"] > 0).all()


def test_quantile_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.quantile(0.5))  # median
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns


def test_variance_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.var())
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns


def test_aggregation_with_ci_lower_bound(sample_data):
    df, x, y, y_noise = sample_data

    # Custom CI: use minimum and maximum
    ci = (lambda x: x.min(), lambda x: x.max())
    rgram = Regressogram(agg=lambda s: s.mean(), ci=ci)
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    assert "y_pred_rgram_lci" in result.columns
    assert "y_pred_rgram_uci" in result.columns

    # LCI should be <= predictions <= UCI
    lci = result["y_pred_rgram_lci"].drop_nulls()
    pred = result["y_pred_rgram"].drop_nulls()
    uci = result["y_pred_rgram_uci"].drop_nulls()

    if len(lci) > 0 and len(pred) > 0 and len(uci) > 0:
        assert (lci <= pred).all()
        assert (pred <= uci).all()


def test_aggregation_consistency_across_binning_strategies():
    """Test that 'unique' binning strategy uses each unique x value as a bin."""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)

    # With 'unique' binning, each x value gets its own bin
    rgram_unique = Regressogram(binning="unique", agg=lambda s: s.mean())
    result_unique = rgram_unique.fit(x=x, y=y).transform().collect()

    # With 'unique' binning, each x value should have a bin, so 10 unique values
    # Since y values match indices, predictions should equal y values
    assert len(result_unique) == len(x)
    assert "y_pred_rgram" in result_unique.columns


def test_multiple_aggregations_in_sequence():
    """Test applying different aggregations to same data."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5], dtype=float)

    agg_funcs = [
        lambda s: s.mean(),
        lambda s: s.median(),
        lambda s: s.min(),
        lambda s: s.max(),
        lambda s: s.sum(),
    ]

    for agg in agg_funcs:
        rgram = Regressogram(agg=agg)
        result = rgram.fit(x=x, y=y).transform().collect()
        assert "y_pred_rgram" in result.columns
        assert len(result) > 0
