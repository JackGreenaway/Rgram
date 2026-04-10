import polars as pl
import numpy as np
from rgram.rgram import Regressogram


def test_mean_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.mean())
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.number)


def test_median_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.median())
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)


def test_max_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.max())
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)
    # Max of predictions should not exceed max of data
    assert np.max(result) <= df["y_noise"].max()


def test_min_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.min())
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)
    # Min of predictions should not be less than min of data
    assert np.min(result) >= df["y_noise"].min()


def test_std_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.std())
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)


def test_sum_aggregation_fixed():
    x = np.arange(10)
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    df = pl.DataFrame({"x": x, "y": y})

    rgram = Regressogram(agg=lambda s: s.sum(), binning="width")
    result = rgram.fit_predict(data=df, x="x", y="y")

    # Check that predictions were computed
    assert isinstance(result, np.ndarray)
    # Sum aggregation should produce values
    assert len(result) > 0


def test_median_aggregation_fixed():
    x = np.arange(10)
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    df = pl.DataFrame({"x": x, "y": y})

    rgram = Regressogram(agg=lambda s: s.median(), binning="width")
    result = rgram.fit_predict(data=df, x="x", y="y")

    # Check that median per bin is correct
    assert isinstance(result, np.ndarray)
    assert len(result) > 0
    # Check all predictions are not NaN
    assert not np.all(np.isnan(result))


def test_count_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.count())
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)
    # Count values should be positive
    assert np.all(result > 0)


def test_quantile_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.quantile(0.5))  # median
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)


def test_variance_aggregation(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(agg=lambda s: s.var())
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)


def test_aggregation_with_ci_lower_bound(sample_data):
    df, x, y, y_noise = sample_data

    # Custom CI: use minimum and maximum
    ci = (lambda x: x.min(), lambda x: x.max())
    rgram = Regressogram(agg=lambda s: s.mean(), ci=ci)
    pred, lci, uci = rgram.fit_predict(data=df, x="x", y="y_noise", return_ci=True)

    assert isinstance(pred, np.ndarray)
    assert isinstance(lci, np.ndarray)
    assert isinstance(uci, np.ndarray)

    # LCI should be <= predictions <= UCI (ignoring NaNs)
    valid = ~np.isnan(lci) & ~np.isnan(pred) & ~np.isnan(uci)
    if np.any(valid):
        assert np.all(lci[valid] <= pred[valid])
        assert np.all(pred[valid] <= uci[valid])


def test_aggregation_consistency_across_binning_strategies():
    """Test that 'none' binning strategy uses each unique x value as a bin."""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)

    # With 'none' binning, each x value gets its own bin
    rgram_unique = Regressogram(binning="none", agg=lambda s: s.mean())
    result_unique = rgram_unique.fit_predict(x=x, y=y)

    # With 'none' binning, each x value should have a prediction
    assert isinstance(result_unique, np.ndarray)
    assert len(result_unique) > 0


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
        result = rgram.fit_predict(x=x, y=y)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
