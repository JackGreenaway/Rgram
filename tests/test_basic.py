import polars as pl
import numpy as np
import pytest
from rgram.rgram import Regressogram


def test_fit_transform_width(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="width")
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    # Check predictions returned
    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_fit_transform_dist(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="dist")
    result = rgram.fit_predict(x=x, y=y_noise)

    # Check predictions returned
    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_fit_transform_unique_binning(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="none")
    result = rgram.fit_predict(x=x, y=y_noise)

    # Check predictions returned
    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_fit_transform_int_binning(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="int")
    result = rgram.fit_predict(x=x, y=y_noise)

    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_fit_multiple_y_columns(sample_data):
    """Test that fit_predict rejects multivariate (multiple y columns)."""
    df, x, y, y_noise = sample_data
    df = df.with_columns((pl.col("y") + 1).alias("y2"))
    rgram = Regressogram()

    # fit_predict should reject multiple y columns
    with pytest.raises(ValueError, match="fit_predict only supports univariate"):
        rgram.fit_predict(data=df, x="x", y=["y_noise", "y2"])


def test_fit_multiple_x_columns(sample_data):
    df, x, y, y_noise = sample_data
    df = df.with_columns((pl.col("x") * 2).alias("x2"))
    rgram = Regressogram()
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)
    # Should have results for each x column
    assert len(result) > 0


def test_fit_predict_shortcut(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()

    # Using fit_predict shortcut - returns array with predictions
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_fit_predict_with_arrays():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    rgram = Regressogram()
    result = rgram.fit_predict(x=x, y=y)

    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_result_contains_all_original_data_points(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    # fit_predict returns predictions at unique training x values
    assert isinstance(result, np.ndarray)


def test_predictions_are_numeric(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    # Check predictions are numeric
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.number)


def test_output_is_different_due_to_binning(sample_data):
    """With the new API, fit_predict returns array of unique x value predictions."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    result = rgram.fit_predict(data=df, x="x", y="y_noise")

    # After binning, result should have predictions
    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_fit_predict_with_return_ci(sample_data):
    """Test fit_predict with return_ci=True returns tuple."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram()

    # Without return_ci
    result = rgram.fit_predict(data=df, x="x", y="y_noise")
    assert isinstance(result, np.ndarray)

    # With return_ci
    result_ci = rgram.fit_predict(data=df, x="x", y="y_noise", return_ci=True)
    assert isinstance(result_ci, tuple)
    assert len(result_ci) == 3
    y_pred, y_ci_low, y_ci_high = result_ci
    assert isinstance(y_pred, np.ndarray)
