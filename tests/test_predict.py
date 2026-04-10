import pytest
import numpy as np
import polars as pl
from rgram.rgram import Regressogram


def test_predict_returns_correct_length(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="width")
    rgram.fit(data=df, x="x", y="y_noise")
    pred = rgram.predict(x=x)

    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(x)


def test_predict_before_fit_raises():
    rgram = Regressogram()

    with pytest.raises(RuntimeError):
        rgram.predict([1, 2, 3])


def test_predict_with_series(sample_data):
    import polars as pl

    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    rgram.fit(data=df, x="x", y="y_noise")

    # Use Polars Series as input
    pred = rgram.predict(pl.Series(x))
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(x)


def test_fit_with_numpy_arrays(sample_data):
    """Test that fit works with numpy arrays as input."""
    _, x, y, y_noise = sample_data
    rgram = Regressogram()

    # Fit using raw numpy arrays
    pred = rgram.fit_predict(x=x, y=y_noise)

    assert len(pred) > 0


def test_predict_with_numpy_array(sample_data):
    """Test that predict works with numpy arrays as input."""
    _, x, y, y_noise = sample_data
    rgram = Regressogram()
    rgram.fit(x=x, y=y_noise)

    # Predict with numpy array
    pred = rgram.predict(x=x)

    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(x)


def test_fit_with_polars_dataframe(sample_data):
    """Test that fit works with polars DataFrame as input."""
    df, x, y, y_noise = sample_data

    rgram = Regressogram()
    pred = rgram.fit_predict(data=df, x="x", y="y_noise")

    assert len(pred) > 0


def test_predict_with_numpy_list():
    """Test that predict works with lists of values."""
    # Create simple test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 1

    rgram = Regressogram()
    rgram.fit(x=x, y=y)

    # Predict with list
    pred = rgram.predict(x=[1.5, 2.5, 3.5])

    assert isinstance(pred, np.ndarray)
    assert len(pred) == 3


def test_fit_and_predict_with_pandas_series(sample_data):
    """Test that fit and predict work with pandas Series (used as array-like)."""
    import pandas as pd

    _, x, y, y_noise = sample_data
    # Create pandas Series - used as array-like input
    x_series = pd.Series(x)
    y_series = pd.Series(y_noise)

    rgram = Regressogram()
    # Pandas Series are treated as array-like when data=None
    rgram.fit(x=x_series, y=y_series)

    # Predict with pandas Series
    pred = rgram.predict(x=x_series)

    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(x)


def test_predict_with_return_ci(sample_data):
    """Test that predict with return_ci=True returns tuple."""
    df, x, y, y_noise = sample_data
    rgram = Regressogram()  # With default CI settings
    rgram.fit(data=df, x="x", y="y_noise")

    # Without return_ci
    pred = rgram.predict(x)
    assert isinstance(pred, np.ndarray)

    # With return_ci
    result = rgram.predict(x, return_ci=True)
    assert isinstance(result, tuple)
    assert len(result) == 3
    y_pred, y_ci_low, y_ci_high = result
    assert isinstance(y_pred, np.ndarray)
    if y_ci_low is not None:
        assert isinstance(y_ci_low, np.ndarray)
        assert isinstance(y_ci_high, np.ndarray)
