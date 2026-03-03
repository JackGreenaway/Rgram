import pytest
import numpy as np
from rgram.rgram import Regressogram


def test_predict_returns_correct_length(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="width")
    rgram.fit(data=df, x="x", y="y_noise")
    pred = rgram.predict(x=x).collect()

    assert len(pred) == len(x)
    assert "y_pred_rgram" in pred.columns


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
    pred = rgram.predict(pl.Series(x)).collect()
    assert len(pred) == len(x)
    assert "y_pred_rgram" in pred.columns


def test_predict_after_transform_consistency(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    rgram.fit(data=df, x="x", y="y_noise")
    transformed = rgram.transform().collect()
    pred = rgram.predict(x).collect()

    # Predictions should correspond to values in transformed result
    assert (pred["y_pred_rgram"].is_in(transformed["y_pred_rgram"].implode())).all()


def test_fit_with_numpy_arrays(sample_data):
    """Test that fit works with numpy arrays as input."""
    _, x, y, y_noise = sample_data
    rgram = Regressogram()

    # Fit using raw numpy arrays
    rgram.fit(x=x, y=y_noise)
    transformed = rgram.transform().collect()

    assert len(transformed) > 0
    assert "y_pred_rgram" in transformed.columns


def test_predict_with_numpy_array(sample_data):
    """Test that predict works with numpy arrays as input."""
    _, x, y, y_noise = sample_data
    rgram = Regressogram()
    rgram.fit(x=x, y=y_noise)

    # Predict with numpy array
    pred = rgram.predict(x=x).collect()

    assert len(pred) == len(x)
    assert "y_pred_rgram" in pred.columns


def test_fit_with_pandas_dataframe(sample_data):
    """Test that fit works with pandas DataFrame converted to Polars."""
    import pandas as pd
    import polars as pl

    _, x, y, y_noise = sample_data
    # Create pandas DataFrame and convert to Polars
    df_pandas = pd.DataFrame({"x": x, "y_noise": y_noise})
    df_polars = pl.from_pandas(df_pandas)

    rgram = Regressogram()
    rgram.fit(data=df_polars, x="x", y="y_noise")
    transformed = rgram.transform().collect()

    assert len(transformed) > 0
    assert "y_pred_rgram" in transformed.columns


def test_fit_with_polars_dataframe(sample_data):
    """Test that fit works with polars DataFrame as input."""
    df, x, y, y_noise = sample_data

    rgram = Regressogram()
    rgram.fit(data=df, x="x", y="y_noise")
    transformed = rgram.transform().collect()

    assert len(transformed) > 0
    assert "y_pred_rgram" in transformed.columns


def test_predict_with_numpy_list():
    """Test that predict works with lists of values."""
    # Create simple test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 1

    rgram = Regressogram()
    rgram.fit(x=x, y=y)

    # Predict with list
    pred = rgram.predict(x=[1.5, 2.5, 3.5]).collect()

    assert len(pred) == 3
    assert "y_pred_rgram" in pred.columns


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
    pred = rgram.predict(x=x_series).collect()

    assert len(pred) == len(x)
    assert "y_pred_rgram" in pred.columns
