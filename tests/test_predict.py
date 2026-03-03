import pytest
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
