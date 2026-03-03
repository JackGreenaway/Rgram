import pytest
import polars as pl
import numpy as np


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 50
    x = np.sort(np.random.normal(0, 1, n))
    y = 1 + x
    y_noise = y + np.random.normal(0, 0.5, n)
    df = pl.DataFrame({"x": x, "y": y, "y_noise": y_noise})

    return df, x, y, y_noise


@pytest.fixture
def sample_data_with_negatives():
    np.random.seed(42)
    n = 50
    x = np.sort(np.random.normal(0, 1, n))
    y = 1 + x
    y_noise = y - 2 + np.random.normal(0, 0.5, n)  # some negatives
    df = pl.DataFrame({"x": x, "y": y, "y_noise_neg": y_noise})

    return df, x, y, y_noise
