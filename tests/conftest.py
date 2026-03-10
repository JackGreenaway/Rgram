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


@pytest.fixture
def linear_data():
    """Fixture for linear relationship data."""
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y = 2 * x + 3 + np.random.normal(0, 0.5, 30)
    return x, y


@pytest.fixture
def sine_data():
    """Fixture for sinusoidal data."""
    np.random.seed(42)
    x = np.linspace(0, 2 * np.pi, 50)
    y = np.sin(x) + np.random.normal(0, 0.1, 50)
    return x, y


@pytest.fixture
def exponential_data():
    """Fixture for exponential data."""
    np.random.seed(42)
    x = np.linspace(0, 5, 40)
    y = np.exp(x / 2) + np.random.normal(0, 1, 40)
    return x, y


@pytest.fixture
def outlier_data():
    """Fixture for data with outliers."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = x + np.random.normal(0, 0.5, 50)
    # Add outliers
    y[10] = 50
    y[25] = -30
    return x, y


@pytest.fixture
def clustered_data():
    """Fixture for clustered data."""
    np.random.seed(42)
    x1 = np.random.normal(2, 0.5, 25)
    x2 = np.random.normal(8, 0.5, 25)
    x = np.concatenate([x1, x2])
    y = x + np.random.normal(0, 0.5, 50)
    return x, y
