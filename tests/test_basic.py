import polars as pl
import numpy as np
from rgram.rgram import Regressogram


def test_fit_transform_width(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="width")
    result = rgram.fit(data=df, x="x", y="y_noise").transform().collect()

    # Columns
    assert "y_pred_rgram" in result.columns
    assert "y_pred_rgram_lci" in result.columns
    assert "y_pred_rgram_uci" in result.columns
    assert "x_val" in result.columns
    assert "y_val" in result.columns


def test_fit_transform_dist(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="dist")
    result = rgram.fit(x=x, y=y_noise).transform().collect()

    # Columns
    assert "y_pred_rgram" in result.columns
    assert "y_pred_rgram_lci" in result.columns
    assert "y_pred_rgram_uci" in result.columns
    assert len(result) > 0


def test_fit_transform_unique_binning(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="unique")
    result = rgram.fit(x=x, y=y_noise).transform().collect()

    assert "y_pred_rgram" in result.columns
    # With 'unique' binning, each unique x value is its own bin
    # (results may contain multiple rows per unique x value due to aggregation)
    assert len(result) > 0


def test_fit_transform_int_binning(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram(binning="int")
    result = rgram.fit(x=x, y=y_noise).transform().collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_fit_multiple_y_columns(sample_data):
    df, x, y, y_noise = sample_data
    df = df.with_columns((pl.col("y") + 1).alias("y2"))
    rgram = Regressogram()
    result = rgram.fit(data=df, x="x", y=["y_noise", "y2"]).transform().collect()

    assert "y_pred_rgram" in result.columns
    # Should have results (y_var is dropped in transform)
    assert len(result) == len(df) * 2  # One result per input data point per y column


def test_fit_multiple_x_columns(sample_data):
    df, x, y, y_noise = sample_data
    df = df.with_columns((pl.col("x") * 2).alias("x2"))
    rgram = Regressogram()
    result = rgram.fit(data=df, x=["x", "x2"], y="y_noise").transform().collect()

    assert "y_pred_rgram" in result.columns
    # Should have results (x_var is dropped in transform)
    assert len(result) == len(df) * 2  # One result per input data point per x column


def test_fit_with_hue(sample_data):
    df, x, y, y_noise = sample_data
    # Create a categorical hue
    df = df.with_columns((pl.arange(0, df.height) % 2).alias("group"))
    rgram = Regressogram()
    result = rgram.fit(data=df, x="x", y="y_noise", hue="group").transform().collect()

    assert "y_pred_rgram" in result.columns
    assert "group" in result.columns


def test_fit_with_multiple_hues(sample_data):
    df, x, y, y_noise = sample_data
    df = df.with_columns(
        [
            (pl.arange(0, df.height) % 2).alias("group1"),
            (pl.arange(0, df.height) % 3).alias("group2"),
        ]
    )
    rgram = Regressogram()
    result = (
        rgram.fit(data=df, x="x", y="y_noise", hue=["group1", "group2"])
        .transform()
        .collect()
    )

    assert "y_pred_rgram" in result.columns
    assert "group1" in result.columns
    assert "group2" in result.columns


def test_fit_with_keys(sample_data):
    df, x, y, y_noise = sample_data
    df = df.with_columns((pl.arange(0, df.height) % 2).alias("grouping"))
    rgram = Regressogram()
    result = (
        rgram.fit(data=df, x="x", y="y_noise", keys="grouping").transform().collect()
    )

    assert "y_pred_rgram" in result.columns


def test_fit_transform_shortcut(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()

    # Using fit_transform shortcut
    result = rgram.fit_transform(data=df, x="x", y="y_noise").collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_fit_transform_with_arrays():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    rgram = Regressogram()
    result = rgram.fit_transform(x=x, y=y).collect()

    assert "y_pred_rgram" in result.columns
    assert len(result) > 0


def test_result_contains_all_original_data_points(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    result = rgram.fit_transform(data=df, x="x", y="y_noise").collect()

    # Should have same number of rows as input
    assert len(result) == len(df)


def test_predictions_are_numeric(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    result = rgram.fit_transform(data=df, x="x", y="y_noise").collect()

    # Check predictions are numeric
    assert result["y_pred_rgram"].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]


def test_output_is_sorted_by_x_val(sample_data):
    df, x, y, y_noise = sample_data
    rgram = Regressogram()
    result = rgram.fit_transform(data=df, x="x", y="y_noise").collect()

    # Check that x_val is sorted
    x_vals = result["x_val"].to_list()
    assert x_vals == sorted(x_vals)
