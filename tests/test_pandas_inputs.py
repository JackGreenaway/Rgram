"""Pandas named selection, nullable numeric inputs, and positional row semantics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rgram import CoverageSearchCV, KernelSmoother, Regressogram


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_named_pandas_selection_matches_arrays_and_preserves_snapshot(cls):
    X = np.arange(12.0)
    y = np.sin(X)
    weights = np.arange(1.0, 13.0)
    frame = pd.DataFrame(
        {"feature": X, "target": y, "unused": [None] * len(X)}, index=[8, 3, 3, 1] * 3
    )
    original = frame.copy(deep=True)
    model = cls().fit(X="feature", y="target", data=frame, sample_weight=weights)
    reference = cls().fit(X, y, sample_weight=weights)
    query = pd.DataFrame({"feature": [8.0, 3.0, 3.0]}, index=[99, 1, 1])
    expected = reference.predict(query.to_numpy())
    np.testing.assert_allclose(model.predict(query), expected)
    assert model.feature_names_in_.tolist() == ["feature"]
    pd.testing.assert_frame_equal(frame, original)
    frame.loc[:, "feature"] = 999
    frame.loc[:, "target"] = 999
    np.testing.assert_allclose(model.predict(query), expected)
    np.testing.assert_array_equal(model.X_, X)
    np.testing.assert_array_equal(model.sample_weight_, weights)


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
@pytest.mark.parametrize("dtype", ["Int64", "Float64", "Float32"])
def test_nullable_numeric_frames_and_series(cls, dtype):
    frame = pd.DataFrame(
        {
            "feature": pd.Series(range(12), dtype=dtype),
            "target": pd.Series(range(12), dtype=dtype),
        }
    )
    named = cls().fit("feature", "target", data=frame)
    direct = cls().fit(frame[["feature"]], frame["target"])
    np.testing.assert_allclose(
        named.predict(frame[["feature"]]), direct.predict(frame[["feature"]])
    )
    np.testing.assert_allclose(
        named.predict(frame["feature"]),
        cls().fit(np.arange(12), np.arange(12)).predict(np.arange(12)),
    )


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_pandas_pairs_are_positional_not_index_aligned(cls):
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}, index=[7, 4, 1, 1])
    y = pd.Series([4.0, 3.0, 2.0, 1.0], index=[1, 1, 4, 7])
    model = cls().fit(X, y)
    np.testing.assert_array_equal(model.y_, y.to_numpy())
    np.testing.assert_allclose(
        model.predict(X), cls().fit(X.to_numpy(), y.to_numpy()).predict(X.to_numpy())
    )


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
@pytest.mark.parametrize("column", ["feature", "target"])
def test_pandas_missing_rows_rejected_before_conversion(cls, column):
    frame = pd.DataFrame(
        {
            "feature": pd.Series([0, 1, 2], dtype="Int64"),
            "target": pd.Series([0, 1, 2], dtype="Int64"),
        }
    )
    frame.loc[1, column] = pd.NA
    for fit in (
        lambda: cls().fit("feature", "target", data=frame),
        lambda: cls().fit(frame[["feature"]], frame["target"]),
    ):
        with pytest.raises(ValueError, match="no rows were dropped"):
            fit()
    model = cls().fit([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="no rows were dropped"):
        model.predict(pd.DataFrame({"feature": pd.Series([1, pd.NA], dtype="Int64")}))


@pytest.mark.parametrize("bad", [[0.0, np.inf, 2.0], [0.0, 1j, 2.0], ["0", "1", "2"]])
def test_invalid_pandas_selected_values(bad):
    frame = pd.DataFrame({"feature": bad, "target": [0.0, 1.0, 2.0]})
    with pytest.raises((ValueError, TypeError)):
        Regressogram().fit("feature", "target", data=frame)


def test_pandas_same_column_and_duplicate_column_labels():
    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    model = Regressogram(n_bins=1).fit("feature", "feature", data=frame)
    np.testing.assert_allclose(model.predict(frame), [1.5] * 4)
    duplicate = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], columns=["feature", "feature", "target"]
    )
    with pytest.raises(ValueError, match="duplicated"):
        Regressogram().fit("feature", "target", data=duplicate)
    with pytest.raises(KeyError):
        Regressogram().fit("missing", "feature", data=frame)


def test_pandas_coverage_search_and_pipeline():
    frame = pd.DataFrame({"feature": np.arange(20.0), "target": np.arange(20.0)})
    search = CoverageSearchCV(Regressogram(), {"n_bins": [1, 3]}, cv=3).fit(
        "feature", "target", data=frame
    )
    assert search.predict(frame[["feature"]]).shape == (20,)
    assert search.feature_names_in_.tolist() == ["feature"]
    pipe = make_pipeline(StandardScaler(), KernelSmoother(kernel="gaussian"))
    pipe.fit(frame[["feature"]], frame["target"])
    assert np.isfinite(pipe.predict(frame[["feature"]])).all()
