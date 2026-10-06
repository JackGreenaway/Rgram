"""Explicit aggregation and data handling contracts."""

import pickle
import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init

from rgram import (
    CoverageSearchCV,
    ExtrapolationWarning,
    KernelSmoother,
    Regressogram,
    RgramWarning,
    SupportWarning,
    UnsortedInputWarning,
    array_aggregation,
    plot_diagnostics,
    quantile,
    weighted_aggregation,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("mean", 4.5),
        ("median", 4.0),
        ("min", 1.0),
        ("max", 9.0),
        ("sum", 18.0),
        ("count", 4.0),
        ("std", np.std([1, 2, 6, 9], ddof=1)),
        ("var", np.var([1, 2, 6, 9], ddof=1)),
        ("first", 1.0),
        ("last", 9.0),
    ],
)
def test_named_aggregations(name, expected):
    model = Regressogram(n_bins=1, agg=name).fit([0, 1, 2, 3], [1, 2, 6, 9])
    np.testing.assert_allclose(model.predict([1.0]), [expected])


def test_numeric_adapter_order_and_client_data_isolation():
    x = np.array([3.0, 0.0, 2.0, 1.0])
    y = np.array([9.0, 1.0, 6.0, 2.0])
    received = []

    def custom(values):
        received.append(values.copy())
        answer = values[0] - values[-1]
        values[:] = 999  # Callback owns a copy, not caller/model storage.
        return answer

    with pytest.warns(UnsortedInputWarning):
        model = Regressogram(n_bins=1, agg=array_aggregation(custom)).fit(x, y)
    np.testing.assert_array_equal(received[0], y)
    np.testing.assert_array_equal(model.X_, x)
    np.testing.assert_array_equal(model.y_, y)
    np.testing.assert_allclose(model.predict([1.0]), [7.0])


def test_quantile_and_explicit_summary_endpoints():
    model = Regressogram(
        n_bins=1, agg=quantile(0.75), ci=(quantile(0.1), quantile(0.9))
    )
    model.fit([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 8.0, 10.0])
    assert model.predict([1.0]).shape == (1,)
    pred, lo, hi = (
        model.predict_diagnostics([1.0])
        .select("prediction", "summary_lower", "summary_upper")
        .to_numpy()
        .T
    )
    np.testing.assert_allclose(
        [pred[0], lo[0], hi[0]], np.quantile([0, 2, 8, 10], [0.75, 0.1, 0.9])
    )
    empty = Regressogram(n_bins=1).fit([0, 1], [0, 1])
    assert "summary_lower" not in empty.predict_diagnostics([0]).columns


@pytest.mark.parametrize(
    "agg",
    [
        "mean",
        array_aggregation(lambda y, w: np.average(y, weights=w), weighted=True),
        weighted_aggregation(lambda y, w: (y * w).sum() / w.sum()),
    ],
)
def test_weighted_mean_and_inspection(agg):
    weights = np.array([1.0, 3.0, 0.0, 2.0])
    y = np.array([1.0, 5.0, 100.0, 9.0])
    model = Regressogram(n_bins=1, agg=agg).fit([0, 1, 2, 3], y, sample_weight=weights)
    np.testing.assert_allclose(model.predict([1.0]), [np.average(y, weights=weights)])
    np.testing.assert_array_equal(model.sample_weight_, weights)
    assert model.bins_["n_samples"].sum() == 4
    assert model.bins_["n_positive_weight"].sum() == 3
    assert model.bins_["weight_sum"].sum() == 6
    assert model.data_summary_["n_rows_retained"] == 4
    assert model.data_summary_["rows_dropped"] == 0


def test_weighted_sum_and_zero_mass_mean_policy():
    x, y, weights = [0, 1, 2, 3], [1, 4, 8, 9], [0, 0, 1, 2]
    total = Regressogram(n_bins=1, agg="sum").fit(x, y, sample_weight=weights)
    np.testing.assert_allclose(total.predict([1.0]), [26.0])
    model = Regressogram(binning="width", n_bins=2).fit(x, y, sample_weight=weights)
    with pytest.warns(SupportWarning):
        result = model.predict_diagnostics([0.0, 2.0])
    assert result["n_samples"].to_list() == [2, 2]
    assert result["weight_sum"].to_list() == [0.0, 3.0]
    assert np.isnan(result["prediction"][0])
    model.set_params(unsupported="raise")
    with pytest.raises(ValueError, match="no finite"):
        model.predict([0.0])


@pytest.mark.parametrize(
    "agg", ["median", "count", lambda y: y.mean(), array_aggregation(np.mean)]
)
def test_weights_never_silently_ignored(agg):
    with pytest.raises(ValueError, match="sample_weight"):
        Regressogram(agg=agg).fit([0, 1, 2], [1, 2, 3], sample_weight=[1, 2, 1])


@pytest.mark.parametrize("func", [lambda y: y, lambda y: "bad", lambda y: 1 + 1j])
def test_scalar_contract(func):
    with pytest.raises(ValueError, match="one real numeric scalar"):
        Regressogram(agg=array_aggregation(func)).fit([0, 1], [0, 1])


@pytest.mark.parametrize("agg", ["mean", array_aggregation(np.median), quantile(0.75)])
def test_aggregation_clone_and_pickle(agg):
    model = Regressogram(n_bins=1, agg=agg)
    clone(model).fit([0, 1, 2], [1, 2, 4])
    model.fit([0, 1, 2], [1, 2, 4])
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict([1]), loaded.predict([1]))


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_feature_names_checked_and_cleared_on_refit(cls):
    frame = pl.DataFrame({"feature": [0.0, 1.0, 2.0], "target": [1.0, 3.0, 2.0]})
    model = cls().fit("feature", "target", data=frame.lazy())
    assert model.feature_names_in_.tolist() == ["feature"]
    expected = model.predict([0.0, 1.0])
    np.testing.assert_allclose(
        model.predict(pl.DataFrame({"feature": [0.0, 1.0]})), expected
    )
    with pytest.raises(ValueError, match="Feature name"):
        model.predict(pl.DataFrame({"wrong": [0.0, 1.0]}))
    with pytest.raises(ValueError, match="Feature name"):
        model.predict_interval(pl.DataFrame({"wrong": [0.0]}), n_resamples=2)
    model.fit([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])
    assert not hasattr(model, "feature_names_in_")
    check_no_attributes_set_in_init(cls.__name__, cls())


def test_smoother_never_sorts_and_snapshots_weights(monkeypatch):
    x, y, weights = (
        np.array([2.0, 0.0, 1.0]),
        np.array([4.0, 0.0, 1.0]),
        np.array([2.0, 0.0, 5.0]),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No sorting is permitted")

    monkeypatch.setattr(np, "sort", forbidden)
    monkeypatch.setattr(np, "argsort", forbidden)
    with pytest.warns(UnsortedInputWarning):
        model = KernelSmoother(bandwidth="manual", bandwidth_value=2).fit(
            x, y, sample_weight=weights
        )
    assert model.algorithm_ == "brute"
    assert not hasattr(model, "_sort_order")
    assert not model.data_summary_["rows_sorted"]
    np.testing.assert_array_equal(model.sample_weight_, weights)
    np.testing.assert_array_equal(model.X_, x)
    np.testing.assert_array_equal(model.y_, y)
    assert model.weight_scale_ == 5
    model.predict([1.0])
    with pytest.raises(ValueError, match="already-sorted"):
        KernelSmoother(
            algorithm="neighbors", bandwidth="manual", bandwidth_value=2
        ).fit(x, y)


def test_already_sorted_search_matches_brute_with_original_weight_columns():
    x, y = np.array([0.0, 1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0, 8.0, 9.0])
    model = KernelSmoother(
        bandwidth="manual", bandwidth_value=1.2, algorithm="neighbors"
    ).fit(x, y)
    coefficients = model.get_weights([0.5, 1.5])
    brute = clone(model).set_params(algorithm="brute").fit(x, y)
    assert model.algorithm_ == "neighbors"
    np.testing.assert_allclose(coefficients, brute.get_weights([0.5, 1.5]))
    np.testing.assert_allclose(coefficients @ y, model.predict([0.5, 1.5]))


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_lossy_integer_conversion_rejected(cls):
    with pytest.raises(ValueError, match="precision"):
        cls().fit(np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64), [0, 1, 2])


@pytest.mark.parametrize(
    "cls,policy", [(Regressogram, "clip"), (KernelSmoother, "allow")]
)
def test_extrapolation_warns_and_does_not_alter_queries(cls, policy):
    model = cls(extrapolation=policy).fit([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    query = np.array([-0.1, 1.0, 2.1])
    original = query.copy()
    with pytest.warns(ExtrapolationWarning):
        result = model.predict_diagnostics(query)
    np.testing.assert_array_equal(query, original)
    np.testing.assert_array_equal(result["x"], original)
    assert result["in_training_range"].to_list() == [False, True, False]
    model.set_params(extrapolation="nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RgramWarning)
        prediction = model.predict(query)
    assert np.isnan(prediction[[0, 2]]).all()
    model.set_params(extrapolation="raise")
    with pytest.raises(ValueError, match="training range"):
        model.predict(query)


def test_cv_is_never_called_unless_requested(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected cross-validation")

    monkeypatch.setattr(CoverageSearchCV, "fit", forbidden)
    for strategy in ("dist", "width"):
        for count in (None, 2, "fd"):
            model = Regressogram(binning=strategy, n_bins=count).fit(
                [0, 1, 2, 3], [1, 3, 4, 2]
            )
            assert not model.data_summary_["cv_used"]
            assert not hasattr(model, "cv_results_")


def test_cv_respects_strict_policies_and_exposes_splits():
    x = np.arange(6.0)
    splits = [(np.array([0, 2, 4]), np.array([1, 3, 5]))]
    model = KernelSmoother(
        bandwidth="manual", bandwidth_value=0.01, unsupported="raise"
    )
    with pytest.raises(ValueError, match="no positive"):
        CoverageSearchCV(model, {}, cv=splits).fit(x, x)
    search = CoverageSearchCV(Regressogram(n_bins=1), {}, cv=splits).fit(x, x)
    np.testing.assert_array_equal(search.cv_splits_[0][0], splits[0][0])
    np.testing.assert_array_equal(search.cv_splits_[0][1], splits[0][1])


def test_weighted_custom_bootstrap_and_default_unweighted_callbacks():
    x, y = np.arange(4.0), np.array([1.0, 2.0, 4.0, 8.0])
    model = Regressogram(n_bins=1, agg=array_aggregation(np.mean)).fit(x, y)
    result = model.predict_interval([1.0], n_resamples=20, random_state=5)
    assert result["n_valid"][0] == 20
    weights = np.array([1.0, 2.0, 1.0, 3.0])
    model = Regressogram(n_bins=1).fit(x, y, sample_weight=weights)
    result = model.predict_interval([1.0], n_resamples=30, random_state=5)
    rng = np.random.default_rng(5)
    expected = []
    for _ in range(30):
        sample = rng.integers(0, 4, 4)
        expected.append(np.average(y[sample], weights=weights[sample]))
    np.testing.assert_allclose(
        [result["lower"][0], result["upper"][0]], np.quantile(expected, [0.025, 0.975])
    )


def test_diagnostic_plots_keep_observation_order():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = [2.0, 0.0, 1.0], [4.0, 0.0, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RgramWarning)
        model = Regressogram(binning="none").fit(x, y)
        diagnostics = model.regression_diagnostics(x, y)
        figure, axes = plot_diagnostics(model, x, y)
    assert diagnostics["x"].to_list() == x
    assert diagnostics["observed"].to_list() == y
    assert len(axes) == 2
    np.testing.assert_array_equal(axes[0].collections[0].get_offsets()[:, 0], x)
    plt.close(figure)


def test_cv_rejects_groups_without_group_aware_splitter():
    with pytest.raises(ValueError, match="group-aware"):
        CoverageSearchCV(Regressogram(n_bins=1), {}, cv=2).fit(
            np.arange(6.0), np.arange(6.0), groups=[0, 0, 1, 1, 2, 2]
        )


def test_cv_preserves_feature_names_in_final_refit():
    search = CoverageSearchCV(Regressogram(n_bins=1), {}, cv=2).fit(
        pl.DataFrame({"feature": np.arange(6.0)}), np.arange(6.0)
    )
    assert search.feature_names_in_.tolist() == ["feature"]
    assert search.best_estimator_.feature_names_in_.tolist() == ["feature"]
    with pytest.raises(ValueError, match="Feature name"):
        search.predict(pl.DataFrame({"wrong": [1.0]}))


def test_weight_normalization_cannot_silently_zero_positive_weights():
    with pytest.raises(ValueError, match="underflows"):
        KernelSmoother().fit([0.0, 1.0], [0.0, 1.0], sample_weight=[1e-300, 1e300])
