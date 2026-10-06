"""Numerical references and integrity contracts for tuning and inference."""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from rgram import (
    BinningWarning,
    CoverageSearchCV,
    KernelSmoother,
    NumericalWarning,
    Regressogram,
    RgramWarning,
    SupportWarning,
    UnsortedInputWarning,
)


def smoother(**kwargs):
    return KernelSmoother(bandwidth="manual", bandwidth_value=0.75, **kwargs)


@pytest.mark.parametrize("binning", ["dist", "width"])
@pytest.mark.parametrize(
    "rule", [None, "auto", "fd", "scott", "sturges", "sqrt", "rice", 1, 7]
)
def test_bin_rules_preserve_rows_and_permutation(binning, rule):
    rng = np.random.default_rng(20)
    x = rng.normal(size=80)
    y = np.sin(x)
    model = Regressogram(binning=binning, n_bins=rule).fit(x, y)
    assert model.bins_["n_samples"].sum() == len(x)
    order = rng.permutation(len(x))
    np.testing.assert_allclose(model.predict(x)[order], model.predict(x[order]))
    assert np.isfinite(model.predict(x)).all()
    assert model.n_bins_ == len(model.bin_edges_) + 1


@pytest.mark.parametrize("rule", ["auto", "fd", "scott", "sturges", "sqrt", "rice"])
def test_width_rules_match_numpy_reference(rule):
    x = np.random.default_rng(2).normal(size=300)
    model = Regressogram(binning="width", n_bins=rule).fit(x, x)
    expected = np.histogram_bin_edges(x, bins=rule)
    assert model.n_bins_ == len(expected) - 1
    np.testing.assert_allclose(model.bin_edges_, expected[1:-1])


def test_quantile_auto_not_driven_by_outlier_range():
    x = np.linspace(0, 1, 1000)
    y = x.copy()
    before = Regressogram().fit(x, y)
    x[-1] = 1e12
    after = Regressogram().fit(x, y)
    assert before.n_bins_ == after.n_bins_ == 10
    assert after.bins_["n_samples"].sum() == 1000


def test_zero_iqr_fallback_and_cap():
    x = np.r_[np.zeros(90), np.arange(10.0)]
    model = Regressogram(binning="width", n_bins="fd").fit(x, x)
    assert model.n_bins_ == int(np.ceil(np.log2(len(x)) + 1))
    x = np.r_[np.linspace(0, 1, 100), 1e12]
    with pytest.warns(BinningWarning, match="capped"):
        model = Regressogram(binning="width", n_bins="fd", max_bins=10).fit(x, x)
    assert model.n_bins_ == 10
    assert model.bins_["n_samples"].sum() == len(x)


def test_explicit_width_and_count_boundaries():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    model = Regressogram(binning="width", n_bins=2).fit(x, x)
    np.testing.assert_allclose(model.predict(x), [0.5, 0.5, 3, 3, 3])
    assert model.bin_width_ == 2
    model = Regressogram(binning="width", bin_width=1.5).fit(x, x)
    np.testing.assert_allclose(model.bin_edges_, [1.5, 3])
    assert model.n_bins_ == 3
    assert model.bin_width_ == 1.5
    with pytest.raises(ValueError, match="max_bins"):
        Regressogram(binning="width", bin_width=1e-200).fit(x, x)
    with pytest.raises(ValueError, match="requires"):
        Regressogram(binning="width", n_bins=3, bin_width=1).fit(x, x)


@pytest.mark.parametrize("binning", ["dist", "width"])
def test_cv_selects_bins_for_y_and_exposes_results(binning):
    x = np.linspace(0, 1, 120)
    y = np.where(x < 0.5, 0.0, 10.0)
    model = Regressogram(
        binning=binning, n_bins="cv", bin_candidates=[1, 2, 4], random_state=7
    ).fit(x, y)
    assert model.selected_n_bins_ != 1
    assert model.cv_results_["eligible"].any()
    assert np.mean((model.predict(x) - y) ** 2) < 1
    assert model.n_samples_in_ == len(x)
    assert clone(model).get_params()["n_bins"] == "cv"


@pytest.mark.parametrize("kernel", sorted(KernelSmoother._COMPACT_KERNELS))
@pytest.mark.parametrize("regression", ["local_constant", "local_linear"])
def test_neighbor_search_matches_brute_on_shuffled_weighted_duplicates(
    kernel, regression
):
    rng = np.random.default_rng(12)
    x = np.r_[rng.normal(size=60), 0.0, 0.0, -0.75, 0.75]
    y = rng.normal(size=len(x))
    weights = rng.uniform(size=len(x))
    weights[::4] = 0
    query = np.r_[rng.normal(size=15), 0.0, 10.0, 0.0]
    model = smoother(kernel=kernel, regression=regression).fit(
        x, y, sample_weight=weights
    )
    fast = model.predict(query)
    coefficients = model.get_weights(query)
    slow = clone(model).set_params(algorithm="brute").fit(x, y, sample_weight=weights)
    np.testing.assert_allclose(fast, slow.predict(query), atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(coefficients, slow.get_weights(query), atol=1e-12)
    valid = np.isfinite(fast)
    np.testing.assert_allclose(coefficients[valid] @ y, fast[valid], atol=1e-12)
    assert np.all(model.get_weights(query, kind="kernel") >= 0)


@pytest.mark.parametrize("kernel", sorted(KernelSmoother._KERNELS))
def test_local_linear_matches_weighted_lstsq_at_boundary(kernel):
    x = np.linspace(0, 1, 12)
    y = np.cos(x) + x**2
    model = smoother(kernel=kernel, regression="local_linear").fit(x, y)
    q = -0.1
    w = model.get_weights([q], kind="kernel")[0]
    design = np.column_stack([np.ones(len(x)), x - q])
    reference = np.linalg.lstsq(
        design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None
    )[0][0]
    np.testing.assert_allclose(model.predict([q]), [reference], atol=1e-12)
    assert not model.predict_diagnostics([q])["local_linear_fallback"][0]


def test_local_linear_reproduces_line_and_singular_policy():
    x = np.linspace(0, 1, 30)
    y = 3 + 2 * x
    model = smoother(regression="local_linear", kernel="tricube").fit(x, y)
    np.testing.assert_allclose(
        model.predict([-0.1, 0, 1, 1.1]), [2.8, 3, 5, 5.2], atol=1e-12
    )
    constant = smoother(regression="local_linear").fit([0, 0], [1, 3])
    with pytest.warns(NumericalWarning, match="fallback"):
        diag = constant.predict_diagnostics([0])
    assert diag["prediction"][0] == 2
    assert diag["local_linear_fallback"][0]
    constant.set_params(singular="raise")
    with pytest.raises(ValueError, match="Singular"):
        constant.predict([0])


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_order_warning_default_and_standard_filter(cls):
    with pytest.warns(UnsortedInputWarning, match="No rows were sorted"):
        model = cls().fit([2.0, 0.0, 1.0], [4.0, 0.0, 1.0])
    with pytest.warns(UnsortedInputWarning):
        model.predict([2.0, 0.0, 1.0])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("ignore", RgramWarning)
        model.predict([2.0, 0.0, 1.0])
    assert not captured
    assert "warn" not in model.get_params()
    assert "warn_unsorted" not in model.get_params()


def test_warning_filter_never_disables_errors_or_masks_nan():
    model = smoother().fit([0.0, 1.0], [0.0, 1.0])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("ignore", RgramWarning)
        assert np.isnan(model.predict([10.0])[0])
        with pytest.raises(ValueError):
            model.predict([np.nan])
        model.set_params(unsupported="raise")
        with pytest.raises(ValueError, match="no positive"):
            model.predict([10.0])
    assert not captured
    assert issubclass(SupportWarning, RgramWarning)


def test_cv_rejects_selective_candidate_and_reports_all_rows():
    x = np.linspace(0, 1, 21)
    y = x.copy()
    cv = [(np.arange(0, 21, 2), np.arange(1, 21, 2))]
    search = CoverageSearchCV(
        KernelSmoother(bandwidth="manual"),
        {"bandwidth_value": [0.001, 0.3]},
        cv=cv,
    ).fit(x, y)
    assert search.best_params_ == {"bandwidth_value": 0.3}
    assert search.cv_results_["coverage"].tolist() == [0, 1]
    assert search.cv_results_["eligible"].tolist() == [False, True]
    assert search.cv_results_["n_validation"].tolist() == [10, 10]
    assert np.isinf(search.cv_results_["selection_loss"][0])
    np.testing.assert_allclose(search.predict(x), search.best_estimator_.predict(x))
    with pytest.raises(ValueError, match="No candidate"):
        CoverageSearchCV(
            KernelSmoother(bandwidth="manual"), {"bandwidth_value": [0.001]}, cv=cv
        ).fit(x, y)


def test_cv_groups_time_splits_and_overlap_guard():
    x = np.linspace(0, 2, 24)
    y = x**2
    for splitter in (GroupKFold(3), TimeSeriesSplit(3)):
        search = CoverageSearchCV(Regressogram(), {"n_bins": [1, 2]}, cv=splitter).fit(
            x,
            y,
            groups=np.repeat(np.arange(6), 4)
            if isinstance(splitter, GroupKFold)
            else None,
        )
        assert search.n_splits_ == 3
    with pytest.raises(ValueError, match="overlap"):
        CoverageSearchCV(Regressogram(), {}, cv=[([0, 1], [1, 2])]).fit(x, y)


@pytest.mark.parametrize("method", ["percentile", "basic"])
@pytest.mark.parametrize("cls", ["kernel", "regressogram"])
def test_bootstrap_intervals_match_independent_sample_mean_reference(method, cls):
    x = np.arange(5.0)
    y = np.array([1.0, 2.0, 4.0, 5.0, 10.0])
    model = (
        KernelSmoother(kernel="uniform", bandwidth="manual", bandwidth_value=100)
        if cls == "kernel"
        else Regressogram(n_bins=1)
    )
    model.fit(x, y)
    result = model.predict_interval(
        [2.0, 2.0], n_resamples=80, random_state=17, method=method
    )
    rng = np.random.default_rng(17)
    means = np.array([y[rng.integers(0, 5, size=5)].mean() for _ in range(80)])
    lower, upper = np.quantile(means, [0.025, 0.975])
    if method == "basic":
        lower, upper = 2 * y.mean() - upper, 2 * y.mean() - lower
    np.testing.assert_allclose(result["lower"], lower)
    np.testing.assert_allclose(result["upper"], upper)
    assert result["n_valid"].to_list() == [80, 80]
    again = model.predict_interval(
        [2.0, 2.0], n_resamples=80, random_state=17, method=method
    )
    assert result.equals(again)


def test_bootstrap_support_is_not_silently_filtered():
    model = KernelSmoother(bandwidth="manual", bandwidth_value=0.1).fit(
        [0.0, 1.0], [0.0, 1.0]
    )
    with pytest.warns(SupportWarning, match="bootstrap"):
        result = model.predict_interval([0.0], n_resamples=60, random_state=5)
    assert 0 < result["n_valid"][0] < 60
    assert np.isnan(result["lower"][0])
    with pytest.warns(SupportWarning):
        partial = model.predict_interval(
            [0.0], n_resamples=60, random_state=5, min_valid_fraction=0.5
        )
    assert partial["lower"][0] == 0


@pytest.mark.parametrize("resampling", ["groups", "blocks"])
def test_dependent_resampling_matches_reference(resampling):
    x = np.arange(6.0)
    y = np.array([0.0, 2.0, 5.0, 8.0, 9.0, 12.0])
    model = Regressogram(n_bins=1).fit(x, y)
    options = (
        {"groups": [0, 0, 1, 1, 2, 2]} if resampling == "groups" else {"block_size": 2}
    )
    result = model.predict_interval(
        [2.0], resampling=resampling, n_resamples=50, random_state=4, **options
    )
    rng = np.random.default_rng(4)
    means = []
    for _ in range(50):
        if resampling == "groups":
            group_ids = rng.integers(0, 3, size=3)
            indices = (2 * group_ids[:, None] + np.arange(2)).ravel()
        else:
            starts = rng.integers(0, 5, size=3)
            indices = (starts[:, None] + np.arange(2)).ravel()
        means.append(y[indices].mean())
    expected = np.quantile(means, [0.025, 0.975])
    np.testing.assert_allclose([result["lower"][0], result["upper"][0]], expected)


def test_kernel_interval_and_validation():
    model = smoother(kernel="gaussian").fit([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])
    interval = model.predict_interval([1.0], n_resamples=20, random_state=1)
    pred, lo, hi = interval.select("prediction", "lower", "upper").to_numpy().T
    np.testing.assert_allclose(pred, model.predict([1.0]))
    assert np.isfinite(lo).all() and (hi >= lo).all()
    for options in (
        {"n_resamples": 1},
        {"confidence_level": 1},
        {"min_valid_fraction": 0},
        {"resampling": "groups"},
        {"resampling": "blocks", "block_size": 3},
        {"groups": [0, 1, 2]},
    ):
        with pytest.raises(ValueError):
            model.predict_interval([1.0], **options)


def test_compact_path_does_not_construct_full_pair_matrix(monkeypatch):
    x = np.linspace(0, 100, 10000)
    model = KernelSmoother(
        kernel="tricube", bandwidth="manual", bandwidth_value=0.05
    ).fit(x, np.sin(x))
    shapes = []
    original = model._normalized_kernel

    def record(u, weights):
        shapes.append(u.shape)
        return original(u, weights)

    monkeypatch.setattr(model, "_normalized_kernel", record)
    query = np.linspace(1, 99, 50)
    assert np.isfinite(model.predict(query)).all()
    assert len(shapes) == len(query)
    assert max(columns for _, columns in shapes) <= 11


def test_cv_requires_coverage_in_each_fold_not_only_pooled():
    x = np.arange(10.0)
    cv = [(np.arange(0, 5), np.arange(5, 10)), (np.arange(0, 9), np.array([9]))]
    search = CoverageSearchCV(
        KernelSmoother(kernel="uniform", bandwidth="manual"),
        {"bandwidth_value": [1.1]},
        cv=cv,
        min_coverage=0.3,
    )
    with pytest.raises(ValueError, match="No candidate"):
        search.fit(x, x)
    assert search.cv_results_["coverage"][0] >= 0.3
    assert search.cv_results_["min_fold_coverage"][0] < 0.3


def test_bootstrap_observation_weights_stay_paired():
    x, y = np.arange(4.0), np.array([1.0, 3.0, 7.0, 9.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    model = KernelSmoother(
        kernel="uniform", bandwidth="manual", bandwidth_value=100
    ).fit(x, y, sample_weight=weights)
    result = model.predict_interval([1.0], n_resamples=40, random_state=3)
    rng = np.random.default_rng(3)
    means = []
    for _ in range(40):
        indices = rng.integers(0, 4, size=4)
        means.append(np.average(y[indices], weights=weights[indices]))
    expected = np.quantile(means, [0.025, 0.975])
    np.testing.assert_allclose([result["lower"][0], result["upper"][0]], expected)
