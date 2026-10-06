"""Regression tests for data integrity and sklearn integration."""

import pickle

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_regressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from rgram import KernelSmoother, Regressogram


@pytest.mark.parametrize(
    "estimator", [Regressogram(n_bins=3), KernelSmoother(kernel="gaussian")]
)
def test_sklearn_workflows(estimator):
    x = np.linspace(0, 5, 30).reshape(-1, 1)
    y = np.sin(x[:, 0])
    assert is_regressor(estimator)
    copy = clone(estimator)
    assert copy.get_params() == estimator.get_params()
    copy.fit(x, y)
    check_is_fitted(copy)
    assert copy.n_features_in_ == 1
    assert copy.n_samples_in_ == len(x)
    assert np.isfinite(copy.score(x, y))
    np.testing.assert_allclose(
        pickle.loads(pickle.dumps(copy)).predict(x), copy.predict(x)
    )
    pipeline = make_pipeline(StandardScaler(), clone(estimator))
    name = type(estimator).__name__.lower()
    params = (
        {f"{name}__n_bins": [2, 3]}
        if name == "regressogram"
        else {f"{name}__bandwidth_adjust": [0.5, 1.0]}
    )
    search = GridSearchCV(
        pipeline, params, cv=3, scoring="neg_mean_squared_error", error_score="raise"
    )
    search.fit(x, y)
    assert search.predict(x).shape == (len(x),)


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, None])
def test_reject_bad_training_rows(cls, bad):
    for column in ("a", "b"):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}).with_columns(
            pl.Series(column, [1.0, bad, 3.0])
        )
        with pytest.raises(ValueError, match="no rows were dropped"):
            cls().fit("a", "b", data=df.lazy())


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_same_column_and_duplicate_order(cls):
    df = pl.DataFrame({"a": [3.0, 1.0, 2.0, 1.0]})
    model = cls().fit("a", "a", data=df)
    assert model.n_samples_in_ == 4
    points = [3.0, 1.0, 2.0, 1.0]
    batch = model.predict(points)
    single = np.array([model.predict([x])[0] for x in points])
    np.testing.assert_allclose(batch, single)


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_multivariate_rejected_and_failed_refit_invalidates(cls):
    model = cls().fit([1.0, 2.0, 3.0], [3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="univariate"):
        model.fit(np.ones((3, 2)), [1.0, 2.0, 3.0])
    with pytest.raises(NotFittedError):
        model.predict([1.0])
    with pytest.raises(ValueError, match="univariate"):
        model.fit(["a", "b"], "a", data=pl.DataFrame({"a": [1.0], "b": [2.0]}))


def test_quantile_known_bins_and_batch_independence():
    model = Regressogram(n_bins=2, ci=None).fit(
        [0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 10.0, 12.0]
    )
    np.testing.assert_allclose(
        model.predict([3.0, 0.0, 2.0, 0.0]), [11.0, 1.0, 11.0, 1.0]
    )
    np.testing.assert_allclose(model.predict([2.0]), [11.0])
    assert model.bins_["n_samples"].sum() == 4
    np.testing.assert_allclose(model.bin_edges_, [1.5])


def test_duplicate_quantile_edges_preserve_maximum_bin():
    x = [0.0] * 4 + [1.0] * 4
    model = Regressogram(n_bins=20, ci=None).fit(x, x)
    np.testing.assert_allclose(model.predict([0.0, 1.0, 2.0]), [0.0, 1.0, 1.0])
    assert model.bins_["n_samples"].sum() == len(x)


@pytest.mark.parametrize("kernel", sorted(KernelSmoother._KERNELS))
def test_kernel_matches_independent_reference(kernel):
    x = np.array([-2.0, 0.0, 1.0, 3.0])
    y = np.array([1.0, 5.0, -1.0, 2.0])
    query = np.array([1.0, -0.5, 1.0])
    u = (x[None, :] - query[:, None]) / 2
    a = np.abs(u)
    formulas = {
        "gaussian": lambda: np.exp(-0.5 * u**2),
        "logistic": lambda: 1 / (np.exp(u) + 2 + np.exp(-u)),
        "uniform": lambda: (a <= 1).astype(float),
        "triangular": lambda: np.maximum(1 - a, 0),
        "cosine": lambda: np.where(a <= 1, np.cos(np.pi / 2 * u), 0),
        "epanechnikov": lambda: np.maximum(1 - u**2, 0),
        "biweight": lambda: np.maximum(1 - u**2, 0) ** 2,
        "tricube": lambda: np.maximum(1 - a**3, 0) ** 3,
    }
    weights = formulas[kernel]() * [1, 2, 0, 1]
    weights /= weights.sum(axis=1, keepdims=True)
    model = KernelSmoother(
        kernel=kernel, bandwidth="manual", bandwidth_value=2, batch_size=1
    )
    model.fit(x, y, sample_weight=[1, 2, 0, 1])
    np.testing.assert_allclose(model.get_weights(query), weights, atol=1e-15)
    np.testing.assert_allclose(model.predict(query), weights @ y)
    diag = model.predict_diagnostics(query)
    np.testing.assert_allclose(diag["effective_n"], 1 / (weights**2).sum(axis=1))
    model.set_params(batch_size=100)
    np.testing.assert_allclose(model.predict(query), weights @ y)


def test_snapshot_and_grid():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([2.0, 4.0, 6.0])
    model = KernelSmoother(kernel="gaussian", n_eval_samples=7).fit(x, y)
    expected = model.predict([1.0])
    x[:] = 99
    y[:] = 99
    np.testing.assert_allclose(model.predict([1.0]), expected)
    assert model.predict_grid().height == 7


def test_no_support_policy_and_constant_training():
    model = KernelSmoother(bandwidth="manual", bandwidth_value=1).fit(
        [0.0, 0.0], [1.0, 3.0]
    )
    with pytest.warns(UserWarning, match="without dropping rows"):
        np.testing.assert_allclose(
            model.predict([0.0, 5.0, 0.0]), [2.0, np.nan, 2.0], equal_nan=True
        )
    model.set_params(unsupported="raise")
    with pytest.raises(ValueError, match="no positive kernel support"):
        model.predict([5.0])
    with pytest.raises(ValueError, match="positive bandwidth"):
        KernelSmoother().fit([0.0, 0.0], [1.0, 3.0])


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_bandwidth(value):
    with pytest.raises(ValueError, match="finite positive"):
        KernelSmoother(bandwidth="manual", bandwidth_value=value).fit(
            [0.0, 1.0], [0.0, 1.0]
        )


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [1], [np.nan, 1]])
def test_invalid_observation_weights(weights):
    with pytest.raises(ValueError):
        KernelSmoother().fit([0.0, 1.0], [0.0, 1.0], sample_weight=weights)


def test_invalid_custom_kernel():
    model = KernelSmoother(kernel=lambda u: u * 0 - 1).fit([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="nonnegative"):
        model.predict([0.0])


def test_gaussian_far_query_does_not_underflow():
    model = KernelSmoother(
        kernel="gaussian", bandwidth="manual", bandwidth_value=1
    ).fit([0.0, 1.0], [3.0, 7.0])
    np.testing.assert_allclose(model.predict([1000.0]), [7.0])


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_standard_keywords_and_legacy_aliases(cls):
    X = np.arange(12.0).reshape(-1, 1)
    y = np.sin(X[:, 0])
    model = cls().fit(X=X, y=y)
    np.testing.assert_allclose(model.predict(X=X), model.predict(X))
    np.testing.assert_allclose(cls().fit_predict(X=X, y=y), model.predict(X))
    legacy = cls().fit(x=X, y=y)
    query_kw = {"x": X} if cls is Regressogram else {"x_eval": X}
    np.testing.assert_allclose(legacy.predict(**query_kw), model.predict(X))
    with pytest.raises(TypeError, match="either X or"):
        model.fit(X=X, x=X, y=y)
    with pytest.raises(TypeError, match="either X or"):
        model.predict(X=X, **query_kw)
    with pytest.raises(NotFittedError):
        cls().predict(X=X)


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_feature_name_contract_and_refit(cls):
    X = pl.DataFrame({"temperature": np.arange(12.0)})
    y = np.arange(12.0)
    model = cls().fit(X=X, y=y)
    np.testing.assert_array_equal(model.feature_names_in_, ["temperature"])
    with pytest.raises(ValueError, match="does not match"):
        model.predict(X.rename({"temperature": "humidity"}))
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        model.predict(X.to_numpy())
    model.fit(X.to_numpy(), y)
    assert not hasattr(model, "feature_names_in_")
    with pytest.warns(UserWarning, match="fitted without feature names"):
        model.predict(X)


def test_coverage_search_failed_fit_is_not_fitted():
    from rgram import CoverageSearchCV

    X = np.arange(12.0).reshape(-1, 1)
    search = CoverageSearchCV(Regressogram(), {"n_bins": [1, 2]}, cv=3)
    search.fit(X=X, y=X[:, 0])
    check_is_fitted(search)
    assert search.predict(X=X).shape == (12,)
    search.set_params(
        estimator=KernelSmoother(bandwidth="manual", bandwidth_value=1e-6)
    )
    with pytest.raises(ValueError, match="No candidate"):
        search.set_params(param_grid={}).fit(X, X[:, 0])
    assert hasattr(search, "cv_results_")
    with pytest.raises(NotFittedError):
        check_is_fitted(search)
    with pytest.raises(NotFittedError):
        search.predict(X)


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_dataframe_selection_pipeline_weights_and_target_transform(cls):
    import pandas as pd
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline

    X = pd.DataFrame({"temperature": np.arange(20.0), "humidity": np.ones(20)})
    y = 2 + X["temperature"].to_numpy()
    weights = np.arange(1.0, 21.0)
    pipeline = Pipeline(
        [
            (
                "select",
                ColumnTransformer(
                    [
                        ("temperature", StandardScaler(), ["temperature"]),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", cls()),
        ]
    )
    pipeline.fit(X, y, model__sample_weight=weights)
    np.testing.assert_array_equal(pipeline[-1].sample_weight_, weights)
    prediction = pipeline.predict(X)
    assert prediction.shape == (20,)
    assert pipeline.score(X, y, sample_weight=weights) == pytest.approx(
        r2_score(y, prediction, sample_weight=weights)
    )
    wrapped = TransformedTargetRegressor(
        regressor=clone(pipeline),
        func=np.log,
        inverse_func=np.exp,
    ).fit(X, y)
    assert np.isfinite(wrapped.predict(X)).all()


@pytest.mark.parametrize("cls", [Regressogram, KernelSmoother])
def test_sklearn_partial_dependence(cls):
    from sklearn.inspection import partial_dependence

    X = np.linspace(0, 5, 30).reshape(-1, 1)
    model = cls().fit(X, np.sin(X[:, 0]))
    result = partial_dependence(model, X, [0], method="brute", grid_resolution=8)
    np.testing.assert_allclose(
        result["average"][0], model.predict(result["grid_values"][0].reshape(-1, 1))
    )
