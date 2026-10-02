"""Property-based row conservation and permutation checks."""

import warnings

import numpy as np
from hypothesis import given, settings, strategies as st

from rgram import KernelSmoother, Regressogram, RgramWarning


@settings(max_examples=35, deadline=None, database=None, derandomize=True)
@given(
    st.lists(
        st.tuples(st.integers(-10, 10), st.integers(-100, 100), st.integers(1, 5)),
        min_size=2,
        max_size=35,
    )
)
def test_all_rows_and_weights_survive_permutation(rows):
    x, y, weights = np.asarray(rows, dtype=float).T
    reverse = np.arange(len(x) - 1, -1, -1)
    models = [
        Regressogram(n_bins=min(3, len(x))),
        KernelSmoother(kernel="gaussian", bandwidth="manual", bandwidth_value=2),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RgramWarning)
        for model in models:
            model.fit(x, y, sample_weight=weights)
            prediction = model.predict(x)
            permuted = model.predict(x[reverse])
            np.testing.assert_allclose(
                permuted, prediction[reverse], rtol=1e-12, atol=1e-12
            )
            np.testing.assert_array_equal(model.X_, x)
            np.testing.assert_array_equal(model.y_, y)
            np.testing.assert_array_equal(model.sample_weight_, weights)
            assert len(prediction) == len(x)
            assert model.data_summary_["rows_dropped"] == 0
            assert model.n_samples_in_ == len(x)
            if isinstance(model, Regressogram):
                assert model.bins_["n_samples"].sum() == len(x)
                assert model.bins_["weight_sum"].sum() == weights.sum()
