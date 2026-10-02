"""Explicitly scoped bootstrap uncertainty for univariate regression curves."""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.base import clone

from rgram._typing import Array, Input
from rgram.base import BaseUtils
from rgram.warnings import SupportWarning


def _resample_indices(
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
    resampling: str,
    groups: Optional[ArrayLike],
    block_size: Optional[int],
) -> Iterator[Array]:
    if resampling == "groups":
        labels = np.asarray(groups)
        if labels.ndim != 1 or len(labels) != n:
            raise ValueError("groups must provide one label per training row")
        if any(
            v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
            for v in labels
        ):
            raise ValueError("groups cannot contain missing or non-finite labels")
        _, inverse = np.unique(labels, return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
        if len(members) < 2:
            raise ValueError("Group bootstrap requires at least two independent groups")
    elif resampling == "blocks":
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, (int, np.integer))
            or not 1 < block_size < n
        ):
            raise ValueError("block_size must be an integer in [2, n_samples - 1]")
    for _ in range(n_resamples):
        if resampling == "iid":
            yield rng.integers(0, n, size=n)
        elif resampling == "groups":
            chosen = rng.integers(0, len(members), size=len(members))
            yield np.concatenate([members[k] for k in chosen])
        else:
            starts = rng.integers(
                0, n - block_size + 1, size=int(np.ceil(n / block_size))
            )
            yield (starts[:, None] + np.arange(block_size)).ravel()[:n]


def bootstrap_interval(
    model: BaseUtils,
    x: Input,
    *,
    confidence_level: float,
    n_resamples: int,
    random_state: Optional[int],
    method: str,
    resampling: str,
    groups: Optional[ArrayLike],
    block_size: Optional[int],
    min_valid_fraction: float,
) -> pl.DataFrame:
    if not model.__sklearn_is_fitted__():
        raise RuntimeError("Call fit() before predict_interval")
    for name, value in (
        ("confidence_level", confidence_level),
        ("min_valid_fraction", min_valid_fraction),
    ):
        if (
            isinstance(value, (bool, str))
            or not np.isscalar(value)
            or not np.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError(f"{name} must be in (0, 1]")
    if confidence_level == 1:
        raise ValueError("confidence_level must be less than 1")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 2
    ):
        raise ValueError("n_resamples must be an integer >= 2")
    if method not in ("percentile", "basic"):
        raise ValueError("method must be 'percentile' or 'basic'")
    if resampling not in ("iid", "groups", "blocks"):
        raise ValueError("resampling must be 'iid', 'groups', or 'blocks'")
    if groups is not None and resampling != "groups":
        raise ValueError("groups requires resampling='groups'")
    if block_size is not None and resampling != "blocks":
        raise ValueError("block_size requires resampling='blocks'")
    if model.n_samples_in_ < 2:
        raise ValueError("Bootstrap intervals require at least two training rows")
    query = model._prediction_features(x)
    estimate = model.predict(query)
    rng = np.random.default_rng(random_state)
    template = clone(model)
    parameters = template.get_params()
    if "bandwidth" in parameters:
        # Inference conditional on the selected smoothing bandwidth, not a rerun
        # of bandwidth selection. Paired resampling keeps x/y/weights aligned.
        template.set_params(
            bandwidth="manual",
            bandwidth_value=model.bandwidth_,
            bandwidth_adjust=1.0,
        )
    elif (
        parameters.get("binning") in ("dist", "width")
        and parameters.get("bin_width") is None
    ):
        template.set_params(n_bins=model.n_bins_)
    draws = np.full((n_resamples, len(query)), np.nan)
    resamples = _resample_indices(
        model.n_samples_in_, n_resamples, rng, resampling, groups, block_size
    )
    for row, indices in enumerate(resamples):
        fit_kw = {}
        if getattr(model, "sample_weight_provided_", False):
            weights = model.sample_weight_[indices]
            if not (weights > 0).any():
                continue  # Counted as unsupported, never silently excluded.
            fit_kw["sample_weight"] = weights
        replica = clone(template).fit(model.X_[indices], model.y_[indices], **fit_kw)
        draws[row] = replica.predict(query)
    finite = np.isfinite(draws)
    counts = finite.sum(axis=0)
    usable = (
        counts >= max(2, int(np.ceil(min_valid_fraction * n_resamples)))
    ) & np.isfinite(estimate)
    lower = np.full(len(query), np.nan)
    upper = np.full(len(query), np.nan)
    alpha = (1 - confidence_level) / 2
    for column in np.flatnonzero(usable):
        lo, hi = np.quantile(draws[finite[:, column], column], [alpha, 1 - alpha])
        if method == "basic":
            lo, hi = 2 * estimate[column] - hi, 2 * estimate[column] - lo
        lower[column], upper[column] = lo, hi
    if not finite.all():
        model._warn(
            "Some bootstrap predictions lack support; inspect n_valid and bootstrap_coverage. "
            "Intervals below min_valid_fraction remain NaN; no query rows were dropped.",
            SupportWarning,
        )
    return pl.DataFrame(
        {
            "x": query,
            "prediction": estimate,
            "lower": lower,
            "upper": upper,
            "n_valid": counts,
            "bootstrap_coverage": counts / n_resamples,
        }
    )
