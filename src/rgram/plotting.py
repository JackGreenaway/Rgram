"""Optional residual and observed-versus-predicted diagnostic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from rgram._typing import Input
from rgram.base import BaseUtils

if TYPE_CHECKING:
    from matplotlib.figure import Figure


import numpy as np


def plot_diagnostics(
    estimator: BaseUtils, X: Input, y: Input
) -> tuple[Figure, NDArray[Any]]:
    """Plot residuals and observed-versus-predicted values without sorting rows.

    Parameters
    ----------
    estimator : Regressogram or KernelSmoother
        Successfully fitted model providing regression_diagnostics. For a
        pipeline, use the final estimator with transformed feature coordinates.
    X : array-like of shape (n_samples,) or (n_samples, 1)
        Evaluation feature values, aligned with y.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Observed responses, for training or held-out observations.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing both diagnostic panels and a support-count title.
    axes : ndarray of shape (2,)
        Residual-versus-feature axis and observed-versus-predicted axis.

    Raises
    ------
    ImportError
        If Matplotlib is absent; install rgram[plot].
    sklearn.exceptions.NotFittedError
        If the estimator has no successful fit.
    ValueError
        If evaluation pairs are invalid or strict prediction policies raise.

    Notes
    -----
    Uses scatter points, never connecting or sorting observation rows.
    Unsupported rows remain in the diagnostic table and are counted in the
    title, though non-finite points cannot be drawn. No data is modified.
    The caller chooses whether to display, save, or close the returned figure.

    Examples
    --------
    >>> from rgram import Regressogram, plot_diagnostics
    >>> model = Regressogram(n_bins=2).fit([0., 1., 2.], [1., 3., 4.])
    >>> figure, axes = plot_diagnostics(model, [0., 1., 2.], [1., 3., 4.])
    >>> len(axes)
    2
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_diagnostics requires matplotlib; install rgram[plot]"
        ) from exc
    diagnostics = estimator.regression_diagnostics(X, y)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    predicted = diagnostics["prediction"].to_numpy()
    observed = diagnostics["observed"].to_numpy()
    axes[0].scatter(
        diagnostics["x"].to_numpy(), diagnostics["residual"].to_numpy(), alpha=0.6
    )
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].set(
        xlabel="x (original rows)", ylabel="Observed − predicted", title="Residuals"
    )
    axes[1].scatter(predicted, observed, alpha=0.6)
    finite_values = np.r_[observed, predicted[np.isfinite(predicted)]]
    low, high = finite_values.min(), finite_values.max()
    axes[1].plot([low, high], [low, high], color="gray", linewidth=1)
    axes[1].set(
        xlabel="Predicted", ylabel="Observed", title="Observed versus predicted"
    )
    count = diagnostics["supported"].sum()
    figure.suptitle(
        f"{count}/{len(observed)} rows have finite predictions; {len(observed) - count} cannot be drawn"
    )
    return figure, axes
