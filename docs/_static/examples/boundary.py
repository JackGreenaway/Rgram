from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np

from rgram import KernelSmoother, Regressogram, SupportWarning, quantile

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# A consistent palette makes comparisons readable across the gallery.
COLORS = ["#1764ab", "#de682b", "#21856b", "#8855a2"]


def _create_axes(
    title: str, xlabel: str = "Feature", ylabel: str = "Response"
) -> tuple[Figure, Axes]:
    """Create a readable, single-panel figure with consistent gallery styling."""
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 20}):
        figure, ax = plt.subplots(figsize=(10, 6.6), layout="constrained")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_facecolor("#f8fafc")
    ax.grid(color="#dce3ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    return figure, ax


def _plot_observations(ax: Axes, X: np.ndarray, y: np.ndarray) -> None:
    """Show original rows without connecting them or hiding response spread."""
    ax.scatter(
        X,
        y,
        color="#64748b",
        alpha=0.4,
        s=28,
        edgecolors="none",
        label="Observed responses",
    )


def _add_legend(ax: Axes) -> None:
    """Identify every plotted comparison using the same readable legend style."""
    ax.legend(fontsize=12, framealpha=0.96, edgecolor="#dce3ec", loc="best")


def _gaussian_smoother(bandwidth: float, **kwargs) -> KernelSmoother:
    """Keep kernel and bandwidth units explicit; kwargs select other model options."""
    return KernelSmoother(
        kernel="gaussian", bandwidth="manual", bandwidth_value=bandwidth, **kwargs
    )


def boundary() -> Figure:
    """Show why a local slope can help near the observed feature boundaries."""
    rng = np.random.default_rng(8)
    X = np.linspace(0, 1, 120)
    # A linear generating mean makes edge bias easier to distinguish from curvature.
    y = 1 + 2 * X + rng.normal(0, 0.1, len(X))
    query = np.linspace(0, 1, 250)
    figure, ax = _create_axes("Local-linear regression near a boundary")
    _plot_observations(ax, X, y)
    ax.plot(query, 1 + 2 * query, "--", color="#334155", lw=2, label="Generating mean")
    # Change only the local fitting order, keeping kernel and bandwidth equal.
    for regression, color in zip(("local_constant", "local_linear"), COLORS):
        model = _gaussian_smoother(0.22, regression=regression).fit(X, y)
        ax.plot(
            query,
            model.predict(query),
            color=color,
            lw=3,
            label=regression.replace("_", " ").capitalize(),
        )
    # These bands mark regions with asymmetric neighborhoods, not confidence bounds.
    ax.axvspan(0, 0.12, color="#dce3ec", alpha=0.55)
    ax.axvspan(0.88, 1, color="#dce3ec", alpha=0.55)
    _add_legend(ax)
    return figure

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    figure = boundary()
    plt.show()
    plt.close(figure)
