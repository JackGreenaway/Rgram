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


def influence() -> Figure:
    """Inspect which original training rows contribute to a selected query."""
    X = np.linspace(0, 6, 100)
    y = np.sin(X)
    # Inspect a single query: get_weights materializes a query-by-training matrix.
    query = 3.0
    observation_weights = np.ones(len(X))
    # Reweight observations on one side without changing their feature locations.
    observation_weights[X > query] = 3.0
    unweighted = _gaussian_smoother(0.6).fit(X, y)
    weighted = _gaussian_smoother(0.6).fit(X, y, sample_weight=observation_weights)
    figure, ax = _create_axes(
        "Observation weights change local influence", ylabel="Normalized kernel weight"
    )
    for model, label, color in zip(
        (unweighted, weighted),
        ("Equal observation weights", "Triple weight above query"),
        COLORS,
    ):
        # Kernel weights are nonnegative and normalized. With local-constant
        # fitting they also give the coefficients used to combine responses.
        values = model.get_weights([query], kind="kernel")[0]
        # This describes weight concentration, not confidence or degrees of freedom.
        effective_n = model.predict_diagnostics([query])["effective_n"][0]
        ax.plot(
            X,
            values,
            color=color,
            lw=3,
            label=f"{label} (effective n ≈ {effective_n:.0f})",
        )
    ax.axvline(query, color="#334155", linestyle="--", lw=2, label="Query = 3")
    _add_legend(ax)
    return figure

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    figure = influence()
    plt.show()
    plt.close(figure)
