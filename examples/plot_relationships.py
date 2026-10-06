"""Plot bin resolution, bandwidth sensitivity, and separate subgroup curves.

Run from the repository root:
    python examples/plot_relationships.py --output docs/_static/relationships.png

Synthetic patterns illustrate interpretation, not estimates from real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rgram import KernelSmoother, Regressogram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Save instead of showing the figure"
    )
    args = parser.parse_args()
    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    X = np.linspace(0, 6, 120)
    y = np.sin(X) + rng.normal(0, 0.2, len(X))
    query = np.linspace(X.min(), X.max(), 250)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax in axes[:2]:
        ax.scatter(X, y, color="gray", alpha=0.35, s=12, label="Observations")
        ax.plot(
            query, np.sin(query), color="black", linestyle=":", label="Generating mean"
        )
    for count in (6, 18):
        model = Regressogram(n_bins=count).fit(X, y)
        axes[0].plot(query, model.predict(query), label=f"{count} bins")
    for bandwidth in (0.12, 0.45, 1.2):
        model = KernelSmoother(
            kernel="gaussian",
            bandwidth="manual",
            bandwidth_value=bandwidth,
        ).fit(X, y)
        axes[1].plot(query, model.predict(query), label=f"Bandwidth {bandwidth}")
    for group, offset in (("A", 0.0), ("B", 0.8)):
        response = np.sin(X) + offset + rng.normal(0, 0.2, len(X))
        model = KernelSmoother(
            kernel="gaussian",
            bandwidth="manual",
            bandwidth_value=0.45,
        ).fit(X, response)
        (line,) = axes[2].plot(query, model.predict(query), label=f"Group {group}")
        axes[2].scatter(X, response, color=line.get_color(), alpha=0.25, s=12)
    for ax, title in zip(
        axes, ("Bin resolution", "Bandwidth sensitivity", "Separate subgroup fits")
    ):
        ax.set(xlabel="Feature", ylabel="Response", title=title)
        ax.legend(fontsize=8)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
