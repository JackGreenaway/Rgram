"""Generate the documentation gallery, with one large figure per example.

Run one interactive example:
    python examples/plot_relationships.py --example bandwidth
Save every figure without a desktop:
    python examples/plot_relationships.py --output-dir docs/_static/gallery
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np

from rgram import KernelSmoother, Regressogram, SupportWarning, quantile

COLORS = ["#1764ab", "#de682b", "#21856b", "#8855a2"]


def _axes(title, xlabel="Feature", ylabel="Response"):
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


def _observations(ax, X, y):
    ax.scatter(
        X,
        y,
        color="#64748b",
        alpha=0.4,
        s=28,
        edgecolors="none",
        label="Observed responses",
    )


def _legend(ax):
    ax.legend(fontsize=12, framealpha=0.96, edgecolor="#dce3ec", loc="best")


def _smoother(bandwidth, **kwargs):
    return KernelSmoother(
        kernel="gaussian", bandwidth="manual", bandwidth_value=bandwidth, **kwargs
    )


def bins():
    """Compare coarse and fine response summaries on identical observations."""
    rng = np.random.default_rng(42)
    X = np.linspace(0, 6, 150)
    y = np.sin(X) + rng.normal(0, 0.2, len(X))
    query = np.linspace(0, 6, 350)
    figure, ax = _axes("How bin count changes a response summary")
    _observations(ax, X, y)
    ax.plot(query, np.sin(query), "--", color="#334155", lw=2, label="Generating mean")
    for count, color in zip((6, 18), COLORS):
        model = Regressogram(n_bins=count).fit(X, y)
        ax.step(
            query,
            model.predict(query),
            where="mid",
            lw=2.8,
            color=color,
            label=f"{count} quantile bins",
        )
    _legend(ax)
    return figure


def bandwidth():
    """Compare three smoothing scales, with the generating curve visible."""
    rng = np.random.default_rng(42)
    X = np.linspace(0, 6, 150)
    y = np.sin(X) + rng.normal(0, 0.2, len(X))
    query = np.linspace(0, 6, 350)
    figure, ax = _axes("Bandwidth controls detail and stability")
    _observations(ax, X, y)
    ax.plot(query, np.sin(query), "--", color="#334155", lw=2, label="Generating mean")
    for value, color in zip((0.12, 0.45, 1.2), COLORS):
        model = _smoother(value).fit(X, y)
        ax.plot(
            query, model.predict(query), lw=2.8, color=color, label=f"Bandwidth {value}"
        )
    _legend(ax)
    return figure


def subgroups():
    """Fit explicitly selected groups with a common bandwidth and query range."""
    rng = np.random.default_rng(12)
    X = np.linspace(0, 6, 120)
    query = np.linspace(0, 6, 350)
    figure, ax = _axes("Compare observed relationships across groups")
    for group, offset, color in zip(("A", "B"), (0.0, 0.8), COLORS):
        y = np.sin(X) + offset + rng.normal(0, 0.2, len(X))
        model = _smoother(0.4).fit(X, y)
        ax.scatter(X, y, color=color, alpha=0.23, s=28, edgecolors="none")
        ax.plot(query, model.predict(query), color=color, lw=3, label=f"Group {group}")
    _legend(ax)
    return figure


def boundary():
    """Show why a local slope can help near the observed feature boundaries."""
    rng = np.random.default_rng(8)
    X = np.linspace(0, 1, 120)
    y = 1 + 2 * X + rng.normal(0, 0.1, len(X))
    query = np.linspace(0, 1, 250)
    figure, ax = _axes("Local-linear regression near a boundary")
    _observations(ax, X, y)
    ax.plot(query, 1 + 2 * query, "--", color="#334155", lw=2, label="Generating mean")
    for regression, color in zip(("local_constant", "local_linear"), COLORS):
        model = _smoother(0.22, regression=regression).fit(X, y)
        ax.plot(
            query,
            model.predict(query),
            color=color,
            lw=3,
            label=regression.replace("_", " ").capitalize(),
        )
    ax.axvspan(0, 0.12, color="#dce3ec", alpha=0.55)
    ax.axvspan(0.88, 1, color="#dce3ec", alpha=0.55)
    _legend(ax)
    return figure


def summaries():
    """Compare a bin mean, median, and descriptive quantile envelope."""
    rng = np.random.default_rng(17)
    X = np.linspace(0, 6, 180)
    y = 0.4 * X + rng.normal(0, 0.22, len(X))
    y[::12] += 2.8  # Deliberate extreme responses, not removed during fitting.
    query = np.linspace(0, 6, 350)
    mean = Regressogram(n_bins=9).fit(X, y)
    median = Regressogram(
        n_bins=9, agg="median", ci=(quantile(0.1), quantile(0.9))
    ).fit(X, y)
    diagnostic = median.predict_diagnostics(query)
    figure, ax = _axes("Mean, median, and within-bin response spread")
    _observations(ax, X, y)
    ax.fill_between(
        query,
        diagnostic["summary_lower"].to_numpy(),
        diagnostic["summary_upper"].to_numpy(),
        step="mid",
        color=COLORS[1],
        alpha=0.15,
        label="10th–90th response percentiles",
    )
    ax.step(
        query,
        mean.predict(query),
        where="mid",
        color=COLORS[0],
        lw=2.8,
        label="Bin mean",
    )
    ax.step(
        query,
        median.predict(query),
        where="mid",
        color=COLORS[1],
        lw=2.8,
        label="Bin median",
    )
    _legend(ax)
    return figure


def influence():
    """Inspect which original training rows contribute to a selected query."""
    X = np.linspace(0, 6, 100)
    y = np.sin(X)
    query = 3.0
    observation_weights = np.ones(len(X))
    observation_weights[X > query] = 3.0
    unweighted = _smoother(0.6).fit(X, y)
    weighted = _smoother(0.6).fit(X, y, sample_weight=observation_weights)
    figure, ax = _axes(
        "Observation weights change local influence", ylabel="Normalized kernel weight"
    )
    for model, label, color in zip(
        (unweighted, weighted),
        ("Equal observation weights", "Triple weight above query"),
        COLORS,
    ):
        values = model.get_weights([query], kind="kernel")[0]
        effective_n = model.predict_diagnostics([query])["effective_n"][0]
        ax.plot(
            X,
            values,
            color=color,
            lw=3,
            label=f"{label} (effective n ≈ {effective_n:.0f})",
        )
    ax.axvline(query, color="#334155", linestyle="--", lw=2, label="Query = 3")
    _legend(ax)
    return figure


def intervals():
    """Display supported pointwise bootstrap bounds for a regression curve."""
    rng = np.random.default_rng(17)
    X = np.sort(rng.uniform(0, 6, 100))
    y = np.sin(X) + rng.normal(0, 0.3, len(X))
    query = np.linspace(X.min(), X.max(), 100)
    model = _smoother(0.4).fit(X, y)
    interval = model.predict_interval(query, n_resamples=100, random_state=17)
    figure, ax = _axes("Pointwise bootstrap uncertainty for the curve")
    _observations(ax, X, y)
    ax.fill_between(
        query,
        interval["lower"].to_numpy(),
        interval["upper"].to_numpy(),
        color=COLORS[0],
        alpha=0.2,
        label="95% pointwise bootstrap bounds",
    )
    ax.plot(
        query,
        interval["prediction"].to_numpy(),
        color=COLORS[0],
        lw=3,
        label="Fitted local mean",
    )
    ax.plot(query, np.sin(query), "--", color="#334155", lw=2, label="Generating mean")
    _legend(ax)
    return figure


def support():
    """Expose a feature gap rather than drawing a supported curve across it."""
    rng = np.random.default_rng(23)
    X = np.r_[np.linspace(0, 2, 70), np.linspace(4, 6, 70)]
    y = np.sin(X) + rng.normal(0, 0.12, len(X))
    query = np.linspace(0, 6, 350)
    model = KernelSmoother(
        kernel="epanechnikov", bandwidth="manual", bandwidth_value=0.35
    ).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", SupportWarning
        )  # Gap is highlighted explicitly below.
        diagnostic = model.predict_diagnostics(query)
    unsupported = ~diagnostic["supported"].to_numpy()
    figure, ax = _axes("A gap in observations can mean no estimate")
    _observations(ax, X, y)
    ax.plot(
        query,
        diagnostic["prediction"].to_numpy(),
        color=COLORS[0],
        lw=3,
        label="Compact-kernel estimate",
    )
    ax.axvspan(
        query[unsupported].min(),
        query[unsupported].max(),
        color=COLORS[1],
        alpha=0.15,
        label="Unsupported queries → NaN",
    )
    _legend(ax)
    return figure


def residuals():
    """Use residuals to inspect structure remaining after heavy smoothing."""
    rng = np.random.default_rng(22)
    X = np.linspace(0, 6, 150)
    y = np.sin(X) + rng.normal(0, 0.15, len(X))
    model = _smoother(1.4).fit(X, y)
    diagnostic = model.regression_diagnostics(X, y)
    figure, ax = _axes(
        "Residuals reveal structure left by heavy smoothing",
        ylabel="Observed − predicted",
    )
    ax.scatter(
        X,
        diagnostic["residual"].to_numpy(),
        color=COLORS[0],
        alpha=0.65,
        s=32,
        edgecolors="none",
        label="Residuals at original rows",
    )
    ax.axhline(0, color="#334155", linestyle="--", lw=2, label="Zero residual")
    _legend(ax)
    return figure


EXAMPLES = {
    "bins": bins,
    "bandwidth": bandwidth,
    "subgroups": subgroups,
    "boundary": boundary,
    "summaries": summaries,
    "influence": influence,
    "intervals": intervals,
    "support": support,
    "residuals": residuals,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["all", *EXAMPLES], default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save separate PNG figures instead of displaying them",
    )
    args = parser.parse_args()
    import matplotlib

    if args.output_dir:
        matplotlib.use("Agg")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    selected = (
        EXAMPLES if args.example == "all" else {args.example: EXAMPLES[args.example]}
    )
    for name, create in selected.items():
        figure = create()
        if args.output_dir:
            figure.savefig(args.output_dir / f"{name}.png", dpi=180)
        else:
            plt.show()  # Close the current figure to continue to the next example.
        plt.close(figure)


if __name__ == "__main__":
    main()
