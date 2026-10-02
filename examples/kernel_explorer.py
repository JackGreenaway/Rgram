"""Explore bins, kernel bandwidth, local-linear fits and row influence.

Run: uv run python examples/kernel_explorer.py
The interval button computes pointwise IID paired-bootstrap confidence intervals
for the fitted mean curve, conditional on bandwidth; not prediction intervals.
Requires an interactive Matplotlib backend (included in dev dependencies).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from rgram import KernelSmoother, Regressogram


def main() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(
        0, 10, 160
    )  # Generate ordered demo data; the estimator never sorts.
    y = np.sin(x) + rng.normal(0, 0.3, len(x))
    grid = np.linspace(0, 10, 180)
    fig, (curve_ax, weights_ax) = plt.subplots(2, 1, figsize=(12, 9))
    fig.subplots_adjust(left=0.29, bottom=0.26, hspace=0.48)
    curve_ax.scatter(x, y, s=12, alpha=0.4, label="All training observations")
    (binned,) = curve_ax.plot([], [], color="gray", alpha=0.7, label="Regressogram")
    (curve,) = curve_ax.plot([], [], color="tab:blue", label="Kernel smoother")
    (marker,) = curve_ax.plot(
        [], [], "o", color="tab:red", label="Inspected prediction"
    )
    query_line = curve_ax.axvline(5, color="tab:red", alpha=0.3)
    curve_ax.legend(loc="upper right", fontsize=8)
    curve_ax.set(xlabel="x", ylabel="y", ylim=(-2, 2))
    bars = weights_ax.bar(x, np.zeros_like(x), width=0.04)
    weights_ax.axhline(0, color="gray", linewidth=0.5)
    weights_ax.set(
        xlim=(0, 10),
        xlabel="Training x (one bar per original row)",
        ylabel="Prediction coefficient",
    )

    bandwidth = Slider(
        fig.add_axes([0.34, 0.17, 0.56, 0.025]), "Bandwidth", 0.15, 3, valinit=0.8
    )
    query = Slider(fig.add_axes([0.34, 0.12, 0.56, 0.025]), "Query x", 0, 10, valinit=5)
    count = Slider(
        fig.add_axes([0.34, 0.07, 0.56, 0.025]), "Bins", 1, 30, valinit=12, valstep=1
    )
    kernel_names = sorted(KernelSmoother._KERNELS)
    kernels = RadioButtons(
        fig.add_axes([0.015, 0.54, 0.2, 0.36]),
        kernel_names,
        active=kernel_names.index("gaussian"),
    )
    regression = RadioButtons(
        fig.add_axes([0.015, 0.38, 0.2, 0.13]), ["local_constant", "local_linear"]
    )
    binning = RadioButtons(fig.add_axes([0.015, 0.24, 0.2, 0.11]), ["dist", "width"])
    interval_button = Button(fig.add_axes([0.02, 0.1, 0.19, 0.07]), "95% mean-curve CI")
    state = {"model": None, "band": None}

    def update(_: object = None) -> None:
        if state["band"] is not None:
            state["band"].remove()
            state["band"] = None
        model = KernelSmoother(
            bandwidth="manual",
            bandwidth_value=bandwidth.val,
            kernel=kernels.value_selected,
            regression=regression.value_selected,
        ).fit(x, y)
        state["model"] = model
        curve.set_data(grid, model.predict(grid))
        binned.set_data(
            grid,
            Regressogram(binning=binning.value_selected, n_bins=int(count.val))
            .fit(x, y)
            .predict(grid),
        )
        diag = model.predict_diagnostics([query.val])
        marker.set_data([query.val], diag["prediction"].to_numpy())
        query_line.set_xdata([query.val, query.val])
        coefficients = model.get_weights([query.val])[0]
        for bar, coefficient in zip(bars, coefficients):
            bar.set_height(coefficient)
        weights_ax.set_ylim(
            min(-0.01, coefficients.min() * 1.15), max(0.05, coefficients.max() * 1.15)
        )
        weights_ax.set_title(
            f"{diag['n_neighbors'][0]} contributing rows; kernel effective n = {diag['effective_n'][0]:.1f}"
        )
        curve_ax.set_title(
            f"{kernels.value_selected}; {regression.value_selected}; h = {model.bandwidth_:.2f}"
        )
        fig.canvas.draw_idle()

    def show_interval(_: object) -> None:
        if state["band"] is not None:
            state["band"].remove()
        result = state["model"].predict_interval(grid, n_resamples=200, random_state=42)
        state["band"] = curve_ax.fill_between(
            grid,
            result["lower"].to_numpy(),
            result["upper"].to_numpy(),
            color="tab:blue",
            alpha=0.2,
        )
        curve_ax.set_title("Pointwise 95% IID bootstrap mean-curve CI; fixed bandwidth")
        fig.canvas.draw_idle()

    for control in (bandwidth, query, count):
        control.on_changed(update)
    for control in (kernels, regression, binning):
        control.on_clicked(update)
    interval_button.on_clicked(show_interval)
    update()
    plt.show()
    return fig  # Allows headless smoke checks without an interactive session.


if __name__ == "__main__":
    main()
