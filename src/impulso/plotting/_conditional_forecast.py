"""Conditional-forecast plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import ConditionalForecastResult


def plot_conditional_forecast(
    result: "ConditionalForecastResult",
    prob: float = 0.89,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot the conditional forecast, one panel per variable.

    The posterior median is drawn with its HDI band; pinned values are
    marked so the conditioning is visible. The suptitle reports the
    median calibrated plausibility when restrictions bind.

    Args:
        result: ConditionalForecastResult.
        prob: Probability mass for the HDI band. Default 0.89.
        figsize: Figure size.  Defaults to (12, 3 * n_vars).

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    hdi = result.hdi(prob)
    steps_axis = np.arange(1, result.steps + 1)
    n_vars = len(result.var_names)
    pins = result.pinned_values()

    if figsize is None:
        figsize = (12, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]
    title = "Conditional Forecast"
    n_restrictions = result.n_restrictions
    if n_restrictions:
        q_cal = result.plausibility()["q_calibrated_median"]
        title += f" (calibrated plausibility q = {q_cal:.2f})"
    fig.suptitle(title)

    for i, var in enumerate(result.var_names):
        axes[i].plot(
            steps_axis,
            med[var].values,
            color="C0",
            linewidth=1.2,
            label="median",
        )
        axes[i].fill_between(
            steps_axis,
            hdi.lower[var].values,
            hdi.upper[var].values,
            color="C0",
            alpha=0.25,
            linewidth=0,
            label=f"{int(prob * 100)}% HDI",
        )
        pinned = pins[var]
        if pinned:
            xs, ys = zip(*pinned, strict=True)
            axes[i].scatter(xs, ys, color="black", marker="x", s=30, zorder=3, label="pinned")
        axes[i].set_ylabel(var)
        if i == 0:
            axes[i].legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    return fig
