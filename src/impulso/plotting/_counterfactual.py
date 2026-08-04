"""Counterfactual plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from impulso._arviz_compat import hdi_bounds

if TYPE_CHECKING:
    from impulso.results import CounterfactualResult


def plot_counterfactual(
    result: "CounterfactualResult",
    prob: float = 0.89,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot actual vs counterfactual paths, one panel per variable.

    The actual path is drawn in black; the counterfactual posterior
    median in colour with its HDI band shaded around it.

    Args:
        result: CounterfactualResult.
        prob: Probability mass for the HDI band. Default 0.89.
        figsize: Figure size.  Defaults to (12, 3 * n_vars).

    Returns:
        Matplotlib Figure.
    """
    pp = result._pp()
    cf_da = pp["counterfactual"]
    actual_da = pp["actual"]
    med = cf_da.median(dim=("chain", "draw"))
    hdi_lower, hdi_upper = hdi_bounds(cf_da, prob)
    time = cf_da.coords["time"].values
    n_vars = len(result.var_names)

    if figsize is None:
        figsize = (12, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]
    fig.suptitle("Historical Counterfactual")

    for i, var in enumerate(result.var_names):
        axes[i].plot(time, actual_da.sel(variable=var).values, color="black", linewidth=1.2, label="actual")
        axes[i].plot(
            time,
            med.sel(variable=var).values,
            color="C0",
            linewidth=1.2,
            label="counterfactual (median)",
        )
        axes[i].fill_between(
            time,
            hdi_lower.sel(variable=var).values,
            hdi_upper.sel(variable=var).values,
            color="C0",
            alpha=0.25,
            linewidth=0,
            label=f"{int(prob * 100)}% HDI",
        )
        axes[i].set_ylabel(var)
        if i == 0:
            axes[i].legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig
