"""Historical decomposition plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import HistoricalDecompositionResult


def plot_historical_decomposition(
    result: HistoricalDecompositionResult,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot historical decomposition as stacked bar charts.

    One panel per response variable, showing the contribution of each
    structural shock over time.

    Args:
        result: HistoricalDecompositionResult.
        figsize: Figure size.  Defaults to (12, 3 * n_vars).

    Returns:
        Matplotlib Figure.
    """
    hd_da = result.idata.posterior_predictive["hd"]
    med = hd_da.median(dim=("chain", "draw"))
    n_vars = len(result.var_names)
    T = med.sizes["time"]

    if figsize is None:
        figsize = (12, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]
    fig.suptitle("Historical Decomposition")

    time_idx = range(T)
    for i, resp in enumerate(result.var_names):
        panel = med.sel(response=resp).values  # (T, n_shocks)
        bottom_pos = None
        bottom_neg = None
        for j, shock in enumerate(result.var_names):
            vals = panel[:, j]
            pos = vals.clip(min=0)
            neg = vals.clip(max=0)
            if bottom_pos is None:
                axes[i].bar(time_idx, pos, width=1.0, label=shock, alpha=0.8)
                axes[i].bar(time_idx, neg, width=1.0, alpha=0.8, color=f"C{j}")
                bottom_pos = pos.copy()
                bottom_neg = neg.copy()
            else:
                axes[i].bar(time_idx, pos, width=1.0, bottom=bottom_pos, label=shock, alpha=0.8)
                axes[i].bar(time_idx, neg, width=1.0, bottom=bottom_neg, alpha=0.8, color=f"C{j}")
                bottom_pos += pos
                bottom_neg += neg
        axes[i].set_ylabel(resp)
        axes[i].axhline(0, color="0.5", linewidth=0.5, linestyle="--")
        if i == 0:
            axes[i].legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig
