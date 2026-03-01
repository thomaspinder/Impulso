"""FEVD plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import FEVDResult


def plot_fevd(
    result: FEVDResult,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot forecast error variance decomposition as stacked areas.

    One panel per response variable, showing the share of forecast error
    variance attributable to each structural shock.

    Args:
        result: FEVDResult from IdentifiedVAR.fevd().
        figsize: Figure size.  Defaults to (12, 3 * n_vars).

    Returns:
        Matplotlib Figure.
    """
    fevd_da = result.idata.posterior_predictive["fevd"]
    med = fevd_da.median(dim=("chain", "draw"))
    n_vars = len(result.var_names)
    horizons = range(result.horizon + 1)

    if figsize is None:
        figsize = (7, 2 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]
    fig.suptitle("Forecast Error Variance Decomposition")

    for i, resp in enumerate(result.var_names):
        shares = med.sel(response=resp).values  # (horizon+1, n_shocks)
        axes[i].stackplot(horizons, shares.T, labels=result.var_names, alpha=0.8)
        axes[i].set_ylabel(resp)
        axes[i].set_ylim(0, 1)
        if i == 0:
            axes[i].legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Horizon")
    fig.tight_layout()
    return fig
