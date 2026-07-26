"""Exogenous dynamic-multiplier plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import DynamicMultiplierResult


def plot_dynamic_multiplier(
    result: "DynamicMultiplierResult",
    variables: list[str] | None = None,
    figsize: tuple[float, float] = (9, 6),
) -> Figure:
    """Plot exogenous dynamic multipliers with credible bands.

    Lays out a grid of response variables (rows) by exogenous drivers
    (columns), mirroring `plot_irf`'s response-by-shock grid.

    Args:
        result: DynamicMultiplierResult from FittedVAR.dynamic_multiplier().
        variables: Optional subset of response variables to plot.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    hdi = result.hdi()
    var_names = variables or result.var_names
    exog_names = result.exog_names

    fig, axes = plt.subplots(len(var_names), len(exog_names), figsize=figsize, squeeze=False)
    kind = "Cumulative dynamic multipliers" if result.cumulative else "Dynamic multipliers"
    fig.suptitle(kind)

    horizons = range(result.horizon + 1)
    for i, resp in enumerate(var_names):
        for j, exog in enumerate(exog_names):
            ax = axes[i][j]
            ax.set_title(f"{exog} -> {resp}", fontsize=9)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.plot(horizons, med[(resp, exog)].values)
            ax.fill_between(
                horizons,
                hdi.lower[(resp, exog)].values,
                hdi.upper[(resp, exog)].values,
                alpha=0.3,
            )

    fig.tight_layout()
    return fig
