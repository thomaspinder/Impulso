"""IRF plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import IRFResult


def plot_irf(
    result: IRFResult,
    variables: list[str] | None = None,
    figsize: tuple[float, float] = (9, 6),
) -> Figure:
    """Plot impulse response functions with credible bands.

    Args:
        result: IRFResult from IdentifiedVAR.impulse_response().
        variables: Optional subset of response variables to plot.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    hdi = result.hdi()
    var_names = variables or result.var_names
    n_vars = len(var_names)

    fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize, squeeze=False)
    fig.suptitle("Impulse Response Functions")

    horizons = range(result.horizon + 1)
    for i, resp in enumerate(var_names):
        for j, shock in enumerate(var_names):
            ax = axes[i][j]
            ax.set_title(f"{shock} -> {resp}", fontsize=9)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            col_idx = i * n_vars + j
            ax.plot(horizons, med.values[:, col_idx])
            ax.fill_between(
                horizons,
                hdi.lower.values[:, col_idx],
                hdi.upper.values[:, col_idx],
                alpha=0.3,
            )

    fig.tight_layout()
    return fig
