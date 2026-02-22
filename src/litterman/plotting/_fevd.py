"""FEVD plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import FEVDResult


def plot_fevd(
    result: FEVDResult,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot forecast error variance decomposition as stacked areas.

    Args:
        result: FEVDResult from IdentifiedVAR.fevd().
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Forecast Error Variance Decomposition")
    ax.stackplot(range(result.horizon + 1), med.values.T, labels=result.var_names, alpha=0.8)
    ax.legend()
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Share")
    fig.tight_layout()
    return fig
