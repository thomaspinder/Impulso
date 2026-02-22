"""Historical decomposition plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import HistoricalDecompositionResult


def plot_historical_decomposition(
    result: HistoricalDecompositionResult,
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Plot historical decomposition as stacked bar chart.

    Args:
        result: HistoricalDecompositionResult.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Historical Decomposition")
    T = med.shape[0]
    bottom = None
    for i, name in enumerate(result.var_names):
        vals = med.iloc[:, i].values
        if bottom is None:
            ax.bar(range(T), vals, label=name, alpha=0.8)
            bottom = vals.copy()
        else:
            ax.bar(range(T), vals, bottom=bottom, label=name, alpha=0.8)
            bottom += vals
    ax.legend()
    ax.set_xlabel("Time")
    fig.tight_layout()
    return fig
