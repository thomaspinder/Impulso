"""Historical decomposition plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    raise NotImplementedError("Plotting is implemented in phase-5-plotting-api")
