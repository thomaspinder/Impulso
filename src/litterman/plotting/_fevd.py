"""Forecast error variance decomposition plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    raise NotImplementedError("Plotting is implemented in phase-5-plotting-api")
