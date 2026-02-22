"""Forecast plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import ForecastResult


def plot_forecast(
    result: ForecastResult,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot forecast fan chart with credible bands.

    Args:
        result: ForecastResult from FittedVAR.forecast().
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    raise NotImplementedError("Plotting is implemented in phase-5-plotting-api")
