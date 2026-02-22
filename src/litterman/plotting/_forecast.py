"""Forecast plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
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
    med = result.median()
    hdi = result.hdi()
    n_vars = len(result.var_names)

    fig, axes = plt.subplots(1, n_vars, figsize=figsize, squeeze=False)
    fig.suptitle("Forecast")

    for i, name in enumerate(result.var_names):
        ax = axes[0][i]
        ax.set_title(name)
        steps = range(result.steps)
        ax.plot(steps, med[name].values)
        ax.fill_between(steps, hdi.lower[name].values, hdi.upper[name].values, alpha=0.3)

    fig.tight_layout()
    return fig
