"""Stochastic-volatility density-forecast plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import SVForecastResult


def plot_sv_forecast(
    result: "SVForecastResult",
    hdi_prob_outer: float = 0.90,
    hdi_prob_inner: float = 0.68,
) -> Figure:
    """Plot density forecast fan for a univariate SV forecast.

    Args:
        result: SVForecastResult.
        hdi_prob_outer: Outer (lighter) HDI probability.
        hdi_prob_inner: Inner (darker) HDI probability.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    hdi_outer = result.hdi(prob=hdi_prob_outer)
    hdi_inner = result.hdi(prob=hdi_prob_inner)

    fig, ax = plt.subplots(figsize=(10, 4))
    col = result.series_name
    steps = np.arange(1, result.steps + 1)
    ax.fill_between(
        steps,
        hdi_outer.lower[col].values,
        hdi_outer.upper[col].values,
        alpha=0.15,
        color="C0",
        label=f"{int(hdi_prob_outer * 100)}% HDI",
    )
    ax.fill_between(
        steps,
        hdi_inner.lower[col].values,
        hdi_inner.upper[col].values,
        alpha=0.3,
        color="C0",
        label=f"{int(hdi_prob_inner * 100)}% HDI",
    )
    ax.plot(steps, med[col].values, color="C0", linewidth=1.5, label="Median")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Step ahead")
    ax.set_ylabel(col)
    ax.set_title(f"Density forecast — {col}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
