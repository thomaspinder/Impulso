"""Plotting functions for VAR results."""

import numpy as np

from impulso.plotting._fevd import plot_fevd
from impulso.plotting._forecast import plot_forecast
from impulso.plotting._historical_decomposition import plot_historical_decomposition
from impulso.plotting._irf import plot_irf

__all__ = ["plot_fevd", "plot_forecast", "plot_historical_decomposition", "plot_irf"]


def plot_volatility(result, hdi_prob_outer: float = 0.90, hdi_prob_inner: float = 0.68):
    """Plot posterior conditional-SD path with HDI bands.

    Args:
        result: VolatilityResult.
        hdi_prob_outer: Outer (lighter) HDI probability.
        hdi_prob_inner: Inner (darker) HDI probability.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    med = result.median()
    hdi_outer = result.hdi(prob=hdi_prob_outer)
    hdi_inner = result.hdi(prob=hdi_prob_inner)

    fig, ax = plt.subplots(figsize=(10, 4))
    col = result.series_name
    ax.fill_between(
        med.index,
        hdi_outer.lower[col].values,
        hdi_outer.upper[col].values,
        alpha=0.15,
        color="C0",
        label=f"{int(hdi_prob_outer * 100)}% HDI",
    )
    ax.fill_between(
        med.index,
        hdi_inner.lower[col].values,
        hdi_inner.upper[col].values,
        alpha=0.3,
        color="C0",
        label=f"{int(hdi_prob_inner * 100)}% HDI",
    )
    ax.plot(med.index, med[col].values, color="C0", linewidth=1.5, label="Median")
    ax.set_ylabel(f"Conditional SD ({col})")
    ax.set_title(f"Posterior volatility — {col}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sv_forecast(result, hdi_prob_outer: float = 0.90, hdi_prob_inner: float = 0.68):
    """Plot density forecast fan for a univariate SV forecast.

    Args:
        result: SVForecastResult.
        hdi_prob_outer: Outer (lighter) HDI probability.
        hdi_prob_inner: Inner (darker) HDI probability.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

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
