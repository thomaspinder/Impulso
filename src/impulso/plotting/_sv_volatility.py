"""Volatility path plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import VolatilityResult


def plot_volatility(
    result: "VolatilityResult",
    hdi_prob_outer: float = 0.90,
    hdi_prob_inner: float = 0.68,
) -> Figure:
    """Plot posterior conditional-SD path with HDI bands.

    Args:
        result: VolatilityResult.
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
