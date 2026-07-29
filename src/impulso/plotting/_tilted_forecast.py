"""Plotting for entropically tilted forecasts and reverse stress tests."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import ReverseStressResult, TiltedForecastResult


def _tilt_subtitle(result: "TiltedForecastResult | ReverseStressResult") -> str:
    """Effective-sample-size and relative-entropy summary for a suptitle."""
    attrs = result.idata.posterior_predictive.attrs
    n_draws = result.idata.posterior_predictive["tilting_weights"].size
    return f"ESS {attrs['ess']:.0f} of {n_draws} draws, KL {attrs['kl_divergence']:.3f}"


def plot_tilted_forecast(
    result: "TiltedForecastResult",
    prob: float = 0.89,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot the tilted forecast against the untilted median, one panel per variable.

    The weighted median and its weighted HDI band show the tilted
    distribution; the untilted median is overlaid dashed so the effect of
    the targets is visible directly.

    Args:
        result: TiltedForecastResult.
        prob: Probability mass for the HDI band. Default 0.89.
        figsize: Figure size. Defaults to `(12, 3 * n_vars)`.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    base = result.base_median()
    hdi = result.hdi(prob=prob)
    steps_axis = np.arange(1, result.steps + 1)
    n_vars = len(result.var_names)

    if figsize is None:
        figsize = (12, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.suptitle(f"Tilted Forecast ({_tilt_subtitle(result)})")

    for i, var in enumerate(result.var_names):
        axes[i].plot(steps_axis, med[var].values, color="C0", linewidth=1.2, label="tilted median")
        axes[i].fill_between(
            steps_axis,
            hdi.lower[var].values,
            hdi.upper[var].values,
            color="C0",
            alpha=0.25,
            linewidth=0,
            label=f"{int(prob * 100)}% HDI (tilted)",
        )
        axes[i].plot(
            steps_axis,
            base[var].values,
            color="grey",
            linewidth=1.0,
            linestyle="--",
            label="untilted median",
        )
        axes[i].set_ylabel(var)
        if i == 0:
            axes[i].legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    return fig


def plot_reverse_stress(
    result: "ReverseStressResult",
    prob: float = 0.89,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot the stressed variable's conditioned fan and the shock cocktail.

    The top panel shows the stressed variable under the event-conditioned
    weights, with the threshold marked and the untilted median dashed for
    reference. One bar panel per structural shock then shows the
    cocktail — the tilted-weighted mean shock path, in
    one-standard-deviation units.

    Args:
        result: ReverseStressResult.
        prob: Probability mass for the HDI band. Default 0.89.
        figsize: Figure size. Defaults to `(12, 3 * (1 + n_shocks))`.

    Returns:
        Matplotlib Figure.
    """
    cocktail = result.shock_cocktail()
    n_shocks = len(result.shock_names)
    steps_axis = np.arange(1, result.steps + 1)

    if figsize is None:
        figsize = (12, 3 * (1 + n_shocks))

    fig, axes = plt.subplots(1 + n_shocks, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]
    attrs = result.idata.posterior_predictive.attrs
    sign = "<" if result.direction == "below" else ">"
    fig.suptitle(
        f"Reverse Stress: P({result.variable}[h={result.horizon}] {sign} {result.threshold:g}) "
        f"= {attrs['achieved_probability']:.2f} (baseline {attrs['baseline_probability']:.2f}; "
        f"{_tilt_subtitle(result)}; q = {attrs['q']:.2f})"
    )

    med = result.median()
    base = result.base_median()
    hdi = result.hdi(prob=prob)
    var = result.variable
    axes[0].plot(steps_axis, med[var].values, color="C3", linewidth=1.2, label="conditioned median")
    axes[0].fill_between(
        steps_axis,
        hdi.lower[var].values,
        hdi.upper[var].values,
        color="C3",
        alpha=0.25,
        linewidth=0,
        label=f"{int(prob * 100)}% HDI (conditioned)",
    )
    axes[0].plot(steps_axis, base[var].values, color="grey", linewidth=1.0, linestyle="--", label="untilted median")
    axes[0].axhline(result.threshold, color="black", linewidth=0.8, linestyle=":", label="threshold")
    axes[0].scatter([result.horizon], [result.threshold], color="black", marker="x", s=30, zorder=3)
    axes[0].set_ylabel(var)
    axes[0].legend(fontsize=8, loc="upper right")

    for j, shock in enumerate(result.shock_names):
        ax = axes[1 + j]
        ax.bar(steps_axis, cocktail[shock].values, color="C1", width=0.7)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{shock}\n(sd units)")

    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    return fig
