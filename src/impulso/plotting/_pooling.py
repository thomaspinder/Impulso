"""Predictive-pool plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.pooling import PredictivePool

_INSIDE_LABEL_FRACTION = 0.75
"""Share of the x-axis a bar must cover before its score label moves inside it."""


def plot_pool_weights(
    pool: "PredictivePool",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot pool weights as a ranked horizontal bar chart.

    Bars run heaviest weight first and are annotated with each model's total
    held-out log score, so a model carrying little weight can be read against
    how badly it actually scored. The title reports the weighting method and
    the pooled score alongside the best single model's, which is the number
    that says whether pooling bought anything.

    Args:
        pool: PredictivePool from `pool_forecasts`.
        figsize: Figure size. Defaults to `(8, 1 + 0.5 * n_models)`.

    Returns:
        Matplotlib Figure.
    """
    summary = pool.summary()
    labels = list(summary.index)
    weights = summary["weight"].to_numpy(dtype=float)
    scores = summary["log_score"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize or (8.0, 1.0 + 0.5 * len(labels)))
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions, weights, color="tab:blue", alpha=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("weight")
    upper = max(1.0, float(weights.max()) * 1.05)
    ax.set_xlim(0.0, upper)

    offset = 0.01 * upper
    for position, weight, score in zip(positions, weights, scores, strict=True):
        # A label parked past the end of a near-full bar runs off the right
        # axis, so once a bar covers most of the width its label moves inside,
        # right-aligned against the bar end and recoloured to stay legible.
        if weight > _INSIDE_LABEL_FRACTION * upper:
            ax.text(
                weight - offset, position, f"log score {score:,.1f}", va="center", ha="right", fontsize=8, color="white"
            )
        else:
            ax.text(weight + offset, position, f"log score {score:,.1f}", va="center", ha="left", fontsize=8)

    pooled = pool.pooled_log_score()
    ax.set_title(
        f"Pool weights ({pool.method}, {pool.density} density) — "
        f"pooled log score {pooled:,.1f} vs best single {scores.max():,.1f}"
    )
    fig.tight_layout()
    return fig
