"""Companion-matrix stability plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.diagnostics import StabilitySummary

# Legend labels for the two reference marks, named so tests can find the
# artists by label rather than by position in `ax.lines`.
UNIT_CIRCLE_LABEL = "unit circle"
UNIT_ROOT_LABEL = "unit root"


def plot_stability(
    summary: "StabilitySummary",
    figsize: tuple[float, float] = (11, 4.5),
    bins: int = 40,
) -> Figure:
    """Plot the spectral-radius posterior beside the eigenvalue scatter.

    Two views of the same question. The left panel is the posterior of the
    companion-matrix spectral radius with the unit root marked, answering
    *how much mass is explosive*. The right panel scatters the individual
    companion roots on the complex plane against the unit circle, answering
    *which* roots drive it: a single real root creeping outward is a
    near-unit-root level series, whereas a conjugate pair approaching the
    circle is an oscillatory mode with a long-lived cycle.

    The scatter uses the eigenvalue subset the summary retains (at most 200
    pooled draws), not every draw, so the panel is bounded in cost and in
    ink regardless of posterior size.

    Args:
        summary: `StabilitySummary` from a convergence report.
        figsize: Figure size.
        bins: Histogram bin count for the spectral-radius panel.

    Returns:
        Matplotlib Figure with two axes.
    """
    radius = np.asarray(summary.radius, dtype=float).reshape(-1)
    eigenvalues = np.asarray(summary.eigenvalues).reshape(-1)

    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Dynamic stability")

    ax_hist.hist(radius, bins=bins, color="C0", alpha=0.8)
    ax_hist.axvline(1.0, color="C3", linestyle="--", linewidth=1.2, label=UNIT_ROOT_LABEL)
    ax_hist.set_xlabel("Spectral radius")
    ax_hist.set_ylabel("Draws")
    ax_hist.set_title(f"Explosive draws: {summary.p_explosive:.1%}")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax_scatter.plot(np.cos(theta), np.sin(theta), color="C3", linewidth=1.2, label=UNIT_CIRCLE_LABEL)
    ax_scatter.scatter(eigenvalues.real, eigenvalues.imag, s=6, alpha=0.3, color="C0", label="eigenvalues")
    ax_scatter.axhline(0.0, color="0.7", linewidth=0.6, zorder=0)
    ax_scatter.axvline(0.0, color="0.7", linewidth=0.6, zorder=0)
    ax_scatter.set_aspect("equal", adjustable="datalim")
    ax_scatter.set_xlabel("Re")
    ax_scatter.set_ylabel("Im")
    ax_scatter.set_title(f"Companion roots ({summary.eigenvalues.shape[0]} draws)")
    ax_scatter.legend(fontsize=8)

    fig.tight_layout()
    return fig
