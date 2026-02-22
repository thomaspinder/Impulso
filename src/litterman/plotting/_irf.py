"""Impulse response function plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import IRFResult


def plot_irf(
    result: IRFResult,
    variables: list[str] | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Plot impulse response functions with credible bands.

    Args:
        result: IRFResult from IdentifiedVAR.impulse_response().
        variables: Optional subset of response variables to plot.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    raise NotImplementedError("Plotting is implemented in phase-5-plotting-api")
