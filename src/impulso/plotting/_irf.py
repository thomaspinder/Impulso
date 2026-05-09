"""IRF plotting."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import IRFResult


def plot_irf(
    result: "IRFResult",
    variables: list[str] | None = None,
    figsize: tuple[float, float] = (9, 6),
) -> Figure:
    """Plot impulse response functions with credible bands.

    Args:
        result: IRFResult from IdentifiedVAR.impulse_response().
        variables: Optional subset of response variables to plot.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    hdi = result.hdi()
    response_names = variables or list(med.coords["response"].values)
    shock_names = list(med.coords["shock"].values)

    fig, axes = plt.subplots(len(response_names), len(shock_names), figsize=figsize, squeeze=False)
    fig.suptitle("Impulse Response Functions")

    horizons = med.coords["horizon"].values
    for i, resp in enumerate(response_names):
        for j, shock in enumerate(shock_names):
            ax = axes[i][j]
            ax.set_title(f"{shock} -> {resp}", fontsize=9)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            line = med.sel(response=resp, shock=shock)
            interval = hdi.sel(response=resp, shock=shock)
            ax.plot(horizons, line.values)
            ax.fill_between(
                horizons,
                interval.sel(hdi="lower").values,
                interval.sel(hdi="higher").values,
                alpha=0.3,
            )

    fig.tight_layout()
    return fig
