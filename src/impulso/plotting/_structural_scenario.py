"""Structural-scenario plotting."""

from typing import TYPE_CHECKING

from matplotlib.figure import Figure

if TYPE_CHECKING:
    from impulso.results import ScenarioResult


def plot_structural_scenario(
    result: "ScenarioResult",
    prob: float = 0.89,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot a structural scenario, one panel per variable.

    Shares the conditional-forecast fan layout (median, HDI band, pinned
    values marked) with a scenario title; the adjusting set, when
    restricted, is reported in the suptitle.

    Args:
        result: ScenarioResult.
        prob: Probability mass for the HDI band. Default 0.89.
        figsize: Figure size.  Defaults to (12, 3 * n_vars).

    Returns:
        Matplotlib Figure.
    """
    from impulso.plotting._conditional_forecast import plot_conditional_forecast

    fig = plot_conditional_forecast(result, prob=prob, figsize=figsize)
    title = fig._suptitle.get_text().replace("Conditional Forecast", "Structural Scenario")
    if result.adjusting is not None:
        names = ", ".join(result.adjusting) if result.adjusting else "(none)"
        title += f" — adjusting: {names}"
    fig.suptitle(title)
    return fig
