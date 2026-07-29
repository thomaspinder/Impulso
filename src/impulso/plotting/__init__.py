"""Plotting functions for VAR results."""

from impulso.plotting._conditional_forecast import plot_conditional_forecast
from impulso.plotting._counterfactual import plot_counterfactual
from impulso.plotting._dynamic_multiplier import plot_dynamic_multiplier
from impulso.plotting._fevd import plot_fevd
from impulso.plotting._forecast import plot_forecast
from impulso.plotting._historical_decomposition import plot_historical_decomposition
from impulso.plotting._irf import plot_irf
from impulso.plotting._stability import plot_stability
from impulso.plotting._structural_scenario import plot_structural_scenario
from impulso.plotting._sv_forecast import plot_sv_forecast
from impulso.plotting._sv_volatility import plot_volatility

__all__ = [
    "plot_conditional_forecast",
    "plot_counterfactual",
    "plot_dynamic_multiplier",
    "plot_fevd",
    "plot_forecast",
    "plot_historical_decomposition",
    "plot_irf",
    "plot_stability",
    "plot_structural_scenario",
    "plot_sv_forecast",
    "plot_volatility",
]
