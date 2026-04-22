"""Plotting functions for VAR results."""

from impulso.plotting._fevd import plot_fevd
from impulso.plotting._forecast import plot_forecast
from impulso.plotting._historical_decomposition import plot_historical_decomposition
from impulso.plotting._irf import plot_irf
from impulso.plotting._sv_forecast import plot_sv_forecast
from impulso.plotting._sv_volatility import plot_volatility

__all__ = [
    "plot_fevd",
    "plot_forecast",
    "plot_historical_decomposition",
    "plot_irf",
    "plot_sv_forecast",
    "plot_volatility",
]
