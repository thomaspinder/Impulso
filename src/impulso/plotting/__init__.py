"""Plotting functions for VAR results."""

from impulso.plotting._fevd import plot_fevd
from impulso.plotting._forecast import plot_forecast
from impulso.plotting._historical_decomposition import plot_historical_decomposition
from impulso.plotting._irf import plot_irf

__all__ = ["plot_fevd", "plot_forecast", "plot_historical_decomposition", "plot_irf"]


def plot_volatility(result):
    """Stub — implemented in Task 7."""
    raise NotImplementedError("plot_volatility is implemented in Task 7")


def plot_sv_forecast(result):
    """Stub — implemented in Task 7."""
    raise NotImplementedError("plot_sv_forecast is implemented in Task 7")
