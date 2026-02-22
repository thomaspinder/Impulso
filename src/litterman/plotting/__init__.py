"""Plotting utilities for VAR analysis."""

from litterman.plotting._fevd import plot_fevd
from litterman.plotting._forecast import plot_forecast
from litterman.plotting._historical_decomposition import plot_historical_decomposition
from litterman.plotting._irf import plot_irf

__all__ = ["plot_fevd", "plot_forecast", "plot_historical_decomposition", "plot_irf"]
