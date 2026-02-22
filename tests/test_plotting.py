"""Tests for plotting functions."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend


class TestPlotImports:
    def test_plot_irf_importable(self):
        from litterman.plotting import plot_irf

        assert callable(plot_irf)

    def test_plot_fevd_importable(self):
        from litterman.plotting import plot_fevd

        assert callable(plot_fevd)

    def test_plot_forecast_importable(self):
        from litterman.plotting import plot_forecast

        assert callable(plot_forecast)

    def test_plot_historical_decomposition_importable(self):
        from litterman.plotting import plot_historical_decomposition

        assert callable(plot_historical_decomposition)
