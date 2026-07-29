"""Tests for plotting functions."""

import arviz as az
import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

matplotlib.use("Agg")

from impulso.plotting import plot_fevd, plot_forecast, plot_historical_decomposition, plot_irf, plot_stability
from impulso.plotting._stability import UNIT_CIRCLE_LABEL, UNIT_ROOT_LABEL
from impulso.results import FEVDResult, ForecastResult, HistoricalDecompositionResult, IRFResult


@pytest.fixture(autouse=True)
def _close_figures():
    """Plot functions return an open pyplot figure; drop it after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _make_forecast_result(n_vars=2, steps=8) -> ForecastResult:
    rng = np.random.default_rng(42)
    names = [f"y{i + 1}" for i in range(n_vars)]
    data = rng.standard_normal((2, 50, steps, n_vars))
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "step", "variable"],
        coords={"variable": names},
        name="forecast",
    )
    idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": da}))
    return ForecastResult.model_construct(idata=idata, steps=steps, var_names=names)


def _make_irf_result(n_vars=2, horizon=10) -> IRFResult:
    rng = np.random.default_rng(42)
    names = [f"y{i + 1}" for i in range(n_vars)]
    data = rng.standard_normal((2, 50, horizon + 1, n_vars, n_vars))
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "horizon", "response", "shock"],
        coords={"response": names, "shock": names, "horizon": np.arange(horizon + 1)},
        name="irf",
    )
    idata = az.InferenceData(posterior_predictive=xr.Dataset({"irf": da}))
    return IRFResult.model_construct(idata=idata, horizon=horizon, var_names=names)


def _make_fevd_result(n_vars=2, horizon=10) -> FEVDResult:
    rng = np.random.default_rng(42)
    names = [f"y{i + 1}" for i in range(n_vars)]
    # FEVD shares should sum to 1 across shocks
    raw = np.abs(rng.standard_normal((2, 50, horizon + 1, n_vars, n_vars)))
    raw = raw / raw.sum(axis=-1, keepdims=True)
    da = xr.DataArray(
        raw,
        dims=["chain", "draw", "horizon", "response", "shock"],
        coords={"response": names, "shock": names},
        name="fevd",
    )
    idata = az.InferenceData(posterior_predictive=xr.Dataset({"fevd": da}))
    return FEVDResult.model_construct(idata=idata, horizon=horizon, var_names=names)


def _make_hd_result(n_vars=2, T=20, shock_names=None) -> HistoricalDecompositionResult:
    rng = np.random.default_rng(42)
    names = [f"y{i + 1}" for i in range(n_vars)]
    shocks = list(shock_names) if shock_names is not None else names
    data = rng.standard_normal((2, 50, T, n_vars, len(shocks)))
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "time", "response", "shock"],
        coords={"response": names, "shock": shocks},
        name="hd",
    )
    idata = az.InferenceData(posterior_predictive=xr.Dataset({"hd": da}))
    return HistoricalDecompositionResult.model_construct(idata=idata, var_names=names)


class TestPlotForecast:
    def test_returns_figure(self):
        result = _make_forecast_result()
        fig = plot_forecast(result)
        assert isinstance(fig, Figure)

    def test_has_correct_axes(self):
        result = _make_forecast_result(n_vars=3)
        fig = plot_forecast(result)
        assert len(fig.axes) == 3

    def test_title(self):
        result = _make_forecast_result()
        fig = plot_forecast(result)
        assert fig._suptitle.get_text() == "Forecast"


class TestPlotIRF:
    def test_returns_figure(self):
        result = _make_irf_result()
        fig = plot_irf(result)
        assert isinstance(fig, Figure)

    def test_has_correct_axes(self):
        result = _make_irf_result(n_vars=2)
        fig = plot_irf(result)
        assert len(fig.axes) == 4  # 2x2 grid

    def test_title(self):
        result = _make_irf_result()
        fig = plot_irf(result)
        assert fig._suptitle.get_text() == "Impulse Response Functions"


class TestPlotFEVD:
    def test_returns_figure(self):
        result = _make_fevd_result()
        fig = plot_fevd(result)
        assert isinstance(fig, Figure)

    def test_has_correct_axes(self):
        result = _make_fevd_result(n_vars=3)
        fig = plot_fevd(result)
        assert len(fig.axes) == 3

    def test_title(self):
        result = _make_fevd_result()
        fig = plot_fevd(result)
        assert fig._suptitle.get_text() == "Forecast Error Variance Decomposition"


class TestPlotHistoricalDecomposition:
    def test_returns_figure(self):
        result = _make_hd_result()
        fig = plot_historical_decomposition(result)
        assert isinstance(fig, Figure)

    def test_has_correct_axes(self):
        result = _make_hd_result(n_vars=3)
        fig = plot_historical_decomposition(result)
        assert len(fig.axes) == 3

    def test_title(self):
        result = _make_hd_result()
        fig = plot_historical_decomposition(result)
        assert fig._suptitle.get_text() == "Historical Decomposition"

    def test_legend_labels_come_from_shock_coord(self):
        """Partial identification: labels read the shock coord, not var_names."""
        result = _make_hd_result(n_vars=3, shock_names=["target", "unidentified_remainder"])
        fig = plot_historical_decomposition(result)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        # matplotlib collects ax.lines before bar containers in the legend.
        assert labels == ["deviation from baseline", "target", "unidentified_remainder"]

    def test_deviation_line_matches_median_of_sum(self):
        """The overlay is the median of the per-draw shock sum (= data - baseline)."""
        result = _make_hd_result()
        fig = plot_historical_decomposition(result)
        expected = result.idata.posterior_predictive["hd"].sum("shock").median(dim=("chain", "draw"))
        for i, resp in enumerate(result.var_names):
            lines = [ln for ln in fig.axes[i].get_lines() if ln.get_label() == "deviation from baseline"]
            assert len(lines) == 1
            np.testing.assert_allclose(lines[0].get_ydata(), expected.sel(response=resp).values, atol=1e-12)


class TestPlotStability:
    @staticmethod
    def _summary(make_var_posterior, **kwargs):
        from impulso.diagnostics import convergence_report

        return convergence_report(make_var_posterior(**kwargs), n_lags=1).stability

    @staticmethod
    def _line(ax, label):
        lines = [line for line in ax.get_lines() if line.get_label() == label]
        assert len(lines) == 1
        return lines[0]

    def test_returns_two_panels(self, make_var_posterior):
        fig = plot_stability(self._summary(make_var_posterior))
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert fig._suptitle.get_text() == "Dynamic stability"

    def test_histogram_bin_count_is_the_argument(self, make_var_posterior):
        summary = self._summary(make_var_posterior)
        assert len(plot_stability(summary).axes[0].patches) == 40
        assert len(plot_stability(summary, bins=15).axes[0].patches) == 15

    def test_histogram_covers_every_radius_draw(self, make_var_posterior):
        summary = self._summary(make_var_posterior)
        ax = plot_stability(summary).axes[0]
        assert sum(patch.get_height() for patch in ax.patches) == summary.radius.size

    def test_unit_root_line_sits_at_one(self, make_var_posterior):
        ax = plot_stability(self._summary(make_var_posterior)).axes[0]
        np.testing.assert_array_equal(self._line(ax, UNIT_ROOT_LABEL).get_xdata(), [1.0, 1.0])

    def test_unit_circle_has_modulus_one(self, make_var_posterior):
        ax = plot_stability(self._summary(make_var_posterior)).axes[1]
        line = self._line(ax, UNIT_CIRCLE_LABEL)
        np.testing.assert_allclose(np.hypot(line.get_xdata(), line.get_ydata()), 1.0)

    def test_scatter_plots_every_retained_eigenvalue(self, make_var_posterior):
        summary = self._summary(make_var_posterior)
        ax = plot_stability(summary).axes[1]
        assert len(ax.collections) == 1
        offsets = ax.collections[0].get_offsets()
        assert offsets.shape == (summary.eigenvalues.size, 2)
        np.testing.assert_allclose(np.sort(offsets[:, 0]), np.sort(summary.eigenvalues.real.reshape(-1)))

    def test_scatter_axes_are_equally_scaled(self, make_var_posterior):
        ax = plot_stability(self._summary(make_var_posterior)).axes[1]
        assert ax.get_aspect() == 1.0

    def test_explosive_mass_reaches_the_histogram_title(self, make_var_posterior):
        fig = plot_stability(self._summary(make_var_posterior, explosive_frac=0.15))
        assert "15.0%" in fig.axes[0].get_title()

    def test_summary_plot_method_delegates(self, make_var_posterior):
        # The house entry point: `report.stability.plot()`.
        fig = self._summary(make_var_posterior).plot()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2


def test_plot_volatility_returns_figure(synthetic_sv_idata):
    import pandas as pd
    from matplotlib.figure import Figure

    from impulso.results import VolatilityResult

    result = VolatilityResult(
        idata=synthetic_sv_idata,
        series_name="sim",
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    fig = result.plot()
    assert isinstance(fig, Figure)


def _make_sv_forecast_result(steps=12, index=None):
    from impulso.results import SVForecastResult

    rng = np.random.default_rng(0)
    forecast = rng.standard_normal((2, 50, steps))
    idata = az.InferenceData(
        posterior_predictive=xr.Dataset({"forecast": xr.DataArray(forecast, dims=["chain", "draw", "step"])})
    )
    return SVForecastResult(idata=idata, series_name="sim", steps=steps, index=index)


def test_plot_sv_forecast_returns_figure():
    result = _make_sv_forecast_result()
    fig = result.plot()
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Step ahead"
    np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), np.arange(12))


def test_plot_sv_forecast_uses_calendar_axis():
    import matplotlib.dates as mdates
    import pandas as pd

    idx = pd.date_range("2008-05-01", periods=12, freq="MS")
    fig = _make_sv_forecast_result(index=idx).plot()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Date"
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(orig=False), mdates.date2num(idx))
