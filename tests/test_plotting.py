"""Tests for plotting functions."""

import arviz as az
import matplotlib
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

matplotlib.use("Agg")

from impulso.plotting import plot_fevd, plot_forecast, plot_historical_decomposition, plot_irf
from impulso.results import FEVDResult, ForecastResult, HistoricalDecompositionResult, IRFResult


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


def _make_hd_result(n_vars=2, T=20) -> HistoricalDecompositionResult:
    rng = np.random.default_rng(42)
    names = [f"y{i + 1}" for i in range(n_vars)]
    data = rng.standard_normal((2, 50, T, n_vars, n_vars))
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "time", "response", "shock"],
        coords={"response": names, "shock": names},
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


def test_plot_sv_forecast_returns_figure():
    import arviz as az
    import numpy as np
    import xarray as xr
    from matplotlib.figure import Figure

    from impulso.results import SVForecastResult

    rng = np.random.default_rng(0)
    steps = 12
    forecast = rng.standard_normal((2, 50, steps))
    idata = az.InferenceData(
        posterior_predictive=xr.Dataset({"forecast": xr.DataArray(forecast, dims=["chain", "draw", "step"])})
    )
    result = SVForecastResult(idata=idata, series_name="sim", steps=steps)
    fig = result.plot()
    assert isinstance(fig, Figure)
