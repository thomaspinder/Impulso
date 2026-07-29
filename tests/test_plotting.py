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


def _make_pool(weights, log_score_columns):
    """Hand-build a two-model PredictivePool without running the sampler."""
    import pandas as pd

    from impulso.pooling import PredictivePool

    labels = ["a", "b"]
    index = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=3, freq="QS"), name="time")
    log_scores = pd.DataFrame(np.column_stack(log_score_columns), index=index, columns=labels)
    draws = np.zeros((4, 3, 2))
    da = xr.DataArray(
        draws[np.newaxis],
        dims=["chain", "draw", "step", "variable"],
        coords={"variable": ["y1", "y2"], "model": ("draw", np.array(["a", "a", "b", "b"], dtype=object))},
        name="forecast",
    )
    return PredictivePool(
        weights=pd.Series(weights, index=labels),
        log_scores=log_scores,
        method="stacking",
        density="gaussian",
        var_names=["y1", "y2"],
        steps=3,
        origin=pd.Timestamp("2019-10-01"),
        holdout_predictive=ForecastResult(
            idata=az.InferenceData(posterior_predictive=xr.Dataset({"forecast": da})),
            steps=3,
            var_names=["y1", "y2"],
        ),
        membership=np.array([0, 0, 1, 1]),
    )


def test_plot_pool_weights_returns_figure():
    """Smoke test: the pool bar chart renders from a hand-built pool."""
    from matplotlib.figure import Figure

    from impulso.plotting import plot_pool_weights

    pool = _make_pool([0.7, 0.3], ([-1.0, -1.5, -0.5], [-2.0, -2.5, -3.0]))
    assert isinstance(plot_pool_weights(pool), Figure)


def test_plot_pool_weights_annotation_placement():
    """Short bars label outside; a near-full bar labels inside, right-aligned (#166)."""
    from impulso.plotting import plot_pool_weights

    # 'a' takes almost all the weight and scores badly enough that its label is
    # long, which is the case that used to overflow the right axis.
    pool = _make_pool([0.98, 0.02], ([-500.0, -400.0, -334.5], [-2.0, -2.5, -3.0]))
    fig = plot_pool_weights(pool)
    fig.canvas.draw()
    ax = fig.axes[0]
    upper = ax.get_xlim()[1]
    renderer = fig.canvas.get_renderer()
    texts = {text.get_text(): text for text in ax.texts}

    inside = texts["log score -1,234.5"]
    assert inside.get_horizontalalignment() == "right"
    assert inside.get_position()[0] < 0.98
    # The whole label must sit within the axes, which is what fails when a long
    # annotation is parked to the right of a bar that already fills the width.
    assert inside.get_window_extent(renderer).x1 <= ax.bbox.x1

    outside = texts["log score -7.5"]
    assert outside.get_horizontalalignment() == "left"
    assert 0.02 < outside.get_position()[0] < upper
    assert outside.get_window_extent(renderer).x1 <= ax.bbox.x1


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
