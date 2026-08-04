"""Tests for SV result types."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso._arviz_compat import InferenceDataLike, make_idata
from impulso._time import forecast_index, infer_index_freq
from impulso.results import SVForecastResult, VolatilityResult


def test_volatility_result_median_shape(synthetic_sv_idata):
    result = VolatilityResult(
        idata=synthetic_sv_idata,
        series_name="sim",
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    med = result.median()
    assert isinstance(med, pd.DataFrame)
    assert med.shape == (100, 1)
    assert "sim" in med.columns


def test_volatility_result_median_positive(synthetic_sv_idata):
    """exp(h/2) must be strictly positive."""
    result = VolatilityResult(
        idata=synthetic_sv_idata,
        series_name="sim",
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    assert (result.median().values > 0).all()


def test_volatility_result_hdi_bounds_ordered(synthetic_sv_idata):
    result = VolatilityResult(
        idata=synthetic_sv_idata,
        series_name="sim",
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    hdi_out = result.hdi(prob=0.89)
    assert hdi_out.prob == 0.89
    assert (hdi_out.lower.values <= hdi_out.upper.values).all()


def test_volatility_result_to_dataframe_has_index(synthetic_sv_idata):
    idx = pd.date_range("2000-01-01", periods=100, freq="MS")
    result = VolatilityResult(idata=synthetic_sv_idata, series_name="sim", index=idx)
    df = result.to_dataframe()
    assert list(df.index) == list(idx)


# --------------- SVForecastResult forecast axis ---------------


def _sv_forecast_idata(steps: int = 12) -> InferenceDataLike:
    """Synthetic posterior-predictive draws for a univariate SV forecast."""
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((2, 50, steps))
    return make_idata(
        posterior_predictive=xr.Dataset({"forecast": xr.DataArray(draws, dims=["chain", "draw", "step"])})
    )


def test_sv_forecast_result_defaults_to_step_index():
    """No index supplied — the historical RangeIndex/step behaviour holds."""
    result = SVForecastResult(idata=_sv_forecast_idata(5), series_name="sim", steps=5)
    for df in (result.median(), result.to_dataframe(), result.hdi().lower, result.hdi().upper):
        assert isinstance(df.index, pd.RangeIndex)
        assert list(df.index) == [0, 1, 2, 3, 4]
        assert df.index.name == "step"


def test_sv_forecast_result_carries_datetime_index():
    idx = pd.date_range("2008-05-01", periods=5, freq="MS")
    result = SVForecastResult(idata=_sv_forecast_idata(5), series_name="sim", steps=5, index=idx)
    for df in (result.median(), result.to_dataframe(), result.hdi(prob=0.5).lower, result.hdi(prob=0.5).upper):
        assert list(df.index) == list(idx)
    assert result.median().columns.tolist() == ["sim"]


def test_sv_forecast_result_rejects_index_length_mismatch():
    idx = pd.date_range("2008-05-01", periods=4, freq="MS")
    with pytest.raises(ValidationError, match="index length 4 != steps 5"):
        SVForecastResult(idata=_sv_forecast_idata(5), series_name="sim", steps=5, index=idx)


# --------------- forecast-axis helper ---------------


def test_forecast_index_continues_calendar():
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    out = forecast_index(idx, 3)
    assert list(out) == list(pd.date_range("2002-01-01", periods=3, freq="MS"))
    assert out.name == "time"


def test_forecast_index_handles_quarterly_and_daily():
    quarterly = forecast_index(pd.date_range("2000-01-01", periods=8, freq="QS"), 2)
    assert list(quarterly) == list(pd.date_range("2002-01-01", periods=2, freq="QS"))

    daily = forecast_index(pd.date_range("2000-01-01", periods=8, freq="D"), 2)
    assert list(daily) == [pd.Timestamp("2000-01-09"), pd.Timestamp("2000-01-10")]


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(None, id="none"),
        pytest.param(pd.RangeIndex(10), id="non-datetime"),
        pytest.param(pd.DatetimeIndex([]), id="empty"),
        pytest.param(pd.DatetimeIndex(["2000-01-01", "2000-03-01"]), id="too-short-to-infer"),
        pytest.param(pd.DatetimeIndex(["2000-01-01", "2000-02-05", "2000-04-17"]), id="irregular"),
    ],
)
def test_forecast_index_falls_back_to_steps(idx):
    out = forecast_index(idx, 3)
    assert isinstance(out, pd.RangeIndex)
    assert list(out) == [0, 1, 2]
    assert out.name == "step"
    assert infer_index_freq(idx) is None
