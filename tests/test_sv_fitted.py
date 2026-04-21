"""Tests for FittedSV (fast path, no MCMC)."""

import numpy as np
import pandas as pd
import pytest

from impulso.results import VolatilityResult
from impulso.sv.data import SVData
from impulso.sv.fitted import FittedSV


@pytest.fixture
def fitted_sv(synthetic_sv_idata):
    index = pd.date_range("2000-01-01", periods=100, freq="MS")
    # Non-constant series: SVData rejects constant input (see validator in Task 1).
    # FittedSV post-fitting logic does not depend on the series values — any
    # non-constant path of the right length is sufficient.
    y = np.linspace(-0.1, 0.1, 100)
    data = SVData(y=y, name="sim", index=index)
    return FittedSV(
        idata=synthetic_sv_idata,
        data=data,
        dynamics="random_walk",
    )


def test_fittedsv_log_volatility_shape(fitted_sv):
    lv = fitted_sv.log_volatility
    assert lv.shape == (2, 50, 100)


def test_fittedsv_volatility_returns_volatilityresult(fitted_sv):
    result = fitted_sv.volatility()
    assert isinstance(result, VolatilityResult)
    assert result.series_name == "sim"


def test_fittedsv_forecast_shape_and_type(fitted_sv):
    from impulso.results import SVForecastResult

    result = fitted_sv.forecast(steps=12)
    assert isinstance(result, SVForecastResult)
    assert result.steps == 12
    forecast = result.idata.posterior_predictive["forecast"].values
    assert forecast.shape == (2, 50, 12)
