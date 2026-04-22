"""Tests for FittedSV (fast path, no MCMC)."""

import numpy as np
import pandas as pd
import pytest

from impulso.results import VolatilityResult
from impulso.sv.data import SVData
from impulso.sv.dynamics import AR1, RandomWalk
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
        dynamics=RandomWalk(),
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


def _build_ar1_fitted_sv():
    """Build a FittedSV with synthetic AR(1) posterior, no MCMC."""
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(0)
    n_chains, n_draws, T = 2, 50, 50
    h = 0.1 * rng.standard_normal((n_chains, n_draws, T))
    mu = 0.01 * rng.standard_normal((n_chains, n_draws))
    sigma_eta = 0.1 * np.ones((n_chains, n_draws))
    phi = 0.9 + 0.05 * rng.standard_normal((n_chains, n_draws))
    alpha = 0.0 + 0.01 * rng.standard_normal((n_chains, n_draws))

    posterior = xr.Dataset({
        "h": xr.DataArray(h, dims=["chain", "draw", "time"]),
        "mu": xr.DataArray(mu, dims=["chain", "draw"]),
        "sigma_eta": xr.DataArray(sigma_eta, dims=["chain", "draw"]),
        "phi": xr.DataArray(phi, dims=["chain", "draw"]),
        "alpha": xr.DataArray(alpha, dims=["chain", "draw"]),
    })
    idata = az.InferenceData(posterior=posterior)

    y = rng.standard_normal(T)
    index = pd.date_range("2000-01-01", periods=T, freq="MS")
    data = SVData(y=y, name="sim", index=index)

    return FittedSV.model_construct(idata=idata, data=data, dynamics=AR1())


def test_fittedsv_forecast_ar1_shape_and_type():
    from impulso.results import SVForecastResult

    fitted = _build_ar1_fitted_sv()
    result = fitted.forecast(steps=6, random_seed=42)
    assert isinstance(result, SVForecastResult)
    assert result.steps == 6
    assert result.series_name == "sim"

    forecast = result.idata.posterior_predictive["forecast"].values
    assert forecast.shape == (2, 50, 6)

    med = result.median()
    assert med.shape[0] == 6
    hdi = result.hdi()
    assert hdi.lower.shape[0] == 6
    assert hdi.upper.shape[0] == 6


def test_fittedsv_forecast_rng_seeds_are_respected():
    fitted = _build_ar1_fitted_sv()

    r1 = fitted.forecast(steps=6, random_seed=42)
    r2 = fitted.forecast(steps=6, random_seed=42)
    r3 = fitted.forecast(steps=6, random_seed=7)

    f1 = r1.idata.posterior_predictive["forecast"].values
    f2 = r2.idata.posterior_predictive["forecast"].values
    f3 = r3.idata.posterior_predictive["forecast"].values

    # Same seed -> identical draws.
    np.testing.assert_array_equal(f1, f2)
    # Different seed -> different draws.
    assert not np.array_equal(f1, f3)
