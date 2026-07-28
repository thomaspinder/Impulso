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


def _build_ar1_fitted_sv(index: pd.DatetimeIndex | None = None):
    """Build a FittedSV with synthetic AR(1) posterior, no MCMC.

    Args:
        index: Optional DatetimeIndex of length 50. Defaults to a monthly
            range; pass an irregular index to exercise the undated path.
    """
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
    if index is None:
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


def test_fittedsv_forecast_index_continues_calendar(fitted_sv):
    """A dated, regular sample yields a calendar forecast axis."""
    result = fitted_sv.forecast(steps=12)

    # fitted_sv spans 100 monthly starts from 2000-01-01, i.e. up to 2008-04-01.
    expected = pd.date_range("2008-05-01", periods=12, freq="MS")
    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.index) == list(expected)
    assert result.index.freqstr == "MS"


def test_fittedsv_forecast_index_carried_by_frames(fitted_sv):
    """median/to_dataframe/hdi all carry the calendar axis."""
    result = fitted_sv.forecast(steps=6)
    expected = pd.date_range("2008-05-01", periods=6, freq="MS")

    for df in (result.median(), result.to_dataframe(), result.hdi().lower, result.hdi().upper):
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.index) == list(expected)
        assert df.index.name == "time"


def test_fittedsv_forecast_index_infers_freq_when_attribute_missing():
    """A regular index whose `.freq` was dropped is still recognised."""
    index = pd.DatetimeIndex(list(pd.date_range("2000-01-01", periods=50, freq="MS")))
    assert index.freq is None

    result = _build_ar1_fitted_sv(index=index).forecast(steps=3, random_seed=0)
    assert list(result.index) == list(pd.date_range("2004-03-01", periods=3, freq="MS"))


def test_fittedsv_forecast_falls_back_to_step_index():
    """An irregular calendar has no detectable frequency — step axis."""
    irregular = pd.date_range("2000-01-01", periods=60, freq="MS").delete([5, 11, 17, 23, 29, 35, 41, 47, 53, 59])
    assert pd.infer_freq(irregular) is None

    result = _build_ar1_fitted_sv(index=irregular).forecast(steps=4, random_seed=0)
    assert isinstance(result.index, pd.RangeIndex)
    assert list(result.index) == [0, 1, 2, 3]
    med = result.median()
    assert med.index.name == "step"
    assert list(med.index) == [0, 1, 2, 3]


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
