"""Tests for FittedVAR."""

import numpy as np
import pandas as pd
import pytest
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.samplers import NUTSSampler
from impulso.spec import VAR


@pytest.fixture
def var_data():
    """Simple VAR(1) DGP."""
    rng = np.random.default_rng(42)
    T = 200
    n = 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


class TestFittedVAR:
    @pytest.mark.slow
    def test_fit_returns_fitted_var(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert isinstance(result, FittedVAR)

    @pytest.mark.slow
    def test_fitted_properties(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert result.n_lags == 1
        assert result.has_exog is False
        assert result.idata is not None

    @pytest.mark.slow
    def test_fit_with_auto_lags(self, var_data):
        spec = VAR(lags="bic", max_lags=4, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert result.n_lags >= 1

    @pytest.mark.slow
    def test_repr_is_compact(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        r = repr(result)
        assert "FittedVAR" in r
        assert "n_lags=1" in r


class TestForecasting:
    @pytest.mark.slow
    def test_forecast_returns_forecast_result(self, var_data):
        from impulso.results import ForecastResult

        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=4)
        assert isinstance(fcast, ForecastResult)
        assert fcast.steps == 4

    @pytest.mark.slow
    def test_forecast_median_shape(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=8)
        med = fcast.median()
        assert med.shape == (8, 2)

    @pytest.mark.slow
    def test_forecast_hdi_returns_hdi_result(self, var_data):
        from impulso.results import HDIResult

        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=4)
        hdi = fcast.hdi(prob=0.89)
        assert isinstance(hdi, HDIResult)

    @pytest.mark.slow
    def test_forecast_exog_required_error(self):
        """If model has exog, forecast without exog_future raises."""
        rng = np.random.default_rng(42)
        T, n = 200, 2
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
        exog = rng.standard_normal((T, 1))
        index = pd.date_range("2000-01-01", periods=T, freq="QS")
        data = VARData(endog=y, endog_names=["y1", "y2"], exog=exog, exog_names=["x1"], index=index)

        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(data, sampler=sampler)
        with pytest.raises(ValueError, match="exog_future"):
            fitted.forecast(steps=4)
