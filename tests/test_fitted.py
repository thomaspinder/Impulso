"""Tests for FittedVAR."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.samplers import NUTSSampler
from impulso.spec import VAR

# var_data_2v comes from conftest.py


@pytest.fixture
def var_data(var_data_2v):
    return var_data_2v


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


class TestFittedVARFast:
    """Fast tests using synthetic InferenceData (no MCMC)."""

    def test_properties_from_synthetic(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        assert fitted.n_lags == 1
        assert fitted.has_exog is False
        assert fitted.coefficients.shape == (2, 50, 2, 2)
        assert fitted.intercepts.shape == (2, 50, 2)
        assert fitted.sigma.shape == (2, 50, 2, 2)

    def test_repr_from_synthetic(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        r = repr(fitted)
        assert "FittedVAR" in r
        assert "n_lags=1" in r

    def test_forecast_shape(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        result = fitted.forecast(steps=4)
        med = result.median()
        assert med.shape == (4, 2)

    def test_forecast_hdi(self, synthetic_idata_2v, var_data_2v):
        from impulso.results import HDIResult

        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        result = fitted.forecast(steps=4)
        hdi = result.hdi(prob=0.89)
        assert isinstance(hdi, HDIResult)
        assert hdi.lower.shape == (4, 2)
        assert hdi.upper.shape == (4, 2)

    def test_default_forecast_matches_conditional_mean_when_simulation_disabled(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )

        default = fitted.forecast(steps=4).idata.posterior_predictive["forecast"]
        explicit = fitted.forecast(steps=4, simulate_innovations=False).idata.posterior_predictive["forecast"]

        np.testing.assert_allclose(default.values, explicit.values)

    def test_forecast_innovation_simulation_is_reproducible(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )

        first = fitted.forecast(steps=4, simulate_innovations=True, random_seed=123).idata.posterior_predictive[
            "forecast"
        ]
        second = fitted.forecast(steps=4, simulate_innovations=True, random_seed=123).idata.posterior_predictive[
            "forecast"
        ]
        third = fitted.forecast(steps=4, simulate_innovations=True, random_seed=456).idata.posterior_predictive[
            "forecast"
        ]

        np.testing.assert_allclose(first.values, second.values)
        assert not np.allclose(first.values, third.values)

    def test_forecast_innovation_simulation_adds_forecast_dispersion(self):
        n_draws = 100
        posterior = xr.Dataset({
            "B": xr.DataArray(np.zeros((1, n_draws, 2, 2)), dims=["chain", "draw", "var", "coeff"]),
            "intercept": xr.DataArray(np.zeros((1, n_draws, 2)), dims=["chain", "draw", "var"]),
            "Sigma": xr.DataArray(
                np.broadcast_to(np.eye(2), (1, n_draws, 2, 2)).copy(),
                dims=["chain", "draw", "var1", "var2"],
            ),
        })
        data = VARData(
            endog=np.zeros((3, 2)),
            endog_names=["y1", "y2"],
            index=pd.date_range("2000-01-01", periods=3, freq="MS"),
        )
        fitted = FittedVAR.model_construct(
            idata=az.InferenceData(posterior=posterior),
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
        )

        deterministic = fitted.forecast(steps=3).idata.posterior_predictive["forecast"].values
        simulated = (
            fitted
            .forecast(steps=3, simulate_innovations=True, random_seed=123)
            .idata.posterior_predictive["forecast"]
            .values
        )

        assert float(deterministic.std()) == 0.0
        assert float(simulated.std()) > 0.0

    def test_forecast_innovation_simulation_supports_exog_future(self):
        n_draws = 10
        posterior = xr.Dataset({
            "B": xr.DataArray(np.zeros((1, n_draws, 2, 2)), dims=["chain", "draw", "var", "coeff"]),
            "B_exog": xr.DataArray(np.zeros((1, n_draws, 2, 1)), dims=["chain", "draw", "var", "exog"]),
            "intercept": xr.DataArray(np.zeros((1, n_draws, 2)), dims=["chain", "draw", "var"]),
            "Sigma": xr.DataArray(
                np.broadcast_to(np.eye(2), (1, n_draws, 2, 2)).copy(),
                dims=["chain", "draw", "var1", "var2"],
            ),
        })
        data = VARData(
            endog=np.zeros((3, 2)),
            endog_names=["y1", "y2"],
            exog=np.zeros((3, 1)),
            exog_names=["x1"],
            index=pd.date_range("2000-01-01", periods=3, freq="MS"),
        )
        fitted = FittedVAR.model_construct(
            idata=az.InferenceData(posterior=posterior),
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
        )

        result = fitted.forecast(
            steps=3,
            exog_future=np.zeros((3, 1)),
            simulate_innovations=True,
            random_seed=123,
        )

        assert result.idata.posterior_predictive["forecast"].shape == (1, n_draws, 3, 2)
