"""Tests for FittedVAR."""

import numpy as np
import pandas as pd
import pytest

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
        from impulso.volatility import Constant

        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        assert fitted.n_lags == 1
        assert fitted.has_exog is False
        assert fitted.coefficients.shape == (2, 50, 2, 2)
        assert fitted.intercepts.shape == (2, 50, 2)
        assert fitted.sigma().shape == (2, 50, 2, 2)

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
        result = fitted.forecast(steps=4, include_shock_uncertainty=False)
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
        result = fitted.forecast(steps=4, include_shock_uncertainty=False)
        hdi = result.hdi(prob=0.89)
        assert isinstance(hdi, HDIResult)
        assert hdi.lower.shape == (4, 2)
        assert hdi.upper.shape == (4, 2)


class TestFittedVARVolatility:
    def test_fitted_var_carries_volatility(self, var_data_2v):
        """FittedVAR must expose the volatility process used at fit time."""
        from impulso.protocols import VolatilityProcess
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR
        from impulso.volatility import Constant

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)

        assert isinstance(fitted.volatility, VolatilityProcess)
        assert isinstance(fitted.volatility, Constant)
        assert fitted.volatility.name == "constant"

    def test_fitted_var_volatility_round_trips_explicit(self, var_data_2v):
        """A custom Constant() instance passed to VAR is preserved on FittedVAR."""
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR
        from impulso.volatility import Constant

        custom = Constant(sigma_sd_beta=3.0)
        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1, volatility=custom).fit(var_data_2v, sampler=sampler)

        assert fitted.volatility is custom
        assert fitted.volatility.sigma_sd_beta == 3.0


class TestSetIdentificationStrategyRoutesThroughSeam:
    def test_calls_volatility_cholesky_at(self, var_data_2v):
        """set_identification_strategy must query volatility.cholesky_at,
        not read posterior['Sigma'] directly."""
        from unittest.mock import MagicMock

        from impulso.identification import Cholesky
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)

        # Replace the volatility on the fitted spec with a spy.
        spy = MagicMock(wraps=fitted.volatility)
        object.__setattr__(fitted, "volatility", spy)

        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        spy.cholesky_at.assert_called_once()
        # Confirm the structural shock matrix made it into the posterior.
        assert "structural_shock_matrix" in identified.idata.posterior
        # Lock down the dims/coords contract that downstream IRF/FEVD/HD depend on.
        ssm = identified.idata.posterior["structural_shock_matrix"]
        assert ssm.dims == ("chain", "draw", "response", "shock")
        assert list(ssm.coords["response"].values) == fitted.var_names
        assert list(ssm.coords["shock"].values) == fitted.var_names


class TestFittedVarSigmaDispatch:
    """`FittedVAR.sigma()` dispatches by volatility adapter type."""

    @pytest.mark.slow
    def test_constant_sigma_unchanged(self, var_data_2v):
        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        sigma = fitted.sigma()
        # For Constant: (chains, draws, n_vars, n_vars).
        assert sigma.shape == (1, 20, 2, 2)

    @pytest.mark.slow
    def test_sv_sigma_returns_per_t(self, var_data_2v):
        from impulso.sv.spec import StochasticVolatility

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1, volatility=StochasticVolatility()).fit(var_data_2v, sampler=sampler)
        sigma_path = fitted.sigma()
        # For SV: (chains, draws, T, n_vars, n_vars).
        T = var_data_2v.endog.shape[0] - 1  # n_lags=1
        assert sigma_path.shape == (1, 20, T, 2, 2)
