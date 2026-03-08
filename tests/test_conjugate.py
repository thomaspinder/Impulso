"""Tests for ConjugateVAR (direct NIW posterior sampling)."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from impulso.data import VARData
from impulso.priors import MinnesotaPrior


@pytest.fixture
def stable_var_data():
    """VAR(1) DGP with known stable coefficients."""
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


class TestConjugateVARConstruction:
    def test_basic_construction(self):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=2)
        assert cvar.lags == 2
        assert cvar.draws == 2000

    def test_custom_prior(self):
        from impulso.conjugate import ConjugateVAR

        prior = MinnesotaPrior(tightness=0.2, cross_shrinkage=0.3)
        cvar = ConjugateVAR(lags=2, prior=prior)
        assert cvar.prior == prior

    def test_frozen(self):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=2)
        with pytest.raises(ValidationError):
            cvar.lags = 4

    def test_rejects_negative_draws(self):
        from impulso.conjugate import ConjugateVAR

        with pytest.raises(ValidationError):
            ConjugateVAR(lags=2, draws=0)


class TestConjugateVARFit:
    def test_fit_returns_fitted_var(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR
        from impulso.fitted import FittedVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert isinstance(fitted, FittedVAR)

    def test_idata_has_required_variables(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert "B" in fitted.idata.posterior
        assert "intercept" in fitted.idata.posterior
        assert "Sigma" in fitted.idata.posterior

    def test_posterior_shapes(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        n_draws = 100
        cvar = ConjugateVAR(lags=2, draws=n_draws, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        B = fitted.idata.posterior["B"].values
        assert B.shape == (1, n_draws, 2, 4)
        intercept = fitted.idata.posterior["intercept"].values
        assert intercept.shape == (1, n_draws, 2)
        sigma = fitted.idata.posterior["Sigma"].values
        assert sigma.shape == (1, n_draws, 2, 2)

    def test_sigma_positive_definite(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        sigma = fitted.idata.posterior["Sigma"].values
        for d in range(sigma.shape[1]):
            eigvals = np.linalg.eigvalsh(sigma[0, d])
            assert np.all(eigvals > 0), f"Draw {d} has non-positive eigenvalue"

    def test_sigma_symmetric(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        sigma = fitted.idata.posterior["Sigma"].values
        np.testing.assert_allclose(sigma, np.swapaxes(sigma, -2, -1), atol=1e-10)

    def test_reproducible_with_seed(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar1 = ConjugateVAR(lags=1, draws=50, random_seed=123)
        cvar2 = ConjugateVAR(lags=1, draws=50, random_seed=123)
        fitted1 = cvar1.fit(stable_var_data)
        fitted2 = cvar2.fit(stable_var_data)
        np.testing.assert_array_equal(
            fitted1.idata.posterior["B"].values,
            fitted2.idata.posterior["B"].values,
        )

    def test_var_names_correct(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert fitted.var_names == ["y1", "y2"]

    def test_n_lags_stored(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=3, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert fitted.n_lags == 3

    def test_downstream_forecast_works(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        result = fitted.forecast(steps=4)
        assert result.median().shape == (4, 2)

    def test_downstream_identification_works(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR
        from impulso.identification import Cholesky

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        irfs = identified.impulse_response(horizon=10)
        assert irfs.median().shape[0] == 11

    def test_lag_selection_string(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags="bic", draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert fitted.n_lags >= 1

    def test_works_with_dummy_observations(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        augmented = stable_var_data.with_dummy_observations(n_lags=2, mu=5.0, delta=1.0)
        cvar = ConjugateVAR(lags=2, draws=50, random_seed=42)
        fitted = cvar.fit(augmented)
        assert fitted.idata.posterior["B"].values.shape == (1, 50, 2, 4)
