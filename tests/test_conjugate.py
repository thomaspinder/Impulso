"""Tests for ConjugateVAR (direct NIW posterior sampling)."""

import numpy as np
import pytest
from pydantic import ValidationError

from impulso.priors import MinnesotaPrior


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
    def test_fit_returns_fitted_var(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR
        from impulso.fitted import FittedVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        assert isinstance(fitted, FittedVAR)

    def test_idata_has_required_variables(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        assert "B" in fitted.idata.posterior
        assert "intercept" in fitted.idata.posterior
        assert "Sigma" in fitted.idata.posterior

    def test_posterior_shapes(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        n_draws = 100
        cvar = ConjugateVAR(lags=2, draws=n_draws, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        B = fitted.idata.posterior["B"].values
        assert B.shape == (1, n_draws, 2, 4)
        intercept = fitted.idata.posterior["intercept"].values
        assert intercept.shape == (1, n_draws, 2)
        sigma = fitted.idata.posterior["Sigma"].values
        assert sigma.shape == (1, n_draws, 2, 2)

    def test_sigma_positive_definite(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        sigma = fitted.idata.posterior["Sigma"].values
        for d in range(sigma.shape[1]):
            eigvals = np.linalg.eigvalsh(sigma[0, d])
            assert np.all(eigvals > 0), f"Draw {d} has non-positive eigenvalue"

    def test_sigma_symmetric(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        sigma = fitted.idata.posterior["Sigma"].values
        np.testing.assert_allclose(sigma, np.swapaxes(sigma, -2, -1), atol=1e-10)

    def test_reproducible_with_seed(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar1 = ConjugateVAR(lags=1, draws=50, random_seed=123)
        cvar2 = ConjugateVAR(lags=1, draws=50, random_seed=123)
        fitted1 = cvar1.fit(var_data_2v)
        fitted2 = cvar2.fit(var_data_2v)
        np.testing.assert_array_equal(
            fitted1.idata.posterior["B"].values,
            fitted2.idata.posterior["B"].values,
        )

    def test_var_names_correct(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        assert fitted.var_names == ["y1", "y2"]

    def test_n_lags_stored(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=3, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        assert fitted.n_lags == 3

    def test_downstream_forecast_works(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        result = fitted.forecast(steps=4)
        assert result.median().shape == (4, 2)

    def test_downstream_identification_works(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR
        from impulso.identification import Cholesky

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        irfs = identified.impulse_response(horizon=10)
        assert irfs.median().shape[0] == 11

    def test_lag_selection_string(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags="bic", draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        assert fitted.n_lags >= 1

    def test_works_with_dummy_observations(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        augmented = var_data_2v.with_dummy_observations(n_lags=2, mu=5.0, delta=1.0)
        cvar = ConjugateVAR(lags=2, draws=50, random_seed=42)
        fitted = cvar.fit(augmented)
        assert fitted.idata.posterior["B"].values.shape == (1, 50, 2, 4)
