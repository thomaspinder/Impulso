"""Tests for VAR specification."""

import pytest
from pydantic import ValidationError

from impulso.priors import MinnesotaPrior
from impulso.spec import VAR
from impulso.volatility import Constant


class TestVARSpec:
    def test_fixed_lags(self):
        spec = VAR(lags=4, prior="minnesota")
        assert spec.lags == 4
        assert spec.max_lags is None

    @pytest.mark.parametrize("criterion", ["aic", "bic", "hq"])
    def test_string_lags(self, criterion):
        spec = VAR(lags=criterion, prior="minnesota")
        assert spec.lags == criterion

    def test_string_lags_with_max(self):
        spec = VAR(lags="aic", max_lags=12, prior="minnesota")
        assert spec.max_lags == 12

    def test_rejects_max_lags_with_fixed(self):
        with pytest.raises(ValueError, match="max_lags is only valid"):
            VAR(lags=4, max_lags=12, prior="minnesota")

    def test_prior_string_resolves(self):
        spec = VAR(lags=2, prior="minnesota")
        assert isinstance(spec.resolved_prior, MinnesotaPrior)

    def test_prior_object(self):
        prior = MinnesotaPrior(tightness=0.2)
        spec = VAR(lags=2, prior=prior)
        assert spec.resolved_prior.tightness == 0.2

    def test_frozen(self):
        spec = VAR(lags=2, prior="minnesota")
        with pytest.raises(ValidationError):
            spec.lags = 3

    def test_rejects_zero_lags(self):
        with pytest.raises(ValidationError):
            VAR(lags=0, prior="minnesota")


class TestVolatilityParameter:
    def test_default_volatility_is_constant_string(self):
        spec = VAR(lags=2)
        assert spec.volatility == "constant"

    def test_volatility_string_resolves_to_constant(self):
        spec = VAR(lags=2)
        assert isinstance(spec.resolved_volatility, Constant)

    def test_volatility_object_pass_through(self):
        adapter = Constant(sigma_sd_beta=3.0)
        spec = VAR(lags=2, volatility=adapter)
        assert spec.resolved_volatility is adapter
        assert spec.resolved_volatility.sigma_sd_beta == 3.0

    def test_unknown_string_raises(self):
        with pytest.raises(ValidationError):
            VAR(lags=2, volatility="nonexistent")


class TestPyMCModelBuild:
    """Verify the PyMC model graph composition after the seam refactor."""

    def test_model_has_expected_unobserved_rvs(self, var_data_2v):
        """The same set of RVs must exist in the PyMC graph as before the seam."""
        # We don't run MCMC here; we just build the model and inspect.
        # Rebuild logic mirrors VAR.fit but stops before sampler.sample().
        import pymc as pm

        from impulso._lag_selection import select_lag_order  # noqa: F401 — import-side-effect parity

        spec = VAR(lags=1)
        prior_params = spec.resolved_prior.build_priors(n_vars=2, n_lags=1)
        volatility = spec.resolved_volatility

        y = var_data_2v.endog
        Y = y[1:]
        X_lag = y[:-1]

        with pm.Model() as model:
            pm.Normal("intercept", mu=0, sigma=1, shape=2)
            pm.Normal("B", mu=prior_params["B_mu"], sigma=prior_params["B_sigma"], shape=(2, 2))
            L = volatility.build_pymc_latent(n_vars=2, T=Y.shape[0])
            pm.Deterministic("Sigma", pm.math.dot(L, L.T))
            mu = pm.math.dot(X_lag, model.named_vars["B"].T)
            pm.MvNormal("obs", mu=mu, chol=L, observed=Y)

        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}
        assert "intercept" in rv_names
        assert "B" in rv_names
        assert "sigma_sd" in rv_names
        assert "tril_offdiag" in rv_names
        assert "Sigma" in det_names

    def test_var_fit_routes_through_volatility_seam(self, var_data_2v):
        """VAR.fit (intercepted before sampling) must register the canonical RV set.

        The companion test above mirrors the model graph by hand, so it would
        keep passing even if VAR.fit silently skipped the volatility delegation.
        This one captures the model from the production codepath via a sampler
        that aborts before MCMC, then asserts the graph shape.
        """
        import pymc as pm

        captured: dict[str, pm.Model] = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model: pm.Model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        spec = VAR(lags=1)
        with pytest.raises(RuntimeError, match="stop before sampling"):
            spec.fit(var_data_2v, sampler=CapturingSampler())

        model = captured["model"]
        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}
        assert {"intercept", "B", "sigma_sd", "tril_offdiag"} <= rv_names
        assert "Sigma" in det_names
        assert {v.name for v in model.observed_RVs} == {"obs"}
