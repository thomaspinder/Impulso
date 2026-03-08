"""Tests for GLP hierarchical prior selection on ConjugateVAR."""

import numpy as np

from impulso.priors import MinnesotaPrior


class TestMarginalLikelihood:
    def test_returns_finite_scalar(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        ml = cvar.marginal_likelihood(var_data_2v)
        assert np.isfinite(ml)

    def test_varies_with_prior(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar_tight = ConjugateVAR(lags=1, prior=MinnesotaPrior(tightness=0.01))
        cvar_loose = ConjugateVAR(lags=1, prior=MinnesotaPrior(tightness=1.0))
        ml_tight = cvar_tight.marginal_likelihood(var_data_2v)
        ml_loose = cvar_loose.marginal_likelihood(var_data_2v)
        assert ml_tight != ml_loose

    def test_higher_for_true_lag_order(self, var_data_2v):
        """Marginal likelihood should favour the true DGP lag order (1)."""
        from impulso.conjugate import ConjugateVAR

        ml_1 = ConjugateVAR(lags=1).marginal_likelihood(var_data_2v)
        ml_8 = ConjugateVAR(lags=8).marginal_likelihood(var_data_2v)
        assert ml_1 > ml_8


class TestOptimizePrior:
    def test_returns_minnesota_prior(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        optimal = cvar.optimize_prior(var_data_2v)
        assert isinstance(optimal, MinnesotaPrior)

    def test_optimal_tightness_positive(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        optimal = cvar.optimize_prior(var_data_2v)
        assert optimal.tightness > 0

    def test_optimal_cross_shrinkage_in_bounds(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        optimal = cvar.optimize_prior(var_data_2v)
        assert 0.01 <= optimal.cross_shrinkage <= 1.0

    def test_optimal_has_higher_marginal_likelihood(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar_default = ConjugateVAR(lags=1)
        ml_default = cvar_default.marginal_likelihood(var_data_2v)

        optimal_prior = cvar_default.optimize_prior(var_data_2v)
        cvar_optimal = ConjugateVAR(lags=1, prior=optimal_prior)
        ml_optimal = cvar_optimal.marginal_likelihood(var_data_2v)

        assert ml_optimal >= ml_default - 1e-6  # allow tiny numerical tolerance

    def test_preserves_decay_setting(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, prior=MinnesotaPrior(decay="geometric"))
        optimal = cvar.optimize_prior(var_data_2v)
        assert optimal.decay == "geometric"

    def test_minnesota_optimized_shorthand(self, var_data_2v):
        """prior='minnesota_optimized' should trigger automatic optimisation."""
        from impulso.conjugate import ConjugateVAR
        from impulso.fitted import FittedVAR

        cvar = ConjugateVAR(lags=1, prior="minnesota_optimized", draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        assert isinstance(fitted, FittedVAR)
