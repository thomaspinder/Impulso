"""Tests for the prior- and posterior-predictive APIs (issue #56)."""

import subprocess
import sys

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

matplotlib.use("Agg")

from impulso import VAR, VARData
from impulso._lag_selection import select_lag_order
from impulso._residuals import fitted_values, reduced_form_residuals
from impulso.fitted import FittedVAR
from impulso.volatility import Constant

# Driven in a fresh interpreter: enable_runtime_checks() beartype-wraps the
# library's classes in place, so the wrapping must not leak into the rest of
# the suite. Beartype resolves return annotations on call, and it cannot
# resolve a DOTTED forward reference whose module is TYPE_CHECKING-only
# (`"pm.Model"` blows up with BeartypeCallHintForwardRefException). This
# script is the regression fence for that.
_RUNTIME_CHECKS_SCRIPT = """
import numpy as np
import pandas as pd

import impulso

impulso.enable_runtime_checks()

from beartype.roar import BeartypeCallHintViolation
from impulso import VAR, VARData

rng = np.random.default_rng(0)
index = pd.date_range("2000-01-01", periods=40, freq="QS")
data = VARData(endog=rng.standard_normal((40, 2)) * 0.1, endog_names=["y1", "y2"], index=index)

spec = VAR(lags=1)
prior = spec.prior_predictive(data, draws=5, random_seed=0)
assert prior.prior_predictive["obs"].shape == (1, 5, 39, 2)

fitted = spec.fit(data, sampler=impulso.NUTSSampler(draws=5, tune=5, chains=1, cores=1, progressbar=False))
assert fitted.posterior_predictive(seed=0).posterior_predictive["obs"].shape == (1, 5, 39, 2)
print("PREDICTIVE_OK")

try:
    spec.prior_predictive(data, draws="five")
except BeartypeCallHintViolation:
    print("VIOLATION_CAUGHT")
else:
    raise AssertionError("beartype did not flag draws='five'")
"""


@pytest.fixture
def var_data_2v_exog(var_data_2v):
    """`var_data_2v` plus a single deterministic exogenous column."""
    T = var_data_2v.endog.shape[0]
    exog = np.linspace(-1.0, 1.0, T).reshape(T, 1)
    return VARData(
        endog=var_data_2v.endog,
        endog_names=list(var_data_2v.endog_names),
        exog=exog,
        exog_names=["x1"],
        index=var_data_2v.index,
    )


@pytest.fixture
def var_data_2v_correlated():
    """VAR(1) DGP whose reduced-form shocks are strongly cross-correlated.

    `var_data_2v`'s shocks are independent, so `L` and `L.T` imply almost
    the same covariance and a Cholesky-orientation bug would slip through
    the drift fence. Here `corr(u1, u2) ~ 0.83`.
    """
    rng = np.random.default_rng(11)
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    L_true = np.array([[0.20, 0.0], [0.15, 0.10]])
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + L_true @ rng.standard_normal(n)
    return VARData(
        endog=y,
        endog_names=["y1", "y2"],
        index=pd.date_range("2000-01-01", periods=T, freq="QS"),
    )


@pytest.fixture
def fitted_2v(synthetic_idata_2v, var_data_2v):
    """FittedVAR over the synthetic 2-var posterior — no MCMC."""
    return FittedVAR(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


class TestPriorPredictive:
    """`VAR.prior_predictive` draws from the same graph `fit` samples."""

    def test_groups_and_variable(self, var_data_2v):
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=50, random_seed=0)

        assert {"prior", "prior_predictive", "observed_data"} <= set(idata.groups())
        assert "obs" in idata.prior_predictive
        assert "obs" in idata.observed_data
        # The latents are simulated too, so the prior itself is inspectable.
        assert {"intercept", "B", "sigma_sd"} <= set(idata.prior.data_vars)

    def test_shape_dims_and_coords(self, var_data_2v):
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=50, random_seed=0)

        obs = idata.prior_predictive["obs"]
        # PyMC's prior predictive is a single chain.
        assert obs.shape == (1, 50, 199, 2)
        assert obs.dims == ("chain", "draw", "time", "var")
        assert list(obs.coords["var"].values) == ["y1", "y2"]
        assert np.array_equal(
            pd.to_datetime(obs.coords["time"].values),
            var_data_2v.index[1:],
        )
        assert idata.observed_data["obs"].dims == ("time", "var")
        assert np.array_equal(idata.observed_data["obs"].values, var_data_2v.endog[1:])

    def test_random_seed_is_reproducible(self, var_data_2v):
        spec = VAR(lags=1)
        a = spec.prior_predictive(var_data_2v, draws=20, random_seed=0)
        b = spec.prior_predictive(var_data_2v, draws=20, random_seed=0)
        c = spec.prior_predictive(var_data_2v, draws=20, random_seed=1)

        assert np.array_equal(a.prior_predictive["obs"].values, b.prior_predictive["obs"].values)
        assert not np.array_equal(a.prior_predictive["obs"].values, c.prior_predictive["obs"].values)

    def test_lag_order_trims_the_time_axis(self, var_data_2v):
        idata = VAR(lags=3).prior_predictive(var_data_2v, draws=10, random_seed=0)

        assert idata.prior_predictive["obs"].shape[2] == 197

    def test_criterion_lags_resolve_like_fit(self, var_data_2v):
        idata = VAR(lags="bic").prior_predictive(var_data_2v, draws=10, random_seed=0)

        expected_lags = select_lag_order(var_data_2v, max_lags=12).bic
        assert idata.prior_predictive["obs"].shape[2] == 200 - expected_lags

    def test_exogenous_regressors_are_simulated(self, var_data_2v_exog):
        idata = VAR(lags=1).prior_predictive(var_data_2v_exog, draws=10, random_seed=0)

        assert "B_exog" in idata.prior
        assert idata.prior["B_exog"].shape == (1, 10, 2, 1)

    def test_observed_series_falls_within_the_prior_band(self, var_data_2v):
        """Issue AC: the observed series must sit inside the 95% prior band.

        Quantiles, never mean +/- k*sd: the HalfCauchy scale prior has no
        finite moments, so a moment-based band is meaningless here.
        """
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=500, random_seed=0)

        draws = idata.prior_predictive["obs"].values[0]  # (draws, time, var)
        lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
        observed = var_data_2v.endog[1:]
        covered = (observed >= lower) & (observed <= upper)
        assert covered.mean() >= 0.9

    def test_plot_ppc_smoke(self, var_data_2v):
        """`az.plot_ppc` must accept the group as returned (datetime time coord)."""
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=20, random_seed=0)

        axes = az.plot_ppc(idata, group="prior", num_pp_samples=5)
        assert axes is not None


class TestPosteriorPredictive:
    """`FittedVAR.posterior_predictive` replicates the estimation sample."""

    def test_shape_dims_and_observed_data(self, fitted_2v, var_data_2v):
        ppc = fitted_2v.posterior_predictive(seed=0)

        obs = ppc.posterior_predictive["obs"]
        assert obs.shape == (2, 50, 199, 2)
        assert obs.dims == ("chain", "draw", "time", "var")
        assert list(obs.coords["var"].values) == ["y1", "y2"]
        assert np.array_equal(pd.to_datetime(obs.coords["time"].values), var_data_2v.index[1:])
        assert ppc.observed_data["obs"].dims == ("time", "var")
        assert np.array_equal(ppc.observed_data["obs"].values, var_data_2v.endog[1:])

    def test_mean_mode_ties_the_residual_helpers(self, fitted_2v, var_data_2v):
        """Mean mode IS the conditional mean, so observed - mean IS the residual."""
        ppc = fitted_2v.posterior_predictive(simulate_innovations=False)

        posterior = fitted_2v.idata.posterior
        mu = fitted_values(posterior, var_data_2v, 1)
        assert np.array_equal(ppc.posterior_predictive["obs"].values, mu)

        implied = ppc.observed_data["obs"].values - ppc.posterior_predictive["obs"].values
        assert np.array_equal(implied, reduced_form_residuals(posterior, var_data_2v, 1))

    def test_mean_mode_consumes_no_randomness(self, fitted_2v):
        """Two unseeded mean-mode calls must agree exactly."""
        a = fitted_2v.posterior_predictive(simulate_innovations=False)
        b = fitted_2v.posterior_predictive(simulate_innovations=False)

        assert np.array_equal(
            a.posterior_predictive["obs"].values,
            b.posterior_predictive["obs"].values,
        )

    def test_seed_is_reproducible_and_accepts_a_generator(self, fitted_2v):
        a = fitted_2v.posterior_predictive(seed=0)
        b = fitted_2v.posterior_predictive(seed=0)
        c = fitted_2v.posterior_predictive(seed=1)
        g = fitted_2v.posterior_predictive(seed=np.random.default_rng(0))

        assert np.array_equal(a.posterior_predictive["obs"].values, b.posterior_predictive["obs"].values)
        assert not np.array_equal(a.posterior_predictive["obs"].values, c.posterior_predictive["obs"].values)
        assert np.array_equal(a.posterior_predictive["obs"].values, g.posterior_predictive["obs"].values)

    def test_innovations_carry_the_models_sigma(self, fitted_2v, var_data_2v):
        """Standardising by each draw's own L must leave white noise.

        This is the assertion that pins the deviation from the issue text:
        Sigma comes from the volatility seam (the model's own Sigma), not
        from an empirical residual covariance.
        """
        ppc = fitted_2v.posterior_predictive(seed=0)

        posterior = fitted_2v.idata.posterior
        mu = fitted_values(posterior, var_data_2v, 1)
        innovations = ppc.posterior_predictive["obs"].values - mu  # (C, D, T, n)
        L = posterior["L"].values  # (C, D, n, n)

        # eps = L^-1 @ innovation, solved per draw against the whole time block.
        eps = np.linalg.solve(
            L[:, :, np.newaxis, :, :],
            innovations[..., np.newaxis],
        )[..., 0]
        pooled = eps.reshape(-1, eps.shape[-1])  # (C*D*T, n)

        assert pooled.shape[0] == 2 * 50 * 199
        assert np.abs(pooled.mean(axis=0)).max() < 0.05
        assert np.allclose(np.cov(pooled, rowvar=False), np.eye(2), atol=0.06)

    def test_does_not_mutate_the_fit(self, fitted_2v):
        before = set(fitted_2v.idata.groups())
        ppc = fitted_2v.posterior_predictive(seed=0)

        assert set(fitted_2v.idata.groups()) == before
        assert "posterior_predictive" not in before

        # The documented recipe for attaching it must work.
        fitted_2v.idata.extend(ppc)
        assert "posterior_predictive" in set(fitted_2v.idata.groups())

    def test_exog_without_b_exog_raises(self, synthetic_idata_2v, var_data_2v_exog):
        fitted = FittedVAR(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v_exog,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )

        with pytest.raises(ValueError, match="B_exog"):
            fitted.posterior_predictive(seed=0)

    def test_conjugate_var_posterior_predictive(self, var_data_2v):
        """ConjugateVAR builds no PyMC graph, so NumPy parity is the point."""
        from impulso.conjugate import ConjugateVAR
        from impulso.priors import NIWPrior

        fitted = ConjugateVAR(lags=1, prior=NIWPrior(), draws=40, tune=40, seed=0).fit(var_data_2v)
        assert fitted.pymc_model is None

        ppc = fitted.posterior_predictive(seed=0)
        assert ppc.posterior_predictive["obs"].shape == (1, 40, 199, 2)
        assert np.isfinite(ppc.posterior_predictive["obs"].values).all()

    def test_plot_ppc_smoke(self, fitted_2v):
        ppc = fitted_2v.posterior_predictive(seed=0)

        axes = az.plot_ppc(ppc, num_pp_samples=5)
        assert axes is not None


class TestPredictiveRuntimeChecks:
    """Both methods must survive `impulso.enable_runtime_checks()`."""

    @pytest.mark.slow
    def test_beartype_resolves_the_predictive_annotations(self):
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _RUNTIME_CHECKS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "PREDICTIVE_OK" in result.stdout, result.stdout
        assert "VIOLATION_CAUGHT" in result.stdout, result.stdout


class TestPredictiveAgainstPyMC:
    """The drift fence for ADR-0011: NumPy replicates must match the graph."""

    @pytest.mark.slow
    def test_matches_sample_posterior_predictive(self, var_data_2v_correlated):
        """Distributional agreement with `pm.sample_posterior_predictive` on a real fit.

        Both routes reuse the same posterior draws, so the conditional
        means are identical by construction and only the innovation law
        can differ. The comparison is therefore stated on the innovations:
        per-(time, var) means in Monte Carlo standard-error units (a
        fixed atol would depend on the DGP's scale), the pooled
        innovation covariance, which a transposed or mis-scaled Cholesky
        breaks immediately, and — issue #232 — the shape of the pooled
        standardised innovations, which the first two moments cannot see.

        The fixture's shocks are strongly cross-correlated on purpose: with
        a near-diagonal Sigma, `L` and `L.T` produce nearly the same
        covariance, so the fence would not bite.
        """
        import pymc as pm

        from impulso.samplers import NUTSSampler

        data = var_data_2v_correlated
        draws = 400
        sampler = NUTSSampler(draws=draws, tune=400, chains=1, cores=1, random_seed=3, progressbar=False)
        fitted = VAR(lags=1).fit(data, sampler=sampler)

        ours = fitted.posterior_predictive(seed=0).posterior_predictive["obs"].values
        with fitted.pymc_model:
            theirs = (
                pm
                .sample_posterior_predictive(
                    fitted.idata,
                    random_seed=1,
                    progressbar=False,
                )
                .posterior_predictive["obs"]
                .values
            )

        assert ours.shape == theirs.shape

        posterior = fitted.idata.posterior
        mu = fitted_values(posterior, data, 1)
        ours_resid = ours - mu  # (C, D, T, n)
        theirs_resid = theirs - mu
        ours_innov = ours_resid.reshape(-1, 2)
        theirs_innov = theirs_resid.reshape(-1, 2)

        # Per-(time, var) means, normalised by the Monte Carlo standard
        # error of the difference (sigma * sqrt(2 / draws)). 398 cells, so
        # 5 sigma is a ~2e-4 false-failure budget.
        se = theirs_innov.std(axis=0) * np.sqrt(2.0 / draws)
        z = np.abs(ours.mean(axis=(0, 1)) - theirs.mean(axis=(0, 1))) / se
        assert z.max() < 5.0

        assert np.allclose(ours_innov.std(axis=0), theirs_innov.std(axis=0), rtol=0.05)
        theirs_cov = np.cov(theirs_innov, rowvar=False)
        assert np.allclose(
            np.cov(ours_innov, rowvar=False),
            theirs_cov,
            atol=0.05 * np.abs(theirs_cov).max(),
        )

        # --- shape (issue #232) ------------------------------------------
        # Everything above is first- and second-moment only, so a Student-t
        # observation likelihood rescaled to matched variance would pass it
        # untouched (variances agree to ~0.1%; only the tails separate).
        # Standardising by each draw's own L is an invertible per-draw map
        # applied identically to both samples, so it can neither create nor
        # mask a difference — but it removes the across-draw scale mixture
        # that would otherwise put excess kurtosis in *both* pools and
        # swamp the signal. Pooled over (chain, draw, time) each sample is
        # then iid standard normal under the null.
        L = posterior["L"].values  # (C, D, n, n)
        ours_eps = np.linalg.solve(L[:, :, np.newaxis, :, :], ours_resid[..., np.newaxis])[..., 0].reshape(-1, 2)
        theirs_eps = np.linalg.solve(L[:, :, np.newaxis, :, :], theirs_resid[..., np.newaxis])[..., 0].reshape(-1, 2)

        # Excess kurtosis of N iid normals has SE = sqrt(24 / N), so the
        # difference of two independent pools has SE = sqrt(48 / N).
        # N = 1 * 400 * 199 = 79_600 gives SE ~ 0.0247, and 6 sigma plus a
        # 0.02 slack is ~0.167. Measured: |delta| <= 0.077 over 200 null
        # replications (~1e-11 flake budget at the bound), against >= 1.56
        # for a t(7) pool rescaled to unit variance — a 10x margin.
        n_pooled = ours_eps.shape[0]
        kurt_tol = 6.0 * np.sqrt(48.0 / n_pooled) + 0.02
        for i, name in enumerate(data.endog_names):
            ours_k = stats.kurtosis(ours_eps[:, i], fisher=True, bias=False)
            theirs_k = stats.kurtosis(theirs_eps[:, i], fisher=True, bias=False)
            assert abs(ours_k - theirs_k) < kurt_tol, f"{name}: excess kurtosis {ours_k:.4f} vs {theirs_k:.4f}"

            # The fourth moment alone is blind to shape changes that keep
            # it (a skewed or mixture innovation law); the two-sample KS
            # statistic is not, and it costs a sort. p is uniform under the
            # null, so 1e-6 is a 1e-6 flake budget per variable; the t(7)
            # alternative lands at p ~ 1e-20.
            ks_p = stats.ks_2samp(ours_eps[:, i], theirs_eps[:, i]).pvalue
            assert ks_p > 1e-6, f"{name}: standardised innovations differ in distribution (KS p={ks_p:.3e})"

    @pytest.mark.slow
    def test_observed_falls_within_the_posterior_band(self, var_data_2v):
        from impulso.samplers import NUTSSampler

        sampler = NUTSSampler(draws=200, tune=200, chains=1, cores=1, random_seed=3, progressbar=False)
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)

        draws = fitted.posterior_predictive(seed=0).posterior_predictive["obs"].values
        flat = draws.reshape(-1, *draws.shape[2:])  # (chain*draw, time, var)
        lower, upper = np.quantile(flat, [0.025, 0.975], axis=0)
        observed = var_data_2v.endog[1:]
        covered = (observed >= lower) & (observed <= upper)
        assert covered.mean() >= 0.85

    @pytest.mark.slow
    def test_stochastic_volatility_innovations_vary_with_t(self, var_data_2v):
        """Under SV the replicate spread must track the model's own Sigma_t.

        This is the assertion behind the volatility-seam deviation: the
        innovations come from `cholesky_path` (per-t), so their spread
        follows Sigma_t. A single pooled residual covariance — or
        `cholesky_at` — would give a flat spread across t.
        """
        from impulso.samplers import NUTSSampler
        from impulso.sv.spec import StochasticVolatility

        sampler = NUTSSampler(
            draws=30, tune=30, chains=1, cores=1, random_seed=0, progressbar=False, nuts_sampler="pymc"
        )
        fitted = VAR(lags=1, volatility=StochasticVolatility()).fit(var_data_2v, sampler=sampler)

        ppc = fitted.posterior_predictive(seed=0).posterior_predictive["obs"].values
        assert np.isfinite(ppc).all()

        posterior = fitted.idata.posterior
        T = var_data_2v.endog.shape[0] - 1
        mu = fitted_values(posterior, var_data_2v, 1)
        empirical_sd = (ppc - mu).std(axis=(0, 1))  # (T, n)

        L_path = fitted.volatility.cholesky_path(posterior, T=T)  # (C, D, T, n, n)
        model_sd = np.sqrt((L_path**2).sum(axis=-1)).mean(axis=(0, 1))  # (T, n)

        # Not flat across t ...
        assert empirical_sd.max() / empirical_sd.min() > 1.5
        # ... and varying in step with the model's own per-t scale.
        for i in range(2):
            corr = np.corrcoef(empirical_sd[:, i], model_sd[:, i])[0, 1]
            assert corr > 0.4
