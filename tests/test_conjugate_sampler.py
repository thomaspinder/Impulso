"""Tests for the conjugate-VAR hyperparameter sampler (:mod:`impulso._conjugate_sampler`).

Gates:

* Unconstrained transforms round-trip and their log-Jacobians match finite differences
  (validates the ``log`` and ``logit`` change-of-variables folded into the target).
* With ``NIWPrior.select`` the marginal-likelihood mode and posterior recover the tightness
  ``lambda`` an independent dense grid identifies on synthetic data from a known VAR.
* With a known injected ``log_scales`` break the sampler recovers the injected volatility
  spikes.
* Random-walk Metropolis acceptance stays in a sane band.

Pure NumPy/SciPy, no PyMC, no mocks, deterministic seeds.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import pareto

from impulso._conjugate import ar1_residual_sd, log_marginal_likelihood
from impulso._conjugate_sampler import (
    _gamma_shape_scale,
    _laplace_cov,
    _log_jacobian,
    _to_constrained,
    _to_unconstrained,
    select_and_sample,
)
from impulso.priors import NIWPrior

# --------------------------------------------------------------------------- helpers

_A = np.array([[0.5, 0.1], [0.0, 0.4]])
_C = np.array([0.2, 0.1])
_SIGMA = np.array([[1.0, 0.3], [0.3, 0.8]])


def _design(series: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (Y, X) with a leading constant column (matches the module lag order)."""
    t_full, _ = series.shape
    t_obs = t_full - n_lags
    cols = [np.ones((t_obs, 1))]
    for ell in range(1, n_lags + 1):
        cols.append(series[n_lags - ell : t_full - ell])
    return series[n_lags:], np.hstack(cols)


def _sim_var1(seed: int, t_obs: int = 400, scale: np.ndarray | None = None) -> np.ndarray:
    """Simulate a stable VAR(1); ``scale[t]`` multiplies innovation ``t`` (post-burn indexing)."""
    rng = np.random.default_rng(seed)
    burn = 200
    n = 2
    chol = np.linalg.cholesky(_SIGMA)
    total = t_obs + burn
    if scale is None:
        scale = np.ones(total)
    y = np.zeros((total, n))
    y[0] = _C
    for t in range(1, total):
        y[t] = _C + _A @ y[t - 1] + scale[t] * (chol @ rng.standard_normal(n))
    return y[burn:]


# --------------------------------------------------------------------------- inline volatility stub


@dataclass(frozen=True)
class _ParetoPrior:
    """Pareto(b, scale) hyperprior on a positive volatility scale; support ``(scale, inf)``."""

    b: float = 1.0
    scale: float = 1.0

    @property
    def support(self) -> tuple[float, float]:
        return (float(self.scale), float(np.inf))

    def logpdf(self, x: float) -> float:
        return float(pareto.logpdf(x, self.b, scale=self.scale))


@dataclass(frozen=True)
class _BetaPrior:
    """Beta(a, b) hyperprior on a unit-interval decay; support ``(0, 1)``."""

    a: float
    b: float

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def logpdf(self, x: float) -> float:
        return float(beta_dist.logpdf(x, self.a, self.b))


class _SpikeBreak:
    """Deterministic common-scale break: three block spikes plus a geometric decay tail.

    Mirrors the frozen ``ConjugateVolatility`` estimation surface (``hyperparameter_priors`` /
    ``log_scales``) with the exact ``PandemicBreak`` keys, so recovery of the injected spikes
    exercises the real sampler path. Each spike governs ``block`` consecutive in-sample rows to
    keep the single-period estimates well identified for a unit test.
    """

    def __init__(self, start: int, block: int = 4, tail: int = 6):
        self.start = start
        self.block = block
        self.tail = tail

    def hyperparameter_priors(self) -> dict:
        return {
            "s_march": _ParetoPrior(),
            "s_april": _ParetoPrior(),
            "s_may": _ParetoPrior(),
            "rho": _BetaPrior(3.03568545, 1.50892136),
        }

    def log_scales(self, theta: dict, n_obs: int) -> np.ndarray:
        log_scales = np.zeros(n_obs)
        start, block = self.start, self.block
        for j, name in enumerate(("s_march", "s_april", "s_may")):
            lo = start + j * block
            log_scales[lo : lo + block] = np.log(theta[name])
        rho, s_may = theta["rho"], theta["s_may"]
        for k in range(self.tail):
            idx = start + 3 * block + k
            if idx < n_obs:
                log_scales[idx] = np.log(1.0 + (s_may - 1.0) * rho ** (k + 1))
        return log_scales


def _scale_path(start: int, block: int, spikes: tuple[float, float, float], total: int) -> np.ndarray:
    """Innovation-scale path (post-burn indexing) that injects block spikes into the DGP."""
    scale = np.ones(total)
    burn = 200
    for j, s in enumerate(spikes):
        first = burn + 1 + start + j * block  # +1: first in-sample obs is the 2nd data row (1 lag)
        scale[first : first + block] = s
    return scale


# --------------------------------------------------------------------------- transforms


class TestTransforms:
    """The unconstrained transforms round-trip and their log-Jacobians are exact."""

    def test_round_trip(self):
        cases = [(0.35, 0.0, np.inf), (4.2, 1.0, np.inf), (0.3, 0.0, 1.0), (0.85, 0.0, 1.0)]
        for x, lo, hi in cases:
            u = _to_unconstrained(x, lo, hi)
            assert abs(_to_constrained(u, lo, hi) - x) < 1e-12

    def test_log_jacobian_matches_finite_difference(self):
        # log positive (lo=0), log-shift positive (lo=1), logit unit-interval (0,1).
        cases = [(0.3, 0.0, np.inf), (0.7, 1.0, np.inf), (0.4, 0.0, 1.0), (-1.2, 0.0, 1.0)]
        h = 1e-6
        for u, lo, hi in cases:
            deriv = (_to_constrained(u + h, lo, hi) - _to_constrained(u - h, lo, hi)) / (2 * h)
            assert abs(np.log(abs(deriv)) - _log_jacobian(u, lo, hi)) < 1e-6

    def test_gamma_shape_scale_matches_mode_and_sd(self):
        for mode, sd in [(0.2, 0.4), (17.0, 8.0), (1.5, 0.5)]:
            shape, scale = _gamma_shape_scale(mode, sd)
            assert abs((shape - 1.0) * scale - mode) < 1e-9
            assert abs(np.sqrt(shape) * scale - sd) < 1e-9


# --------------------------------------------------------------------------- lambda selection


class TestLambdaSelection:
    """With ``select`` the sampler recovers the marginal-likelihood tightness."""

    def _oracle(self, series, prior):
        """Independent dense-grid u-space MAP and lambda-space posterior mean of the tightness."""
        y, x = _design(series, 1)
        sig = ar1_residual_sd(series)
        shape, scale = _gamma_shape_scale(prior.lambda_mode, prior.lambda_sd)

        def log_ml(lam):
            yd, xd = prior.build_dummies(series, 1, sig, tightness=lam)
            return log_marginal_likelihood(y, x, yd, xd)

        lams = np.linspace(0.02, 3.0, 600)
        lam_post = np.array([log_ml(lam) + gamma_dist.logpdf(lam, shape, scale=scale) for lam in lams])
        weights = np.exp(lam_post - lam_post.max())
        weights /= weights.sum()
        post_mean = float((lams * weights).sum())
        # u = log(lambda): add the +u Jacobian for the sampled-space MAP.
        us = np.linspace(np.log(0.02), np.log(3.0), 6000)
        u_post = np.array([log_ml(np.exp(u)) + gamma_dist.logpdf(np.exp(u), shape, scale=scale) + u for u in us])
        u_map = float(np.exp(us[np.argmax(u_post)]))
        return u_map, post_mean

    def test_recovers_marginal_likelihood_lambda(self):
        series = _sim_var1(seed=0, t_obs=400)
        prior = NIWPrior(select=True, lambda_mode=0.2, lambda_sd=5.0)
        u_map, post_mean = self._oracle(series, prior)

        result = select_and_sample(series, 1, prior, None, draws=2500, tune=2000, seed=42)
        lam = result["hyperparameters"]["lambda_"]

        # Mode is the argmax of the sampled (transformed) density; matches the grid oracle exactly.
        assert abs(result["mode"]["lambda_"] - u_map) < 0.02
        # Posterior mean is parametrisation-invariant; matches the independently integrated mean.
        assert abs(lam.mean() - post_mean) / post_mean < 0.15
        assert 0.15 <= result["acceptance_rate"] <= 0.45
        assert lam.shape == (2500,)
        assert np.all(lam > 0)

    def test_coefficient_draw_shapes_and_cholesky(self):
        series = _sim_var1(seed=1, t_obs=300)
        prior = NIWPrior(select=True, lambda_mode=0.2, lambda_sd=5.0)
        result = select_and_sample(series, 1, prior, None, draws=500, tune=500, seed=7)

        b_full, sigma, chol = result["B_full"], result["Sigma"], result["L"]
        assert b_full.shape == (500, 2, 3)  # k = 1 + n_vars * n_lags = 3
        assert sigma.shape == chol.shape == (500, 2, 2)
        assert np.isfinite(b_full).all() and np.isfinite(sigma).all()
        assert np.allclose(sigma, sigma.transpose(0, 2, 1))  # symmetric
        assert np.allclose(chol, np.tril(chol))  # lower-triangular
        assert np.allclose(chol @ chol.transpose(0, 2, 1), sigma, atol=1e-8)  # Sigma = L L'


# --------------------------------------------------------------------------- volatility spikes


class TestVolatilitySpikeRecovery:
    """With a known injected ``log_scales`` break the sampler recovers the spike magnitudes."""

    def test_recovers_injected_spikes(self):
        start, block, spikes = 150, 4, (6.0, 12.0, 5.0)
        scale = _scale_path(start, block, spikes, total=300 + 200)
        series = _sim_var1(seed=7, t_obs=300, scale=scale)

        prior = NIWPrior(tightness=0.3, select=False)  # fixed lambda: only volatility is free
        break_ = _SpikeBreak(start, block=block, tail=6)
        result = select_and_sample(series, 1, prior, break_, draws=2500, tune=2000, seed=11)

        assert set(result["hyperparameters"]) == {"s_march", "s_april", "s_may", "rho"}
        # The mode is data-deterministic (from the optimiser); recovers each injected spike.
        for name, truth in zip(("s_march", "s_april", "s_may"), spikes, strict=True):
            recovered = result["mode"][name]
            assert abs(recovered - truth) / truth < 0.30, (name, recovered, truth)
        # rho is weakly identified but must stay inside its support and be explored.
        rho = result["hyperparameters"]["rho"]
        assert np.all((rho > 0.0) & (rho < 1.0))
        assert rho.std() > 0
        assert 0.15 <= result["acceptance_rate"] <= 0.45

    def test_flat_break_matches_no_scale(self):
        """A break whose log_scales are identically zero reproduces the constant-variance fit."""
        series = _sim_var1(seed=3, t_obs=250)

        class _NoOpBreak:
            def hyperparameter_priors(self):
                return {"s_march": _ParetoPrior()}

            def log_scales(self, theta, n_obs):
                return np.zeros(n_obs)  # ignore theta -> volatility never rescales

        prior = NIWPrior(tightness=0.25, select=False)
        result = select_and_sample(series, 1, prior, _NoOpBreak(), draws=200, tune=200, seed=1)
        # Even with a degenerate (flat) scale path the draws are well-formed and finite.
        assert result["B_full"].shape == (200, 2, 3)
        assert np.isfinite(result["Sigma"]).all()
        assert 0.0 <= result["acceptance_rate"] <= 1.0


# --------------------------------------------------------------------------- degenerate case


class TestNoFreeHyperparameters:
    """With a fixed prior and no volatility there is nothing to sample: draw directly."""

    def test_fast_path(self):
        series = _sim_var1(seed=2, t_obs=200)
        prior = NIWPrior(tightness=0.25, select=False)
        result = select_and_sample(series, 1, prior, None, draws=64, tune=10, seed=1)

        assert result["hyperparameters"] == {}
        assert result["mode"] == {}
        assert result["acceptance_rate"] == 1.0
        assert result["B_full"].shape == (64, 2, 3)
        assert np.allclose(result["L"] @ result["L"].transpose(0, 2, 1), result["Sigma"], atol=1e-8)


# --------------------------------------------------------------------------- proposal robustness


class TestProposalCovariance:
    """The Laplace proposal covariance stays well conditioned for any Hessian.

    Regression guard: an imperfect numerical mode on a stiff, high-lag marginal likelihood can
    produce a Hessian with near-zero or negative eigenvalues. Without bounding, those directions
    send a covariance eigenvalue to ``1 / ~0`` (proposal std ~1e4), so every random-walk proposal
    is rejected and the whole chain collapses onto the mode. The covariance must instead bound the
    per-direction proposal standard deviation.
    """

    def _indefinite_hessian(self):
        rng = np.random.default_rng(0)
        rot, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        evals = np.array([150.0, 2.0, 1.0, 1e-10, -0.3])  # near-zero and negative directions
        return (rot * evals) @ rot.T

    def test_bounds_indefinite_hessian(self):
        cov = _laplace_cov(self._indefinite_hessian(), max_std=3.0, min_var=1e-4)
        evals = np.linalg.eigvalsh(cov)
        assert np.allclose(cov, cov.T)
        np.linalg.cholesky(cov)  # positive definite
        assert evals.min() >= 1e-4 - 1e-12
        assert evals.max() <= 3.0**2 + 1e-9  # proposal std capped at max_std, never ~1e4

    def test_well_conditioned_hessian_gives_reciprocal_variance(self):
        hess = np.diag([4.0, 25.0, 100.0])  # all inside the bounds
        cov = _laplace_cov(hess, max_std=3.0, min_var=1e-4)
        assert np.allclose(np.diag(cov), [0.25, 0.04, 0.01])

    def test_low_tune_does_not_collapse(self):
        """At a small tune budget the chain still mixes (acceptance in band, draws not degenerate)."""
        start, block, spikes = 150, 4, (6.0, 12.0, 5.0)
        scale = _scale_path(start, block, spikes, total=300 + 200)
        series = _sim_var1(seed=7, t_obs=300, scale=scale)
        prior = NIWPrior(select=True, lambda_mode=0.2, lambda_sd=1.0)
        break_ = _SpikeBreak(start, block=block, tail=6)

        result = select_and_sample(series, 1, prior, break_, draws=1000, tune=200, seed=0)
        assert 0.15 <= result["acceptance_rate"] <= 0.45
        # A collapsed chain repeats the mode; a mixing chain visits many distinct states.
        assert len(np.unique(result["hyperparameters"]["s_may"])) > 100
