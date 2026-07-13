"""Empirical-Bayes hyperparameter sampler for the conjugate VAR (Giannone-Lenza-Primiceri).

:func:`select_and_sample` estimates the free hyperparameters of a conjugate
Normal-Inverse-Wishart VAR by marginal likelihood and samples their posterior. The free set
is the Minnesota tightness ``lambda`` (when :attr:`NIWPrior.select` is set) together with the
volatility-break hyperparameters (when a volatility adapter is supplied, e.g. the three
outbreak scales and the decay of a pandemic break).

The procedure is:

1. Transform every bounded/positive hyperparameter to the whole real line — ``log`` for the
   positive tightness and volatility scales, ``logit`` for a unit-interval decay — and fold
   the change-of-variable log-Jacobian into the target so the *sampled* density is exactly the
   posterior in the transformed space.
2. Maximise ``log_marginal_likelihood + log_hyperprior + log_jacobian`` to the mode
   (:func:`scipy.optimize.minimize`).
3. Build a Gaussian random-walk proposal from the Laplace covariance (inverse Hessian at the
   mode), scaled toward ~25% acceptance and lightly adapted during the tuning phase.
4. Run random-walk Metropolis (``tune`` warm-up + ``draws`` retained iterations).
5. For every retained draw, form the conjugate posterior at that hyperparameter value (on the
   volatility-rescaled data with unscaled dummies) and draw one ``(B_full, Sigma, L)`` triple.

Pure NumPy/SciPy; no PyMC.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import optimize
from scipy.special import expit
from scipy.stats import gamma as _gamma

from impulso._conjugate import ar1_residual_sd, draw_niw, log_marginal_likelihood, niw_posterior
from impulso.priors import NIWPrior


class HyperPrior(Protocol):
    """A one-dimensional hyperprior: a log density plus its ``(lower, upper)`` support."""

    support: tuple[float, float]

    def logpdf(self, x: float) -> float:
        """Log probability density at ``x`` (``-inf`` outside the support)."""
        ...


class VolatilityEstimation(Protocol):
    """Estimation surface a conjugate volatility adapter exposes to the sampler."""

    def hyperparameter_priors(self) -> dict[str, HyperPrior]:
        """Named hyperpriors for the volatility break, one per free hyperparameter."""
        ...

    def log_scales(self, theta: dict[str, float], n_obs: int) -> np.ndarray:
        """Per-in-sample-``t`` log scale path of shape ``(n_obs,)`` for hyperparameters ``theta``."""
        ...


# --------------------------------------------------------------------------- transforms


def _log_sigmoid(u: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable ``log(sigmoid(u)) = -log(1 + exp(-u))``."""
    return -np.logaddexp(0.0, -u)


def _to_constrained(u: float, lo: float, hi: float) -> float:
    """Map an unconstrained ``u`` back to the parameter support ``(lo, hi)``."""
    if np.isinf(hi):
        return lo + float(np.exp(u))
    return lo + (hi - lo) * float(expit(u))


def _to_unconstrained(x: float, lo: float, hi: float) -> float:
    """Map a constrained ``x`` in ``(lo, hi)`` to the whole real line."""
    if np.isinf(hi):
        return float(np.log(x - lo))
    return float(np.log((x - lo) / (hi - x)))


def _log_jacobian(u: float, lo: float, hi: float) -> float:
    """Log absolute derivative ``log|dx/du|`` of the constraining map at ``u``."""
    if np.isinf(hi):
        return float(u)  # x = lo + exp(u) -> dx/du = exp(u), log = u
    return float(np.log(hi - lo) + _log_sigmoid(u) + _log_sigmoid(-u))


def _gamma_shape_scale(mode: float, sd: float) -> tuple[float, float]:
    """Gamma ``(shape, scale)`` with the given ``mode`` and standard deviation ``sd``."""
    scale = 0.5 * (-mode + np.sqrt(mode * mode + 4.0 * sd * sd))
    shape = mode / scale + 1.0
    return float(shape), float(scale)


# --------------------------------------------------------------------------- design


def _design(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(Y, X)`` with a leading constant column, matching :func:`minnesota_dummies`."""
    t_full, _ = y.shape
    t_obs = t_full - n_lags
    response = y[n_lags:]
    cols = [np.ones((t_obs, 1))]
    for ell in range(1, n_lags + 1):
        cols.append(y[n_lags - ell : t_full - ell])
    return response, np.hstack(cols)


def _rescale(Y: np.ndarray, X: np.ndarray, log_scales: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Whiten rows of ``(Y, X)`` by ``exp(-log_scales)``; identity when ``log_scales`` is ``None``."""
    if log_scales is None:
        return Y, X
    weights = np.exp(-np.asarray(log_scales, dtype=float).ravel())[:, None]
    return Y * weights, X * weights


# --------------------------------------------------------------------------- parameters


class _ParamSpec:
    """A single free hyperparameter: name, support, log hyperprior and grouping."""

    __slots__ = ("hi", "init", "is_lambda", "lo", "logpdf", "name")

    def __init__(self, name, lo, hi, logpdf, is_lambda, init):
        self.name = name
        self.lo = float(lo)
        self.hi = float(hi)
        self.logpdf = logpdf
        self.is_lambda = is_lambda
        self.init = float(init)


def _param_specs(prior: NIWPrior, volatility: VolatilityEstimation | None) -> list[_ParamSpec]:
    """Assemble the ordered free-hyperparameter specs: tightness first, then volatility."""
    specs: list[_ParamSpec] = []
    if prior.select:
        shape, scale = _gamma_shape_scale(prior.lambda_mode, prior.lambda_sd)
        specs.append(
            _ParamSpec(
                "lambda_",
                0.0,
                np.inf,
                lambda x, a=shape, s=scale: float(_gamma.logpdf(x, a, scale=s)),
                True,
                prior.tightness,
            )
        )
    if volatility is not None:
        for name, hyper in volatility.hyperparameter_priors().items():
            lo, hi = hyper.support
            lo, hi = float(lo), float(hi)
            init = lo + 1.0 if np.isinf(hi) else lo + 0.5 * (hi - lo)
            specs.append(_ParamSpec(name, lo, hi, hyper.logpdf, False, init))
    return specs


def _num_hessian(f, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central-difference Hessian of scalar ``f`` at ``x``."""
    d = x.size
    step = np.eye(d) * eps
    f0 = f(x)
    fp = np.array([f(x + step[i]) for i in range(d)])
    fm = np.array([f(x - step[i]) for i in range(d)])
    hess = np.zeros((d, d))
    for i in range(d):
        hess[i, i] = (fp[i] - 2.0 * f0 + fm[i]) / (eps * eps)
    for i in range(d):
        for j in range(i + 1, d):
            fpp = f(x + step[i] + step[j])
            fmm = f(x - step[i] - step[j])
            fpm = f(x + step[i] - step[j])
            fmp = f(x - step[i] + step[j])
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    return hess


def _laplace_cov(hess: np.ndarray, *, max_std: float = 3.0, min_var: float = 1e-4) -> np.ndarray:
    """Well-conditioned proposal covariance from a Hessian of the *negative* log target.

    The covariance eigenvalues are the reciprocal Hessian eigenvalues, but bounded to
    ``[min_var, max_std ** 2]``. Bounding is essential: a flat, indefinite or poorly estimated
    Hessian direction (which happens when the numerical mode is imperfect on a stiff, high-lag
    marginal likelihood) would otherwise send a covariance eigenvalue to ``1 / ~0`` and blow the
    random-walk proposal up so far that every draw is rejected — the sampler then collapses onto
    the mode. Capping the per-direction proposal standard deviation at ``max_std`` (a factor of
    ``exp(3)`` in log space, or the whole unit interval in logit space) keeps the proposal
    exploratory but finite regardless of Hessian quality.
    """
    sym = 0.5 * (hess + hess.T)
    evals, evecs = np.linalg.eigh(sym)
    var = np.where(evals > 0.0, 1.0 / np.maximum(evals, 1e-12), np.inf)
    var = np.clip(var, min_var, max_std * max_std)
    return (evecs * var) @ evecs.T


def select_and_sample(  # noqa: C901
    y: np.ndarray,
    n_lags: int,
    prior: NIWPrior,
    volatility: VolatilityEstimation | None,
    *,
    draws: int,
    tune: int,
    seed: int,
) -> dict:
    """Estimate and sample the conjugate-VAR hyperparameters, then draw coefficients.

    Maximises ``log_marginal_likelihood + log_hyperprior`` over the free hyperparameters
    (tightness ``lambda`` when ``prior.select``; the volatility-break hyperparameters when
    ``volatility`` is not ``None``) in an unconstrained parametrisation with the log-Jacobian
    included, then runs random-walk Metropolis and draws ``(B_full, Sigma, L)`` for each
    retained hyperparameter draw.

    Args:
        y: Raw data of shape ``(T_full, n_vars)`` (levels, before lag trimming).
        n_lags: Number of lags ``p``.
        prior: The conjugate :class:`~impulso.priors.NIWPrior`.
        volatility: A volatility adapter exposing ``hyperparameter_priors`` and ``log_scales``,
            or ``None`` for a constant-variance conjugate VAR.
        draws: Number of retained Metropolis draws.
        tune: Number of warm-up (adaptation) iterations, discarded.
        seed: Seed for the single :class:`numpy.random.Generator` driving the whole routine.

    Returns:
        Dictionary with:

        * ``hyperparameters``: ``{name: array of shape (draws,)}`` for each free hyperparameter.
        * ``B_full``: coefficient draws of shape ``(draws, n_vars, 1 + n_vars * n_lags)``.
        * ``Sigma``: base-covariance draws of shape ``(draws, n_vars, n_vars)``.
        * ``L``: lower-triangular Cholesky draws of ``Sigma`` (``Sigma = L @ L.T``).
        * ``acceptance_rate``: Metropolis acceptance rate over the retained ``draws``.
        * ``mode``: ``{name: float}`` the transformed-posterior mode of each hyperparameter.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    response, regressors = _design(y, n_lags)
    n_obs, n_vars = response.shape
    n_coeff = regressors.shape[1]
    sigma = ar1_residual_sd(y)

    specs = _param_specs(prior, volatility)
    d = len(specs)

    # Fast path: no free hyperparameters -> draw directly from the fixed-prior posterior.
    if d == 0:
        dummies_y, dummies_x = prior.build_dummies(y, n_lags, sigma, tightness=prior.tightness)
        drawn = draw_niw(niw_posterior(response, regressors, dummies_y, dummies_x), draws, rng)
        return {
            "hyperparameters": {},
            "B_full": drawn["B_full"],
            "Sigma": drawn["Sigma"],
            "L": drawn["L"],
            "acceptance_rate": 1.0,
            "mode": {},
        }

    los = np.array([s.lo for s in specs])
    his = np.array([s.hi for s in specs])
    lam_free = any(s.is_lambda for s in specs)
    if not lam_free:
        fixed_dummies = prior.build_dummies(y, n_lags, sigma, tightness=prior.tightness)

    def unpack(u: np.ndarray) -> tuple[np.ndarray, float, dict[str, float], float]:
        """Return (constrained x, lambda, vol theta, log-prior + log-jacobian)."""
        x = np.array([_to_constrained(u[i], los[i], his[i]) for i in range(d)])
        if not np.all(np.isfinite(x)):
            return x, prior.tightness, {}, -np.inf
        log_pj = 0.0
        lam = prior.tightness
        vol_theta: dict[str, float] = {}
        for i, spec in enumerate(specs):
            log_pj += spec.logpdf(float(x[i])) + _log_jacobian(u[i], los[i], his[i])
            if spec.is_lambda:
                lam = float(x[i])
            else:
                vol_theta[spec.name] = float(x[i])
        return x, lam, vol_theta, log_pj

    def log_target(u: np.ndarray) -> float:
        _, lam, vol_theta, log_pj = unpack(u)
        if not np.isfinite(log_pj):
            return -np.inf
        if lam_free:
            dummies_y, dummies_x = prior.build_dummies(y, n_lags, sigma, tightness=lam)
        else:
            dummies_y, dummies_x = fixed_dummies
        log_scales = None if volatility is None else volatility.log_scales(vol_theta, n_obs)
        lml = log_marginal_likelihood(response, regressors, dummies_y, dummies_x, log_scales=log_scales)
        value = lml + log_pj
        return value if np.isfinite(value) else -np.inf

    def neg_log_target(u: np.ndarray) -> float:
        value = log_target(np.asarray(u, dtype=float))
        return -value if np.isfinite(value) else 1e12

    # --- mode
    u0 = np.array([_to_unconstrained(specs[i].init, los[i], his[i]) for i in range(d)])
    result = optimize.minimize(
        neg_log_target,
        u0,
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000, "maxfev": 10000},
    )
    u_mode = np.asarray(result.x, dtype=float)

    # --- proposal from the Laplace covariance
    cov = _laplace_cov(_num_hessian(neg_log_target, u_mode))
    chol = np.linalg.cholesky(cov)

    # --- random-walk Metropolis with light diminishing adaptation during tuning
    target_acc = 0.25
    step = 2.38 / np.sqrt(d)
    u = u_mode.copy()
    current = log_target(u)
    kept = np.empty((draws, d))
    accepted_draws = 0
    for it in range(tune + draws):
        proposal = u + step * (chol @ rng.standard_normal(d))
        cand = log_target(proposal)
        accept = np.log(rng.random()) < cand - current
        if accept:
            u, current = proposal, cand
        if it < tune:
            gain = min(0.1, 1.0 / (it + 1) ** 0.6)
            step *= float(np.exp(gain * (float(accept) - target_acc)))
        else:
            kept[it - tune] = u
            accepted_draws += int(accept)

    # --- coefficient draws + hyperparameter arrays
    hyper = {spec.name: np.empty(draws) for spec in specs}
    b_full = np.empty((draws, n_vars, n_coeff))
    sigma_draws = np.empty((draws, n_vars, n_vars))
    chol_draws = np.empty((draws, n_vars, n_vars))
    for j in range(draws):
        x, lam, vol_theta, _ = unpack(kept[j])
        for i, spec in enumerate(specs):
            hyper[spec.name][j] = x[i]
        if lam_free:
            dummies_y, dummies_x = prior.build_dummies(y, n_lags, sigma, tightness=lam)
        else:
            dummies_y, dummies_x = fixed_dummies
        log_scales = None if volatility is None else volatility.log_scales(vol_theta, n_obs)
        ys, xs = _rescale(response, regressors, log_scales)
        drawn = draw_niw(niw_posterior(ys, xs, dummies_y, dummies_x), 1, rng)
        b_full[j] = drawn["B_full"][0]
        sigma_draws[j] = drawn["Sigma"][0]
        chol_draws[j] = drawn["L"][0]

    _, _, _, _ = unpack(u_mode)  # ensure mode is in-support
    mode = {spec.name: _to_constrained(u_mode[i], los[i], his[i]) for i, spec in enumerate(specs)}

    return {
        "hyperparameters": hyper,
        "B_full": b_full,
        "Sigma": sigma_draws,
        "L": chol_draws,
        "acceptance_rate": accepted_draws / draws,
        "mode": mode,
    }
