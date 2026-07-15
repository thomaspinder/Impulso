"""Conjugate Normal-Inverse-Wishart VAR engine (Giannone-Lenza-Primiceri, 2015).

Pure NumPy/SciPy implementation of the closed-form conjugate Bayesian VAR used to
reproduce Lenza & Primiceri (2020). The prior is the natural-conjugate Minnesota
prior expressed through Banbura-Giannone-Reichlin (2010) dummy observations, so the
posterior is available in closed form and its mean is an ordinary least-squares fit
on the data stacked with the dummies.

Conventions used throughout this module:

* The regressor matrix ``X`` has a **leading constant column of ones**; with ``n``
  variables and ``p`` lags it has ``k = 1 + n * p`` columns.
* The coefficient matrix ``B_full`` is ``(n, k)``: row ``i`` holds equation ``i``'s
  coefficients, column ``0`` is the intercept and columns ``1:`` are the lag
  coefficients ordered ``lag1(var0..var_{n-1}), lag2(...), ...``.
* The inverse-Wishart prior degrees of freedom are fixed to ``d0 = n + 2`` (the
  smallest value guaranteeing a finite prior mean of ``Sigma``), exactly as in
  Giannone-Lenza-Primiceri (2015). The dummy observations supply the prior location
  ``B0``, coefficient covariance ``Omega0`` and scale ``Psi0``.

The closed-form log marginal likelihood implemented here equals the reference GLP
factorisation (cross-checked against the MATLAB port at
github.com/Allisterh/Large-BVAR-Python-codes-) and an independent matrix-t /
multivariate-t evaluation to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import multigammaln  # ty: ignore[unresolved-import]
from scipy.stats import invwishart

# Diffuse prior variance on the intercept (GLP ``set_priors`` default ``Vc = 10e6``).
_CONST_PRIOR_VAR: float = 10e6


def ar1_residual_sd(y: np.ndarray) -> np.ndarray:
    """Per-variable residual standard deviation of a univariate AR(1) fit.

    This is the ``sigma`` scale consumed by :func:`minnesota_dummies`: each series is
    regressed on a constant and its own first lag, and the (unbiased) residual
    standard deviation is returned.

    Args:
        y: Data array of shape ``(T, n)``.

    Returns:
        Array of shape ``(n,)`` with the AR(1) residual standard deviation of each
        column.
    """
    y = np.asarray(y, dtype=float)
    t_obs = y.shape[0]
    lhs = y[1:]
    rhs = np.column_stack([np.ones(t_obs - 1), np.zeros(t_obs - 1)])
    sds = np.empty(y.shape[1])
    for i in range(y.shape[1]):
        rhs[:, 1] = y[:-1, i]
        beta, *_ = np.linalg.lstsq(rhs, lhs[:, i], rcond=None)
        resid = lhs[:, i] - rhs @ beta
        sds[i] = np.sqrt(resid @ resid / max(t_obs - 1 - 2, 1))
    return sds


def minnesota_dummies(
    y: np.ndarray,
    n_lags: int,
    *,
    lam: float,
    decay: float,
    cross: float,
    sigma: np.ndarray,
    mu_sur: float | None = None,
    mu_soc: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Banbura-Giannone-Reichlin dummy observations for the conjugate Minnesota prior.

    The returned dummies ``(Yd, Xd)`` encode the whole prior: stacking them under the
    data yields the conjugate Normal-Inverse-Wishart posterior. Four blocks are always
    produced and two are optional:

    * **lag coefficients** (``n * n_lags`` rows) — the Minnesota tightness ``lam`` and
      lag decay. The prior variance of the coefficient on lag ``l`` of variable ``v``
      is ``cross * lam**2 / (l**decay * sigma[v]**2)``; the own first-lag prior mean is
      1 (a random-walk prior).
    * **covariance scale** (``n`` rows) — ``Xd = 0``, ``Yd = diag(sigma)`` so the
      inverse-Wishart prior scale is ``diag(sigma**2)``.
    * **intercept** (1 row) — a diffuse prior with variance ``10e6``.
    * **single unit root** (``mu_sur``, 1 row) and **sum of coefficients**
      (``mu_soc``, ``n`` rows) — optional, ``None`` disables the block.

    Args:
        y: Raw data of shape ``(T_full, n)``; only its first ``n_lags`` rows are used
            (their mean seeds the sum-of-coefficients and single-unit-root dummies).
        n_lags: Number of lags ``p``.
        lam: Overall Minnesota tightness ``lambda`` (prior standard deviation).
        decay: Lag-decay exponent on the prior *variance* (GLP ``alpha``; ``2`` gives a
            harmonic decay of the prior standard deviation).
        cross: Multiplier on the lag-coefficient prior variance. In the natural
            conjugate (Kronecker) prior the coefficient variance is shared across
            equations, so per-equation own/cross asymmetry is not representable — that
            requires the independent-Normal ``MinnesotaPrior`` + MCMC path. ``cross``
            acts as a shared lag-variance scale; ``cross = 1.0`` reproduces GLP (2015).
        sigma: Per-variable scale of shape ``(n,)`` (AR(1) residual sd), e.g. from
            :func:`ar1_residual_sd`.
        mu_sur: Single-unit-root (dummy-initial-observation) scale (GLP ``theta``); the
            dummy weight is ``1 / mu_sur``. ``None`` disables the prior.
        mu_soc: Sum-of-coefficients scale (GLP ``miu``); the dummy weight is
            ``1 / mu_soc``. ``None`` disables the prior.

    Returns:
        Tuple ``(Yd, Xd)`` with ``Yd`` of shape ``(T_d, n)`` and ``Xd`` of shape
        ``(T_d, 1 + n * n_lags)``.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float).ravel()
    n = y.shape[1]
    p = n_lags
    k = 1 + n * p
    y0 = y[:p].mean(axis=0)

    lags = np.arange(1, p + 1)
    # X magnitude per (lag, var): sqrt(1 / prior variance).
    mag = (lags[:, None] ** (decay / 2.0)) * sigma[None, :] / (lam * np.sqrt(cross))

    # Block A: Minnesota prior on the lag coefficients.
    x_lag = np.zeros((n * p, k))
    x_lag[np.arange(n * p), 1 + np.arange(n * p)] = mag.reshape(-1)
    y_lag = np.zeros((n * p, n))
    y_lag[np.arange(n), np.arange(n)] = mag[0]  # own first lag, prior mean 1

    # Block B: inverse-Wishart prior scale (regressors zero).
    x_cov = np.zeros((n, k))
    y_cov = np.diag(sigma)

    # Block C: diffuse intercept prior.
    x_const = np.zeros((1, k))
    x_const[0, 0] = 1.0 / np.sqrt(_CONST_PRIOR_VAR)
    y_const = np.zeros((1, n))

    x_blocks = [x_lag, x_cov, x_const]
    y_blocks = [y_lag, y_cov, y_const]

    if mu_sur is not None:
        w = 1.0 / mu_sur
        x_sur = np.zeros((1, k))
        x_sur[0, 0] = w
        x_sur[0, 1:] = w * np.tile(y0, p)
        x_blocks.append(x_sur)
        y_blocks.append((w * y0).reshape(1, n))

    if mu_soc is not None:
        w = 1.0 / mu_soc
        x_soc = np.zeros((n, k))
        x_soc[:, 1:] = w * np.tile(np.diag(y0), p)
        x_blocks.append(x_soc)
        y_blocks.append(w * np.diag(y0))

    return np.vstack(y_blocks), np.vstack(x_blocks)


@dataclass(frozen=True)
class NIWPosterior:
    """Posterior of a conjugate Normal-Inverse-Wishart VAR.

    Attributes:
        B_hat: Posterior mean coefficients of shape ``(n, k)`` — equal to the ordinary
            least-squares fit on the data stacked with the dummy observations.
        V: Posterior coefficient covariance factor ``Omega_T`` of shape ``(k, k)``;
            conditional on ``Sigma`` the coefficient covariance is ``Sigma (x) V``.
        S: Posterior inverse-Wishart scale ``Psi_T`` of shape ``(n, n)``.
        nu: Posterior inverse-Wishart degrees of freedom ``d_T = n + 2 + T``.
    """

    B_hat: np.ndarray
    V: np.ndarray
    S: np.ndarray
    nu: float


def niw_posterior(Y: np.ndarray, X: np.ndarray, Yd: np.ndarray, Xd: np.ndarray) -> NIWPosterior:
    """Conjugate Normal-Inverse-Wishart posterior from data and dummy observations.

    Args:
        Y: Response data of shape ``(T, n)``.
        X: Regressors of shape ``(T, k)`` with a leading constant column.
        Yd: Dummy responses of shape ``(T_d, n)`` from :func:`minnesota_dummies`.
        Xd: Dummy regressors of shape ``(T_d, k)`` from :func:`minnesota_dummies`.

    Returns:
        The :class:`NIWPosterior`. ``B_hat`` equals ``((X* ' X*)^-1 X* ' Y*).T`` on the
        stacked design ``[X; Xd]``, ``[Y; Yd]``.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = Y.shape[1]
    t_obs = Y.shape[0]

    ys = np.vstack([Y, np.asarray(Yd, dtype=float)])
    xs = np.vstack([X, np.asarray(Xd, dtype=float)])
    xtx = xs.T @ xs
    b_stacked = np.linalg.solve(xtx, xs.T @ ys)  # (k, n)
    resid = ys - xs @ b_stacked
    v = np.linalg.inv(xtx)
    return NIWPosterior(
        B_hat=b_stacked.T,
        V=0.5 * (v + v.T),  # symmetrise: inv of an ill-conditioned xtx is not exactly symmetric
        S=resid.T @ resid,
        nu=float(n + 2 + t_obs),
    )


def log_marginal_likelihood(
    Y: np.ndarray,
    X: np.ndarray,
    Yd: np.ndarray,
    Xd: np.ndarray,
    *,
    log_scales: np.ndarray | None = None,
) -> float:
    """Closed-form log marginal likelihood of the conjugate VAR (GLP 2015).

    The prior is defined by the dummy observations with inverse-Wishart degrees of
    freedom ``d0 = n + 2``. The value is the conjugate matrix-t marginal likelihood

    ``-nT/2 log(pi) + logGamma_n(d_T/2) - logGamma_n(d0/2)
      + n/2 (log|Omega_T| - log|Omega_0|) + d0/2 log|Psi_0| - d_T/2 log|Psi_T|``

    with ``d_T = d0 + T``.

    Args:
        Y: Response data of shape ``(T, n)``.
        X: Regressors of shape ``(T, k)`` with a leading constant column.
        Yd: Dummy responses of shape ``(T_d, n)``.
        Xd: Dummy regressors of shape ``(T_d, k)``.
        log_scales: Optional per-observation log volatility scales of shape ``(T,)``.
            Row ``t`` of ``(Y, X)`` is rescaled by ``exp(-log_scales[t])`` and the
            change-of-variables Jacobian term ``-n * sum(log_scales)`` is added to the
            marginal likelihood. ``None`` (or an all-zero array) leaves the value
            unchanged.

    Returns:
        The log marginal likelihood as a float.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    Yd = np.asarray(Yd, dtype=float)
    Xd = np.asarray(Xd, dtype=float)
    n = Y.shape[1]
    t_obs = Y.shape[0]

    jacobian = 0.0
    if log_scales is not None:
        scales = np.asarray(log_scales, dtype=float).ravel()
        weights = np.exp(-scales)[:, None]
        Y = Y * weights
        X = X * weights
        jacobian = -n * float(scales.sum())

    d0 = n + 2
    d_t = d0 + t_obs
    ys = np.vstack([Y, Yd])
    xs = np.vstack([X, Xd])

    xdtxd = Xd.T @ Xd  # Omega_0^{-1}
    xstxs = xs.T @ xs  # Omega_T^{-1}
    b0 = np.linalg.solve(xdtxd, Xd.T @ Yd)
    b_t = np.linalg.solve(xstxs, xs.T @ ys)
    resid0 = Yd - Xd @ b0
    resid_t = ys - xs @ b_t
    psi0 = resid0.T @ resid0
    psi_t = resid_t.T @ resid_t

    log_det = lambda mat: np.linalg.slogdet(mat)[1]
    log_ml = (
        -0.5 * n * t_obs * np.log(np.pi)
        + multigammaln(d_t / 2.0, n)
        - multigammaln(d0 / 2.0, n)
        + 0.5 * n * (log_det(xdtxd) - log_det(xstxs))
        + 0.5 * d0 * log_det(psi0)
        - 0.5 * d_t * log_det(psi_t)
    )
    return float(log_ml + jacobian)


def _pd_factor(matrix: np.ndarray) -> np.ndarray:
    """Symmetric factor ``A`` such that ``A @ A.T`` reconstructs *matrix*.

    Returns the lower-triangular Cholesky factor when *matrix* is comfortably positive
    definite, and falls back to a clipped eigendecomposition at the positive-definite
    boundary. The coefficient covariance ``V = inv(Xs' Xs)`` turns near-singular when a
    loose prior (large ``lambda``) leaves coefficient directions weakly identified, so
    ``inv`` can return a slightly-asymmetric, boundary-PD matrix that a plain Cholesky
    rejects; the eigendecomposition clips numerically-negative eigenvalues to zero
    (those directions are fully pinned) without the bias that additive jitter adds.
    """
    matrix = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


def draw_niw(posterior: NIWPosterior, n_draws: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw ``(B_full, Sigma, L)`` from a conjugate Normal-Inverse-Wishart posterior.

    Each draw samples ``Sigma ~ IW(S, nu)`` then the coefficients from the matrix-normal
    ``B | Sigma ~ MN(B_hat, V, Sigma)``.

    Args:
        posterior: A :class:`NIWPosterior`.
        n_draws: Number of draws.
        rng: NumPy random generator.

    Returns:
        Dictionary with ``B_full`` of shape ``(n_draws, n, k)``, ``Sigma`` of shape
        ``(n_draws, n, n)`` and ``L`` of shape ``(n_draws, n, n)``, the lower-triangular
        Cholesky factor of each ``Sigma`` (``Sigma = L @ L.T``).
    """
    b_hat = posterior.B_hat
    n, k = b_hat.shape

    sigma = np.asarray(invwishart.rvs(df=posterior.nu, scale=posterior.S, size=n_draws, random_state=rng)).reshape(
        n_draws, n, n
    )
    chol_sigma = np.linalg.cholesky(sigma)
    chol_v = _pd_factor(posterior.V)
    noise = rng.standard_normal((n_draws, n, k))
    # MN(B_hat, V, Sigma): row covariance Sigma, column covariance V.
    b_full = b_hat[None] + chol_sigma @ noise @ chol_v.T
    return {"B_full": b_full, "Sigma": sigma, "L": chol_sigma}


def split_intercept(B_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a full coefficient matrix into its intercept and lag blocks.

    Args:
        B_full: Coefficients of shape ``(..., n, k)`` with ``k = 1 + n * n_lags`` and a
            leading intercept column.

    Returns:
        Tuple ``(intercept, B_lags)`` with ``intercept`` of shape ``(..., n)`` and
        ``B_lags`` of shape ``(..., n, n * n_lags)``.
    """
    B_full = np.asarray(B_full)
    return B_full[..., :, 0], B_full[..., :, 1:]
