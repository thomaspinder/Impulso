"""Tests for the conjugate Normal-Inverse-Wishart VAR engine.

Three hard gates from the build contract:

1. The posterior mean equals the stacked dummy-observation OLS solution.
2. ``log_marginal_likelihood`` matches an independent matrix-t / multivariate-t
   evaluation on a 1-2 variable, 1-lag, few-obs model to 1e-8.
3. Jacobian exactness: all-zero ``log_scales`` equals the no-scale ML, and a nonzero
   ``log_scales`` shifts the ML by exactly ``-n_vars * sum(log_scales)``.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import multigammaln  # ty: ignore[unresolved-import]
from scipy.stats import multivariate_t as mvt

from impulso._conjugate import (
    NIWPosterior,
    _pd_factor,
    ar1_residual_sd,
    draw_niw,
    log_marginal_likelihood,
    minnesota_dummies,
    niw_posterior,
    split_intercept,
)
from impulso.priors import NIWPrior

# --------------------------------------------------------------------------- helpers


def _make_series(seed: int, n: int, t_full: int) -> np.ndarray:
    """Small persistent (random-walk-ish) synthetic series of shape (t_full, n)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((t_full, n)), axis=0) + 5.0


def _make_yx(series: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (Y, X) with a leading constant column, matching the module lag order."""
    t_full, _n = series.shape
    t_obs = t_full - n_lags
    y = series[n_lags:]
    cols = [np.ones((t_obs, 1))]
    for ell in range(1, n_lags + 1):
        cols.append(series[n_lags - ell : t_full - ell])
    return y, np.hstack(cols)


def _build(seed: int, n: int, t_obs: int, **kw):
    """Convenience: series -> (Y, X, Yd, Xd) for a 1-lag model."""
    series = _make_series(seed, n, t_obs + 1)
    sigma = ar1_residual_sd(series)
    y, x = _make_yx(series, 1)
    yd, xd = minnesota_dummies(
        series,
        1,
        lam=kw.get("lam", 0.3),
        decay=kw.get("decay", 2.0),
        cross=kw.get("cross", 1.0),
        sigma=sigma,
        mu_sur=kw.get("mu_sur"),
        mu_soc=kw.get("mu_soc"),
    )
    return y, x, yd, xd


def _matrix_t_logml(Y, X, Yd, Xd) -> float:
    """Independent log ML via the matrix-t density (T x T determinant form)."""
    n, t_obs = Y.shape[1], Y.shape[0]
    d0, d_t = n + 2, n + 2 + t_obs
    omega0 = np.linalg.inv(Xd.T @ Xd)
    b0 = omega0 @ Xd.T @ Yd
    psi0 = (Yd - Xd @ b0).T @ (Yd - Xd @ b0)
    a_mat = np.eye(t_obs) + X @ omega0 @ X.T
    err = Y - X @ b0
    quad = psi0 + err.T @ np.linalg.solve(a_mat, err)
    ld = lambda m: np.linalg.slogdet(m)[1]
    return float(
        -0.5 * n * t_obs * np.log(np.pi)
        + multigammaln(d_t / 2, n)
        - multigammaln(d0 / 2, n)
        - 0.5 * n * ld(a_mat)
        + 0.5 * d0 * ld(psi0)
        - 0.5 * d_t * ld(quad)
    )


def _sequential_logml(Y, X, Yd, Xd) -> float:
    """Independent log ML as a product of one-step conjugate predictive multivariate-t
    densities (recursive Bayesian updating; well-conditioned, no batch determinants)."""
    n = Y.shape[1]
    omega = np.linalg.inv(Xd.T @ Xd)
    b = omega @ Xd.T @ Yd
    r0 = Yd - Xd @ b
    psi = r0.T @ r0
    nu = float(n + 2)
    total = 0.0
    for t in range(Y.shape[0]):
        x, y = X[t], Y[t]
        pred_mean = b.T @ x
        s = 1.0 + x @ omega @ x
        df = nu - n + 1.0
        total += mvt.logpdf(y, loc=pred_mean, shape=(s / df) * psi, df=df)
        omega_x = omega @ x
        omega = omega - np.outer(omega_x, omega_x) / s
        resid = y - pred_mean
        b = b + np.outer(omega_x, resid) / s
        psi = psi + np.outer(resid, resid) / s
        nu += 1.0
    return float(total)


# --------------------------------------------------------------- gate 1: posterior mean


class TestPosteriorMean:
    """Gate 1 — posterior mean equals the stacked dummy-observation OLS solution."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("n_lags", [1, 2])
    def test_b_hat_equals_stacked_ols(self, n, n_lags):
        series = _make_series(seed=1, n=n, t_full=40)
        sigma = ar1_residual_sd(series)
        y, x = _make_yx(series, n_lags)
        yd, xd = minnesota_dummies(series, n_lags, lam=0.25, decay=2.0, cross=1.0, sigma=sigma)
        post = niw_posterior(y, x, yd, xd)

        ys, xs = np.vstack([y, yd]), np.vstack([x, xd])
        ols = np.linalg.solve(xs.T @ xs, xs.T @ ys)  # (k, n)
        assert post.B_hat.shape == (n, 1 + n * n_lags)
        assert np.allclose(post.B_hat, ols.T, atol=1e-10, rtol=0)

    def test_b_hat_equals_stacked_ols_with_sur_soc(self):
        series = _make_series(seed=2, n=2, t_full=60)
        sigma = ar1_residual_sd(series)
        y, x = _make_yx(series, 2)
        yd, xd = minnesota_dummies(series, 2, lam=0.2, decay=2.0, cross=1.0, sigma=sigma, mu_sur=1.0, mu_soc=1.0)
        post = niw_posterior(y, x, yd, xd)
        ys, xs = np.vstack([y, yd]), np.vstack([x, xd])
        ols = np.linalg.solve(xs.T @ xs, xs.T @ ys)
        assert np.allclose(post.B_hat, ols.T, atol=1e-10, rtol=0)


# ------------------------------------------------------- gate 2: marginal likelihood


class TestMarginalLikelihood:
    """Gate 2 — closed form matches independent matrix-t / multivariate-t to 1e-8."""

    @pytest.mark.parametrize("n", [1, 2])
    @pytest.mark.parametrize("t_obs", [5, 8, 12])
    def test_matches_sequential_predictive_t(self, n, t_obs):
        y, x, yd, xd = _build(seed=7, n=n, t_obs=t_obs)
        value = log_marginal_likelihood(y, x, yd, xd)
        # Independent evaluation: product of one-step predictive multivariate-t densities.
        assert abs(value - _sequential_logml(y, x, yd, xd)) < 1e-8

    @pytest.mark.parametrize("n", [1, 2])
    @pytest.mark.parametrize("t_obs", [5, 8, 12])
    def test_matches_matrix_t_density(self, n, t_obs):
        y, x, yd, xd = _build(seed=11, n=n, t_obs=t_obs)
        value = log_marginal_likelihood(y, x, yd, xd)
        # Independent evaluation: batch matrix-t determinant form (T x T linear algebra).
        assert abs(value - _matrix_t_logml(y, x, yd, xd)) <= 1e-8 * abs(value)

    def test_matches_scipy_multivariate_t_univariate(self):
        y, x, yd, xd = _build(seed=3, n=1, t_obs=6)
        omega0 = np.linalg.inv(xd.T @ xd)
        b0 = omega0 @ xd.T @ yd
        psi0 = ((yd - xd @ b0).T @ (yd - xd @ b0)).item()
        d0 = 3  # n + 2, n = 1
        a_mat = np.eye(y.shape[0]) + x @ omega0 @ x.T
        scipy_val = mvt.logpdf(y.ravel(), loc=(x @ b0).ravel(), shape=(psi0 / d0) * a_mat, df=d0)
        assert abs(log_marginal_likelihood(y, x, yd, xd) - scipy_val) < 1e-7


# ------------------------------------------------------------- gate 3: Jacobian exactness


class TestJacobian:
    """Gate 3 — data rescaling plus the -n_vars * sum(log_scales) Jacobian term."""

    @pytest.mark.parametrize("n", [1, 2])
    def test_zero_log_scales_equals_no_scale(self, n):
        y, x, yd, xd = _build(seed=5, n=n, t_obs=10)
        base = log_marginal_likelihood(y, x, yd, xd)
        zeros = log_marginal_likelihood(y, x, yd, xd, log_scales=np.zeros(y.shape[0]))
        assert base == pytest.approx(zeros, abs=1e-12)

    @pytest.mark.parametrize("n", [1, 2])
    def test_nonzero_log_scales_shift_is_exact(self, n):
        y, x, yd, xd = _build(seed=6, n=n, t_obs=10)
        rng = np.random.default_rng(99)
        v = rng.standard_normal(y.shape[0])
        # Core ML on the manually rescaled data (row t by exp(-v[t])), no Jacobian.
        core = log_marginal_likelihood(y * np.exp(-v)[:, None], x * np.exp(-v)[:, None], yd, xd)
        shifted = log_marginal_likelihood(y, x, yd, xd, log_scales=v)
        assert (shifted - core) == pytest.approx(-n * v.sum(), abs=1e-9)


# --------------------------------------------------------------------- posterior draws


class TestDrawNIW:
    def test_shapes_and_cholesky(self):
        y, x, yd, xd = _build(seed=8, n=2, t_obs=50)
        post = niw_posterior(y, x, yd, xd)
        draws = draw_niw(post, n_draws=64, rng=np.random.default_rng(0))
        n, k = post.B_hat.shape
        assert draws["B_full"].shape == (64, n, k)
        assert draws["Sigma"].shape == (64, n, n)
        assert draws["L"].shape == (64, n, n)
        assert np.all(np.isfinite(draws["B_full"]))
        assert np.allclose(draws["L"] @ draws["L"].transpose(0, 2, 1), draws["Sigma"])
        # Lower-triangular Cholesky factor.
        assert np.allclose(np.triu(draws["L"], k=1), 0.0)

    def test_monte_carlo_recovers_posterior_moments(self):
        y, x, yd, xd = _build(seed=9, n=2, t_obs=200)
        post = niw_posterior(y, x, yd, xd)
        draws = draw_niw(post, n_draws=40_000, rng=np.random.default_rng(1))
        n = post.B_hat.shape[0]
        assert np.allclose(draws["B_full"].mean(0), post.B_hat, atol=5e-3)
        # E[Sigma] = S / (nu - n - 1) for the inverse-Wishart posterior.
        assert np.allclose(draws["Sigma"].mean(0), post.S / (post.nu - n - 1), atol=5e-3)


class TestSplitIntercept:
    def test_splits_intercept_and_lags(self):
        rng = np.random.default_rng(0)
        n, n_lags = 3, 2
        b_full = rng.standard_normal((5, 4, n, 1 + n * n_lags))
        intercept, b_lags = split_intercept(b_full)
        assert intercept.shape == (5, 4, n)
        assert b_lags.shape == (5, 4, n, n * n_lags)
        assert np.array_equal(intercept, b_full[..., 0])
        assert np.array_equal(b_lags, b_full[..., 1:])

    def test_roundtrip_recovers_full(self):
        b_full = np.arange(2 * 5).reshape(2, 5).astype(float)  # (n=2, k=5), n_lags=2
        intercept, b_lags = split_intercept(b_full)
        assert np.array_equal(np.concatenate([intercept[:, None], b_lags], axis=1), b_full)


# ----------------------------------------------------------- dummy observations / prior


class TestMinnesotaDummies:
    def test_known_small_case_shapes_and_values(self):
        series = _make_series(seed=4, n=2, t_full=20)
        sigma = np.array([1.0, 2.0])
        yd, xd = minnesota_dummies(series, 1, lam=0.2, decay=2.0, cross=1.0, sigma=sigma)
        # 2 lag rows + 2 covariance rows + 1 intercept row.
        assert yd.shape == (5, 2)
        assert xd.shape == (5, 3)
        # Block A: X magnitude = ell^(decay/2) * sigma_v / (lam * sqrt(cross)).
        assert xd[0, 1] == pytest.approx(1.0 / 0.2)  # var 0, sigma 1
        assert xd[1, 2] == pytest.approx(2.0 / 0.2)  # var 1, sigma 2
        # Own first-lag prior mean is 1: Yd = magnitude.
        assert yd[0, 0] == pytest.approx(1.0 / 0.2)
        assert yd[1, 1] == pytest.approx(2.0 / 0.2)
        assert np.allclose(yd[0], [1.0 / 0.2, 0.0])
        # Block B: X = 0, Y = diag(sigma).
        assert np.allclose(xd[2:4], 0.0)
        assert np.allclose(yd[2:4], np.diag(sigma))
        # Block C: diffuse intercept.
        assert xd[4, 0] == pytest.approx(1.0 / np.sqrt(10e6))
        assert np.allclose(yd[4], 0.0)

    def test_prior_variances_are_glp_minnesota(self):
        sigma = np.array([1.5, 0.5])
        series = _make_series(seed=1, n=2, t_full=10)
        lam, decay = 0.3, 2.0
        _yd, xd = minnesota_dummies(series, 2, lam=lam, decay=decay, cross=1.0, sigma=sigma)
        omega = 1.0 / np.diag(xd.T @ xd)  # prior coefficient variances
        # Intercept: diffuse.
        assert omega[0] == pytest.approx(10e6)
        # lag l, var v -> lam**2 / (l**decay * sigma_v**2).
        for ell in (1, 2):
            for v in (0, 1):
                col = 1 + (ell - 1) * 2 + v
                assert omega[col] == pytest.approx(lam**2 / (ell**decay * sigma[v] ** 2))

    def test_optional_blocks_change_row_count(self):
        series = _make_series(seed=1, n=2, t_full=10)
        sigma = np.array([1.0, 1.0])
        base, _ = minnesota_dummies(series, 1, lam=0.2, decay=2.0, cross=1.0, sigma=sigma)
        with_sur, _ = minnesota_dummies(series, 1, lam=0.2, decay=2.0, cross=1.0, sigma=sigma, mu_sur=1.0)
        with_both, _ = minnesota_dummies(series, 1, lam=0.2, decay=2.0, cross=1.0, sigma=sigma, mu_sur=1.0, mu_soc=1.0)
        assert with_sur.shape[0] == base.shape[0] + 1  # single-unit-root: 1 row
        assert with_both.shape[0] == base.shape[0] + 1 + 2  # + sum-of-coefficients: n rows


class TestNIWPrior:
    def test_defaults_reproduce_glp(self):
        prior = NIWPrior()
        assert prior.tightness == 0.2
        assert prior.decay == 2.0
        assert prior.cross_shrinkage == 1.0
        assert prior.sum_of_coefficients is None
        assert prior.single_unit_root is None
        assert prior.select is False

    def test_frozen(self):
        prior = NIWPrior()
        with pytest.raises(ValidationError):
            prior.tightness = 0.5

    @pytest.mark.parametrize("bad", [0.0, -0.1])
    def test_rejects_nonpositive_tightness(self, bad):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NIWPrior(tightness=bad)

    def test_build_dummies_matches_module_function(self):
        series = _make_series(seed=2, n=2, t_full=30)
        sigma = ar1_residual_sd(series)
        prior = NIWPrior(tightness=0.25, decay=2.0, cross_shrinkage=1.0)
        yd_p, xd_p = prior.build_dummies(series, 1, sigma)
        yd_m, xd_m = minnesota_dummies(series, 1, lam=0.25, decay=2.0, cross=1.0, sigma=sigma)
        assert np.array_equal(yd_p, yd_m)
        assert np.array_equal(xd_p, xd_m)

    def test_build_dummies_tightness_override(self):
        series = _make_series(seed=2, n=2, t_full=30)
        sigma = ar1_residual_sd(series)
        prior = NIWPrior(tightness=0.25)
        _, xd_override = prior.build_dummies(series, 1, sigma, tightness=0.5)
        _, xd_direct = minnesota_dummies(series, 1, lam=0.5, decay=2.0, cross=1.0, sigma=sigma)
        assert np.array_equal(xd_override, xd_direct)

    def test_build_dummies_computes_sigma_when_absent(self):
        series = _make_series(seed=2, n=2, t_full=30)
        prior = NIWPrior()
        yd_auto, xd_auto = prior.build_dummies(series, 1)
        yd_exp, xd_exp = prior.build_dummies(series, 1, ar1_residual_sd(series))
        assert np.array_equal(xd_auto, xd_exp)
        assert np.array_equal(yd_auto, yd_exp)


class TestNumericalRobustness:
    """draw_niw must survive the boundary-PD coefficient covariance that a loose prior
    (large sampled lambda) induces via ``V = inv(Xs' Xs)`` on a near-singular design."""

    @staticmethod
    def _boundary_posterior():
        rng = np.random.default_rng(0)
        k, n = 8, 2
        q, _ = np.linalg.qr(rng.standard_normal((k, k)))
        # Symmetric spectrum with a tiny negative eigenvalue -> plain Cholesky fails.
        eig = np.array([50.0, 5.0, 1.0, 0.3, 0.05, 1e-3, 1e-7, -1e-9])
        v = 0.5 * (((q * eig) @ q.T) + ((q * eig) @ q.T).T)
        s = np.array([[1.0, 0.3], [0.3, 2.0]])
        b_hat = rng.standard_normal((n, k))
        return NIWPosterior(B_hat=b_hat, V=v, S=s, nu=float(n + 2 + 300)), v

    def test_plain_cholesky_would_fail(self):
        _, v = self._boundary_posterior()
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(v)

    def test_draw_niw_survives_and_is_finite(self):
        post, _ = self._boundary_posterior()
        draws = draw_niw(post, n_draws=128, rng=np.random.default_rng(1))
        n, k = post.B_hat.shape
        assert draws["B_full"].shape == (128, n, k)
        assert np.all(np.isfinite(draws["B_full"]))
        assert np.all(np.isfinite(draws["Sigma"]))
        assert np.all(np.isfinite(draws["L"]))

    def test_draw_niw_recovers_moments_on_boundary(self):
        post, v = self._boundary_posterior()
        w, qe = np.linalg.eigh(v)
        v_clip = (qe * np.clip(w, 0.0, None)) @ qe.T
        draws = draw_niw(post, n_draws=80_000, rng=np.random.default_rng(2))
        n = post.B_hat.shape[0]
        assert np.allclose(draws["B_full"].mean(0), post.B_hat, atol=1e-2)
        # Marginal column covariance of equation 0 is E[Sigma_00] * V_clipped.
        emp = np.cov(draws["B_full"][:, 0, :].T)
        theo = (post.S[0, 0] / (post.nu - n - 1)) * v_clip
        assert np.allclose(emp, theo, atol=5e-3)

    def test_pd_factor_reconstructs_clipped_matrix(self):
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        eig = np.array([4.0, 1.0, 0.2, 1e-6, -1e-10])
        m = (q * eig) @ q.T
        factor = _pd_factor(m)
        w, qe = np.linalg.eigh(0.5 * (m + m.T))
        clipped = (qe * np.clip(w, 0.0, None)) @ qe.T
        assert np.allclose(factor @ factor.T, clipped, atol=1e-10)

    def test_pd_factor_matches_cholesky_when_pd(self):
        rng = np.random.default_rng(4)
        a = rng.standard_normal((6, 6))
        m = a @ a.T + np.eye(6)  # comfortably positive definite
        assert np.allclose(_pd_factor(m), np.linalg.cholesky(m))

    def test_niw_posterior_v_is_symmetric(self):
        # A very loose prior makes Xs' Xs ill-conditioned; V must still be symmetric.
        series = _make_series(seed=1, n=2, t_full=40)
        sigma = ar1_residual_sd(series)
        y, x = _make_yx(series, 2)
        yd, xd = minnesota_dummies(series, 2, lam=5.0, decay=2.0, cross=1.0, sigma=sigma)
        post = niw_posterior(y, x, yd, xd)
        assert np.array_equal(post.V, post.V.T)
