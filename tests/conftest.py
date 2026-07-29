"""Shared test fixtures for Impulso."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso.data import VARData

# --------------- Raw data helpers ---------------


@pytest.fixture
def rng():
    """Deterministic RNG for test reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_endog(rng):
    """3 variables, 100 observations."""
    return rng.standard_normal((100, 3))


@pytest.fixture
def sample_index():
    return pd.date_range("2000-01-01", periods=100, freq="QS")


@pytest.fixture
def endog_names():
    return ["gdp", "inflation", "rate"]


# --------------- VARData fixtures ---------------


@pytest.fixture
def var_data_3v(sample_endog, sample_index, endog_names):
    """VARData with 3 endogenous variables, 100 obs."""
    return VARData(endog=sample_endog, endog_names=endog_names, index=sample_index)


@pytest.fixture
def var_data_2v():
    """VAR(1) DGP with 2 endogenous variables, 200 obs.

    Used across fitted, identified, and lag_selection tests.
    """
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


@pytest.fixture
def var_data_3v_dgp2():
    """VAR(2) DGP with 3 endogenous variables, 200 obs.

    Used for lag selection tests.
    """
    rng = np.random.default_rng(42)
    T, n = 200, 3
    y = np.zeros((T, n))
    for t in range(2, T):
        y[t] = 0.5 * y[t - 1] + 0.2 * y[t - 2] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2", "y3"], index=index)


# --------------- Synthetic InferenceData for fast tests ---------------


@pytest.fixture
def synthetic_idata_2v():
    """Synthetic InferenceData mimicking a fitted 2-var VAR(1).

    No MCMC required. 2 chains, 50 draws, 2 variables, 1 lag.
    B shape: (2, 50, 2, 2) -- coefficient matrix
    intercept shape: (2, 50, 2)
    Sigma shape: (2, 50, 2, 2) -- positive definite covariance
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_vars, n_lags = 2, 50, 2, 1

    B = rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.3
    intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01

    # Positive definite Sigma via A @ A.T + I, plus its Cholesky factor L
    # so the fixture mirrors a real Constant-adapter posterior.
    sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
    L = np.zeros_like(sigma)
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars)) * 0.5
            sigma[c, d] = A @ A.T + np.eye(n_vars)
            L[c, d] = np.linalg.cholesky(sigma[c, d])

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(
            sigma,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        ),
        "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
    })
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def synthetic_idata_2v_t(synthetic_idata_2v):
    """`synthetic_idata_2v` plus a constant `nu` of 5.0, for Student-t tests.

    A *sibling* of `synthetic_idata_2v` rather than a modification of it, so
    Gaussian tests keep the exact posterior they have always had. `nu` is
    constant across (chain, draw) on purpose: it makes exact distributional
    assertions about the innovation draws possible. Draw-varying `nu` is
    covered by the unit tests in test_error_distributions.py.
    """
    posterior = synthetic_idata_2v.posterior.copy()
    n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    posterior["nu"] = xr.DataArray(
        np.full((n_chains, n_draws), 5.0),
        dims=["chain", "draw"],
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def permanent_transitory_2v():
    """Exact-arithmetic 2-var VAR(1) with a known long-run structure.

    Built backwards from the answer so the long-run identification has a
    closed form to be checked against:

        A_1 = [[0.5, 0.1], [0.2, 0.4]]   (eigenvalues 0.6, 0.3 — stable)
        M   = I - A_1                    (det 0.28)
        C1  = M^-1                       (long-run multiplier)
        G   = [[1.0, 0.0], [0.5, 0.8]]   (imposed cumulative impact, G[0, 1] = 0)
        P   = M @ G = [[0.45, -0.08], [0.10, 0.48]]
        Sigma = P @ P.T

    `P` is deliberately not lower-triangular, so it is distinguishable
    from the Cholesky factor of Sigma.

    Returns:
        Dict with keys `A1`, `M`, `C1`, `G`, `P_true`, `Sigma`, `L`
        (Cholesky factor of Sigma) and `idata` — an InferenceData with 2
        chains x 50 draws whose draws all equal the truth, laid out like
        `synthetic_idata_2v`.
    """
    n_chains, n_draws, n_vars = 2, 50, 2

    A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
    M = np.eye(n_vars) - A1
    C1 = np.linalg.inv(M)
    G = np.array([[1.0, 0.0], [0.5, 0.8]])
    P_true = M @ G
    Sigma = P_true @ P_true.T
    L = np.linalg.cholesky(Sigma)

    shape = (n_chains, n_draws, n_vars, n_vars)
    B = np.broadcast_to(A1, shape).copy()
    intercept = np.zeros((n_chains, n_draws, n_vars))
    sigma_draws = np.broadcast_to(Sigma, shape).copy()
    L_draws = np.broadcast_to(L, shape).copy()

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(
            sigma_draws,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        ),
        "L": xr.DataArray(L_draws, dims=["chain", "draw", "var1", "var2"]),
    })
    return {
        "A1": A1,
        "M": M,
        "C1": C1,
        "G": G,
        "P_true": P_true,
        "Sigma": Sigma,
        "L": L_draws,
        "idata": az.InferenceData(posterior=posterior),
    }


@pytest.fixture
def flat_band_accumulator():
    """Stand-in for `MaxShare._spectral_accumulator` returning `K = I`.

    Monkeypatch it onto the class to reduce the band variance form to
    `M = L' L`, so the eigenvalue ratio — and with it the degeneracy
    warning — is steered entirely by the caller's `L`. No draw comes back
    singular or explosive, so the weak-identification warning is the only
    one that can fire.
    """

    def _accumulator(self, posterior, n_lags, n_vars, target_index):
        shape = posterior["B"].values.shape[:2]
        return (
            np.broadcast_to(np.eye(n_vars), (*shape, n_vars, n_vars)).copy(),
            np.zeros(shape, dtype=bool),
            np.zeros(shape),
            np.ones(shape),
        )

    return _accumulator


# --------------- SV fixtures ---------------


@pytest.fixture
def sv_series_rw():
    """1-D series simulated from RW log-volatility SV DGP.

    T=500, sigma_eta=0.1. Used by slow recovery tests. Returns a
    dict with 'y', 'h_true', 'mu_true', 'sigma_eta_true' so
    recovery tests can compare to the truth.
    """
    rng = np.random.default_rng(42)
    T = 500
    sigma_eta_true = 0.1
    mu_true = 0.0
    h_true = np.zeros(T)
    h_true[0] = 0.0
    for t in range(1, T):
        h_true[t] = h_true[t - 1] + sigma_eta_true * rng.standard_normal()
    y = mu_true + np.exp(0.5 * h_true) * rng.standard_normal(T)
    return {
        "y": y,
        "h_true": h_true,
        "mu_true": mu_true,
        "sigma_eta_true": sigma_eta_true,
    }


@pytest.fixture
def sv_data_rw(sv_series_rw):
    """SVData wrapping the RW SV DGP series."""
    from impulso.sv.data import SVData

    index = pd.date_range("1980-01-01", periods=len(sv_series_rw["y"]), freq="MS")
    return SVData(y=sv_series_rw["y"], name="sim", index=index)


@pytest.fixture
def synthetic_sv_idata():
    """Synthetic InferenceData mimicking a fitted random-walk SV posterior.

    No MCMC required. 2 chains, 50 draws, T=100.
    posterior["h"] shape: (2, 50, 100)
    posterior["mu"] shape: (2, 50)
    posterior["sigma_eta"] shape: (2, 50)
    """
    rng = np.random.default_rng(123)
    n_chains, n_draws, T = 2, 50, 100

    # Simulate a plausible posterior around a mild vol path
    h_mean = np.linspace(-0.5, 0.5, T)
    h = h_mean[None, None, :] + 0.1 * rng.standard_normal((n_chains, n_draws, T))
    mu = 0.01 * rng.standard_normal((n_chains, n_draws))
    sigma_eta = 0.1 + 0.02 * np.abs(rng.standard_normal((n_chains, n_draws)))

    posterior = xr.Dataset({
        "h": xr.DataArray(h, dims=["chain", "draw", "time"]),
        "mu": xr.DataArray(mu, dims=["chain", "draw"]),
        "sigma_eta": xr.DataArray(sigma_eta, dims=["chain", "draw"]),
    })
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def synthetic_sv_idata_2v():
    """Synthetic InferenceData mimicking a fitted multivariate SV model.

    Shape: 2 chains, 50 draws, T=20, n_vars=2.
    Contains: h (2, 50, 20, 2), R_chol (2, 50, 2, 2), v0_mu, v1_mu,
    v0_sigma_eta, v1_sigma_eta.
    """
    rng = np.random.default_rng(0)
    n_chains, n_draws, T, n_vars = 2, 50, 20, 2

    h = rng.standard_normal((n_chains, n_draws, T, n_vars)) * 0.3 - 1.0

    R_chol = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars))
            R_chol[c, d] = np.linalg.cholesky(A @ A.T + np.eye(n_vars))
    diag_inv = 1.0 / np.diagonal(R_chol, axis1=-2, axis2=-1)[:, :, :, None]
    R_chol = R_chol * diag_inv

    posterior = xr.Dataset({
        "h": (("chain", "draw", "time", "variable"), h),
        "R_chol": (("chain", "draw", "i", "j"), R_chol),
        "v0_h": (("chain", "draw", "time"), h[:, :, :, 0]),
        "v1_h": (("chain", "draw", "time"), h[:, :, :, 1]),
        "v0_mu": (("chain", "draw"), rng.standard_normal((n_chains, n_draws))),
        "v1_mu": (("chain", "draw"), rng.standard_normal((n_chains, n_draws))),
        "v0_sigma_eta": (("chain", "draw"), np.abs(rng.standard_normal((n_chains, n_draws)))),
        "v1_sigma_eta": (("chain", "draw"), np.abs(rng.standard_normal((n_chains, n_draws)))),
    })
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def single_driver_2v():
    """Exact-arithmetic 2-var VAR(1) where one shock drives `y2` entirely.

    Built backwards from the answer so max-share identification has a
    closed form to be checked against:

        A_1    = [[0.5, 0.0], [0.0, 0.3]]   (diagonal — rows decouple)
        P_true = [[0.3, 0.9], [0.7, 0.0]]   (row 1 loads only on shock 0)
        Sigma  = P_true @ P_true.T = [[0.90, 0.21], [0.21, 0.49]]

    Because `A_1` is diagonal the transfer function is diagonal too, so
    row 1 of `C(w) P_true` is `c_2(w) * [0.7, 0]` at *every* frequency:
    shock 0 explains 100% of `y2`'s variance in every band, and the
    unique maximiser is `p = [0.3, 0.7]` exactly. `P_true` is deliberately
    not lower-triangular, so it is distinguishable from `chol(Sigma)`.

    Returns:
        Dict with keys `A1`, `P_true`, `Sigma`, `L` (Cholesky factor of
        Sigma, broadcast over draws) and `idata` — an InferenceData with
        2 chains x 50 draws whose draws all equal the truth, laid out like
        `synthetic_idata_2v`.
    """
    n_chains, n_draws, n_vars = 2, 50, 2

    A1 = np.array([[0.5, 0.0], [0.0, 0.3]])
    P_true = np.array([[0.3, 0.9], [0.7, 0.0]])
    Sigma = P_true @ P_true.T
    L = np.linalg.cholesky(Sigma)

    shape = (n_chains, n_draws, n_vars, n_vars)
    posterior = xr.Dataset({
        "B": xr.DataArray(np.broadcast_to(A1, shape).copy(), dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(np.zeros((n_chains, n_draws, n_vars)), dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(
            np.broadcast_to(Sigma, shape).copy(),
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        ),
        "L": xr.DataArray(np.broadcast_to(L, shape).copy(), dims=["chain", "draw", "var1", "var2"]),
    })
    return {
        "A1": A1,
        "P_true": P_true,
        "Sigma": Sigma,
        "L": np.broadcast_to(L, shape).copy(),
        "idata": az.InferenceData(posterior=posterior),
    }
