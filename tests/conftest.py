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


# --------------- Diagnostics posterior factory ---------------

# PyMC and nutpie spell almost every sampler statistic differently; `diverging`
# and `energy` are the only common ones, and max-treedepth saturation needs a
# name map. The factory can emit either shape so tests can prove the report
# reads the right key per backend and ignores the rest.
_MAX_TREEDEPTH_NAME = {False: "reached_max_treedepth", True: "maxdepth_reached"}
_PYMC_NOISE_STATS = ("tree_depth", "acceptance_rate", "lp", "step_size")
_NUTPIE_NOISE_STATS = ("depth", "mean_tree_accept", "logp", "step_size", "tuning")


def _coefficient_draws(rng, shape, explosive_frac, bad_coord):
    """Lag-coefficient draws centred on `0.5 * I`, with optional pathologies."""
    n_chains, n_draws, n_vars, n_coeff = shape
    base = np.zeros((n_vars, n_coeff))
    base[:, :n_vars] = 0.5 * np.eye(n_vars)
    if explosive_frac > 0:
        # Noiseless two-point mixture: every radius is exactly 0.5 or 1.2, so
        # the reported statistics are exact rather than approximate.
        B = np.broadcast_to(base, shape).copy()
        n_explosive = round(explosive_frac * n_draws)
        for chain in range(n_chains):
            B[chain, rng.permutation(n_draws)[:n_explosive], 0, 0] = 1.2
    else:
        B = base + 0.02 * rng.standard_normal(shape)
    if bad_coord is not None:
        row, col = bad_coord
        B[:, :, row, col] += np.arange(n_chains)[:, None] * 2.0
    return B


def _cholesky_draws(rng, sigma_sd, n_chains, n_draws, n_vars):
    """Lower-triangular Cholesky draws; the upper triangle is a structural zero."""
    L = np.zeros((n_chains, n_draws, n_vars, n_vars))
    diag = np.arange(n_vars)
    L[..., diag, diag] = sigma_sd
    for i in range(1, n_vars):
        for j in range(i):
            L[..., i, j] = 0.1 * rng.standard_normal((n_chains, n_draws))
    return L


def _flag_draws(n_chains, n_draws, count):
    """Boolean per-transition flags, `count` of them set at fixed positions."""
    flat = np.zeros(n_chains * n_draws, dtype=bool)
    flat[:count] = True
    return flat.reshape(n_chains, n_draws)


def _energy_draws(rng, n_chains, n_draws, energy_rho):
    """AR(1) energy trace whose E-BFMI is approximately `2 * (1 - energy_rho)`.

    `energy_rho=0.3` gives a healthy trace (E-BFMI near 1.4); pushing it
    toward 1 makes successive energies nearly identical, which is exactly the
    slow energy exploration a low E-BFMI reports.
    """
    energy = np.empty((n_chains, n_draws))
    energy[:, 0] = rng.standard_normal(n_chains)
    innovation = np.sqrt(1.0 - energy_rho**2)
    for t in range(1, n_draws):
        energy[:, t] = energy_rho * energy[:, t - 1] + innovation * rng.standard_normal(n_chains)
    return energy


def _sampler_stats(rng, n_chains, n_draws, divergences, treedepth_hits, energy_rho, nutpie_shaped):
    """A `sample_stats` group in either backend's shape, plus nutpie's warmup group."""
    stats = {
        "diverging": (("chain", "draw"), _flag_draws(n_chains, n_draws, divergences)),
        "energy": (("chain", "draw"), _energy_draws(rng, n_chains, n_draws, energy_rho)),
        _MAX_TREEDEPTH_NAME[nutpie_shaped]: (
            ("chain", "draw"),
            _flag_draws(n_chains, n_draws, treedepth_hits),
        ),
    }
    for name in _NUTPIE_NOISE_STATS if nutpie_shaped else _PYMC_NOISE_STATS:
        stats[name] = (("chain", "draw"), rng.standard_normal((n_chains, n_draws)))
    groups = {"sample_stats": xr.Dataset(stats)}
    if nutpie_shaped:
        # Warmup divergences, saturations and energy belong to adaptation and
        # must not be counted.
        groups["warmup_sample_stats"] = xr.Dataset({
            "diverging": (("chain", "draw"), np.ones((n_chains, n_draws), dtype=bool)),
            "maxdepth_reached": (("chain", "draw"), np.ones((n_chains, n_draws), dtype=bool)),
            "energy": (("chain", "draw"), np.zeros((n_chains, n_draws))),
        })
    return groups


@pytest.fixture
def make_var_posterior():
    """Factory building synthetic VAR posteriors with controlled pathologies.

    Returns a callable accepting:

    * `n_chains`, `n_draws`, `n_vars`, `n_lags` — posterior shape.
    * `bad_coord` — `(row, col)` index into `B`; that coordinate gets a
      chain-dependent offset, so R-hat blows up while every other
      coordinate stays healthy.
    * `explosive_frac` — fraction of draws per chain whose lag block is
      `diag(1.2, 0.5, ...)` instead of `diag(0.5, ...)`. Switches `B` to a
      noiseless two-point mixture so the spectral-radius statistics are
      exact, with the explosive draws scattered within each chain so the
      indicator is neither autocorrelated nor chain-dependent.
    * `divergences` — divergent-transition count, placed at fixed positions.
      `None` omits the `sample_stats` group entirely.
    * `treedepth_hits` — number of transitions flagged as having saturated
      the maximum tree depth, under whichever backend name is in force.
    * `energy_rho` — AR(1) coefficient of the energy trace. The default 0.3
      is healthy; values near 1 drive E-BFMI below its warning threshold.
    * `extra_vars` — mapping of posterior variable name to trailing shape.
    * `coords` — attach `var`/`coeff` coords (as `VAR.fit` does) or leave
      the dims bare (as `ConjugateVAR` does).
    * `nutpie_shaped` — emit nutpie's sampler-stat names plus a
      `warmup_sample_stats` group full of divergences and saturations that
      must be ignored.
    """

    def _make(
        *,
        n_chains=4,
        n_draws=200,
        n_vars=2,
        n_lags=1,
        bad_coord=None,
        explosive_frac=0.0,
        divergences=0,
        treedepth_hits=0,
        energy_rho=0.3,
        extra_vars=None,
        coords=True,
        nutpie_shaped=False,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        n_coeff = n_vars * n_lags
        var_names = [f"y{i + 1}" for i in range(n_vars)]
        coeff_names = [f"L{lag}.{name}" for lag in range(1, n_lags + 1) for name in var_names]

        B = _coefficient_draws(rng, (n_chains, n_draws, n_vars, n_coeff), explosive_frac, bad_coord)
        sigma_sd = 0.5 + 0.05 * np.abs(rng.standard_normal((n_chains, n_draws, n_vars)))
        L = _cholesky_draws(rng, sigma_sd, n_chains, n_draws, n_vars)

        data_vars = {
            "B": (("chain", "draw", "var", "coeff"), B),
            "intercept": (("chain", "draw", "var"), 0.01 * rng.standard_normal((n_chains, n_draws, n_vars))),
            "sigma_sd": (("chain", "draw", "var"), sigma_sd),
            "L": (("chain", "draw", "var1", "var2"), L),
        }
        for name, shape in (extra_vars or {}).items():
            dims = tuple(f"{name}_dim_{k}" for k in range(len(shape)))
            data_vars[name] = (("chain", "draw", *dims), rng.standard_normal((n_chains, n_draws, *shape)))

        posterior_coords = (
            {"var": var_names, "coeff": coeff_names, "var1": var_names, "var2": var_names} if coords else None
        )
        groups = {"posterior": xr.Dataset(data_vars, coords=posterior_coords)}

        if divergences is not None:
            groups |= _sampler_stats(rng, n_chains, n_draws, divergences, treedepth_hits, energy_rho, nutpie_shaped)

        return az.InferenceData(**groups)

    return _make


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
