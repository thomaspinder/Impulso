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

    # Positive definite Sigma via A @ A.T + I
    sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars)) * 0.5
            sigma[c, d] = A @ A.T + np.eye(n_vars)

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(
            sigma,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        ),
    })
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def synthetic_identified_idata_2v(synthetic_idata_2v):
    """Synthetic InferenceData with structural_shock_matrix added.

    Applies Cholesky decomposition to each Sigma draw.
    """
    sigma = synthetic_idata_2v.posterior["Sigma"].values
    n_chains, n_draws = sigma.shape[:2]

    P = np.zeros_like(sigma)
    for c in range(n_chains):
        for d in range(n_draws):
            P[c, d] = np.linalg.cholesky(sigma[c, d])

    P_da = xr.DataArray(
        P,
        dims=["chain", "draw", "response", "shock"],
        coords={"response": ["y1", "y2"], "shock": ["y1", "y2"]},
    )
    new_posterior = synthetic_idata_2v.posterior.assign(structural_shock_matrix=P_da)
    return az.InferenceData(posterior=new_posterior)
