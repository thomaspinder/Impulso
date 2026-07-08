"""Regression pins for Cholesky IRF/FEVD/HD outputs.

These tests capture the exact numerical output of impulse_response, fevd,
and historical_decomposition under Cholesky identification via the public
pipeline (FittedVAR -> set_identification_strategy). They MUST pass before
AND after the #91 shock_matrix refactor to prove the refactor is
behaviour-preserving.

Fast tests use a self-consistent synthetic InferenceData fixture with
n_lags=1 and B shape (2, 50, 2, 2).
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.volatility import Constant


@pytest.fixture
def fitted_and_identified():
    """Self-consistent FittedVAR + IdentifiedVAR through the public flow.

    B has shape (C, D, n_vars, n_vars*n_lags) = (2, 50, 2, 2) matching
    n_lags=1.  Deterministic rng seed ensures reproducible outputs.
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_vars, n_lags = 2, 50, 2, 1

    B_raw = rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.1
    B_raw[:, :, 0, 0] += 0.5
    B_raw[:, :, 1, 1] += 0.3
    B_raw[:, :, 0, 1] += 0.1
    B_raw[:, :, 1, 0] -= 0.2
    intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.1
    intercept[:, :, 0] += 0.1
    intercept[:, :, 1] -= 0.05
    sigma_raw = rng.standard_normal((n_chains, n_draws, n_vars, n_vars)) * 0.3
    sigma_raw[:, :, 0, 0] += 1.5
    sigma_raw[:, :, 1, 1] += 1.2
    L_raw = np.tril(sigma_raw)
    L_raw[:, :, 0, 0] = np.abs(L_raw[:, :, 0, 0])
    L_raw[:, :, 1, 1] = np.abs(L_raw[:, :, 1, 1])
    sd = np.abs(rng.standard_normal((n_chains, n_draws, n_vars))) * 0.5 + 0.5
    L_raw[:, :, range(n_vars), range(n_vars)] = sd
    Sigma = np.einsum("cdij,cdkj->cdik", L_raw, L_raw)

    posterior = xr.Dataset({
        "B": (("chain", "draw", "var1", "var2"), B_raw),
        "intercept": (("chain", "draw", "var1"), intercept),
        "L": (("chain", "draw", "var1", "var2"), L_raw),
        "Sigma": (("chain", "draw", "var1", "var2"), Sigma),
    })
    idata = az.InferenceData(posterior=posterior)

    A1 = np.array([[0.5, 0.1], [-0.2, 0.3]])
    intercept_true = np.array([0.1, -0.05])
    y = np.zeros((200, 2))
    y[0] = intercept_true / 0.7
    for t in range(1, 200):
        y[t] = intercept_true + A1 @ y[t - 1] + rng.standard_normal(2) * 0.5
    index = pd.date_range("2000-01-01", periods=200, freq="QS")
    data = VARData(endog=y, endog_names=["y1", "y2"], index=index)

    fitted = FittedVAR(
        idata=idata,
        n_lags=n_lags,
        data=data,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
    return fitted, identified


class TestCholeskyIRFPin:
    """Regression pin: IRF output under Cholesky identification."""

    def test_irf_median_shape(self, fitted_and_identified):
        _, identified = fitted_and_identified
        irf = identified.impulse_response(horizon=5)
        assert irf.median().shape == (6, 4)

    def test_irf_horizon_0_is_lower_triangular(self, fitted_and_identified):
        """IRF at h=0 = I @ P = P; lower-triangular under Cholesky."""
        _, identified = fitted_and_identified
        irf = identified.impulse_response(horizon=5)
        med = irf.median()
        assert med.values[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert med.values[0, 0] > 0
        assert med.values[0, 3] > 0

    def test_irf_values_pinned(self, fitted_and_identified):
        """Pin exact median IRF values for regression detection."""
        _, identified = fitted_and_identified
        irf = identified.impulse_response(horizon=5)
        med = irf.median()
        expected = np.array([
            [0.831534, 0.0, 0.006446, 0.793896],
            [0.418751, 0.073324, -0.162831, 0.244156],
            [0.182067, 0.056493, -0.120533, 0.065252],
            [0.075273, 0.035107, -0.067667, 0.009670],
            [0.028684, 0.016822, -0.031755, -0.001846],
            [0.011834, 0.006212, -0.012512, -0.002320],
        ])
        np.testing.assert_allclose(med.values, expected, rtol=1e-3, atol=1e-5)


class TestCholeskyFEVDPin:
    """Regression pin: FEVD output under Cholesky identification."""

    def test_fevd_sums_to_one(self, fitted_and_identified):
        _, identified = fitted_and_identified
        fevd = identified.fevd(horizon=5)
        fevd_da = fevd.idata.posterior_predictive["fevd"]
        med = fevd_da.median(dim=("chain", "draw"))
        for resp in ["y1", "y2"]:
            sums = med.sel(response=resp).values.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_fevd_h0_cholesky_ordering(self, fitted_and_identified):
        """At h=0 with Cholesky [y1,y2], y1 is 100% own shock."""
        _, identified = fitted_and_identified
        fevd = identified.fevd(horizon=5)
        fevd_da = fevd.idata.posterior_predictive["fevd"]
        med = fevd_da.median(dim=("chain", "draw"))
        y1_h0 = med.sel(response="y1", horizon=0).values
        np.testing.assert_allclose(y1_h0, [1.0, 0.0], atol=1e-10)

    def test_fevd_values_pinned(self, fitted_and_identified):
        """Pin exact median FEVD values for regression detection."""
        _, identified = fitted_and_identified
        fevd = identified.fevd(horizon=5)
        med = fevd.median()
        expected = np.array([
            [1.0, 0.0, 0.051933, 0.948067],
            [0.993285, 0.006715, 0.101073, 0.898927],
            [0.989732, 0.010268, 0.128241, 0.871759],
            [0.988027, 0.011973, 0.135176, 0.864824],
            [0.987507, 0.012493, 0.137366, 0.862634],
            [0.987340, 0.012660, 0.137887, 0.862113],
        ])
        np.testing.assert_allclose(med.values, expected, rtol=1e-4)


class TestCholeskyHDPin:
    """Regression pin: historical decomposition under Cholesky."""

    def test_hd_shape(self, fitted_and_identified):
        _, identified = fitted_and_identified
        hd = identified.historical_decomposition()
        hd_da = hd.idata.posterior_predictive["hd"]
        assert hd_da.shape == (2, 50, 199, 2, 2)

    def test_hd_shocks_match_scheme(self, fitted_and_identified):
        """HD shock coords must match the identification scheme."""
        _, identified = fitted_and_identified
        hd = identified.historical_decomposition()
        hd_da = hd.idata.posterior_predictive["hd"]
        assert list(hd_da.coords["shock"].values) == ["y1", "y2"]

    def test_hd_reconstructs_nontrivially(self, fitted_and_identified):
        """Sum across shocks should be non-trivial."""
        _, identified = fitted_and_identified
        hd = identified.historical_decomposition()
        hd_da = hd.idata.posterior_predictive["hd"]
        hd_sum = hd_da.sum(dim="shock").median(dim=("chain", "draw"))
        assert not np.allclose(hd_sum.values, 0.0)

    def test_hd_sum_values_pinned(self, fitted_and_identified):
        """Pin HD residual reconstruction at first 3 time points."""
        _, identified = fitted_and_identified
        hd = identified.historical_decomposition()
        hd_da = hd.idata.posterior_predictive["hd"]
        hd_sum = hd_da.sum(dim="shock").median(dim=("chain", "draw"))
        expected_first3 = np.array([
            [-0.653623, -0.240780],
            [0.218211, -0.050645],
            [-0.318535, -0.322762],
        ])
        np.testing.assert_allclose(hd_sum.values[:3], expected_first3, rtol=1e-4)
