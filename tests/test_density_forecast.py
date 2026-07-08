"""Tests for density forecasts (issue #92).

Pin-first: mean-mode regression pins captured against current code BEFORE
any behaviour change. Then density-mode tests verify the new behaviour.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.volatility import Constant


@pytest.fixture
def fitted_constant():
    """FittedVAR with Constant volatility, deterministic fixture."""
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_vars = 2, 50, 2
    B_raw = rng.standard_normal((n_chains, n_draws, n_vars, n_vars)) * 0.1
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
    posterior = xr.Dataset({
        "B": (("chain", "draw", "var1", "var2"), B_raw),
        "intercept": (("chain", "draw", "var1"), intercept),
        "L": (("chain", "draw", "var1", "var2"), L_raw),
    })
    idata = az.InferenceData(posterior=posterior)
    A1 = np.array([[0.5, 0.1], [-0.2, 0.3]])
    y = np.zeros((200, 2))
    y[0] = np.array([0.1, -0.05]) / 0.7
    for t in range(1, 200):
        y[t] = np.array([0.1, -0.05]) + A1 @ y[t - 1] + rng.standard_normal(2) * 0.5
    index = pd.date_range("2000-01-01", periods=200, freq="QS")
    data = VARData(endog=y, endog_names=["y1", "y2"], index=index)
    return FittedVAR(
        idata=idata,
        n_lags=1,
        data=data,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


class TestMeanModeRegressionPin:
    """Regression pins: mean-mode forecast values captured BEFORE any change.

    These MUST pass against the current code and continue to pass after
    the refactor when called with include_shock_uncertainty=False.
    """

    def test_mean_mode_median_pinned(self, fitted_constant):
        result = fitted_constant.forecast(steps=5, include_shock_uncertainty=False)
        med = result.median()
        expected = np.array([
            [0.236119, -0.062535],
            [0.204321, -0.119617],
            [0.189768, -0.134171],
            [0.181691, -0.129007],
            [0.174914, -0.118809],
        ])
        np.testing.assert_allclose(med.values, expected, rtol=1e-4)

    def test_mean_mode_hdi_pinned(self, fitted_constant):
        result = fitted_constant.forecast(steps=5, include_shock_uncertainty=False)
        hdi = result.hdi(0.89)
        expected_lower = np.array([
            [0.040915, -0.216668],
            [-0.085083, -0.308106],
            [-0.161075, -0.340470],
            [-0.222975, -0.357058],
            [-0.229783, -0.412952],
        ])
        expected_upper = np.array([
            [0.373229, 0.115438],
            [0.415315, 0.080508],
            [0.430260, 0.088558],
            [0.414080, 0.094502],
            [0.435951, 0.055401],
        ])
        np.testing.assert_allclose(hdi.lower.values, expected_lower, rtol=1e-4)
        np.testing.assert_allclose(hdi.upper.values, expected_upper, rtol=1e-4)


class TestDensityMode:
    """Density mode (include_shock_uncertainty=True, default) draws shocks."""

    def test_density_is_default(self, fitted_constant):
        result = fitted_constant.forecast(steps=5)
        assert result.mode == "density"

    def test_mean_mode_flag(self, fitted_constant):
        result = fitted_constant.forecast(steps=5, include_shock_uncertainty=False)
        assert result.mode == "mean"

    def test_density_hdi_wider_than_mean(self, fitted_constant):
        result_mean = fitted_constant.forecast(steps=5, include_shock_uncertainty=False)
        result_density = fitted_constant.forecast(steps=5, include_shock_uncertainty=True, seed=42)

        hdi_mean = result_mean.hdi(0.89)
        hdi_density = result_density.hdi(0.89)

        width_mean = hdi_mean.upper.values - hdi_mean.lower.values
        width_density = hdi_density.upper.values - hdi_density.lower.values
        assert np.all(width_density > width_mean)

    def test_density_widens_with_horizon(self, fitted_constant):
        result = fitted_constant.forecast(steps=10, include_shock_uncertainty=True, seed=42)
        hdi = result.hdi(0.89)
        width = hdi.upper.values - hdi.lower.values
        assert np.any(width[9] > width[0])


class TestSeedReproducibility:
    """Seed parameter for density forecasts."""

    def test_same_seed_same_result(self, fitted_constant):
        r1 = fitted_constant.forecast(steps=5, seed=42)
        r2 = fitted_constant.forecast(steps=5, seed=42)
        np.testing.assert_array_equal(
            r1.idata.posterior_predictive["forecast"].values,
            r2.idata.posterior_predictive["forecast"].values,
        )

    def test_different_seed_different_result(self, fitted_constant):
        r1 = fitted_constant.forecast(steps=5, seed=42)
        r2 = fitted_constant.forecast(steps=5, seed=99)
        assert not np.array_equal(
            r1.idata.posterior_predictive["forecast"].values,
            r2.idata.posterior_predictive["forecast"].values,
        )

    def test_seed_accepts_generator(self, fitted_constant):
        rng = np.random.default_rng(42)
        result = fitted_constant.forecast(steps=5, seed=rng)
        assert result.median().shape == (5, 2)
