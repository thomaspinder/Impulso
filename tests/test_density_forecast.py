"""Tests for density forecasts (issue #92).

Pin-first: mean-mode regression pins captured against current code BEFORE
any behaviour change. Then density-mode tests verify the new behaviour.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso._arviz_compat import get_group_dataset, make_idata
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
    idata = make_idata(posterior=posterior)
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


@pytest.fixture
def fitted_constant_t(fitted_constant):
    """`fitted_constant` re-labelled with Student-t errors, nu = 5.

    Shares the *identical* posterior (plus a constant `nu`), so any
    difference against `fitted_constant` is attributable to the error law
    alone.
    """
    from impulso.observation import StudentT

    posterior = get_group_dataset(fitted_constant.idata, "posterior").copy()
    n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    posterior["nu"] = xr.DataArray(np.full((n_chains, n_draws), 5.0), dims=["chain", "draw"])
    return FittedVAR(
        idata=make_idata(posterior=posterior),
        n_lags=fitted_constant.n_lags,
        data=fitted_constant.data,
        var_names=fitted_constant.var_names,
        volatility=Constant(),
        error_dist=StudentT(nu=5.0),
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


class TestDensityModeRegressionPin:
    """Regression pins: Gaussian *density*-mode values captured BEFORE any change.

    Sentinel for two things at once: that Gaussian models retain their current
    numbers, and that the RNG stream order inside `forecast` is unchanged. Any
    reordering of generator consumption (e.g. drawing a mixing variable before
    the standard normals) shifts every value here.
    """

    def test_gaussian_density_median_pinned(self, fitted_constant):
        result = fitted_constant.forecast(steps=5, seed=42)
        med = result.median()
        expected = np.array([
            [0.227643, -0.153848],
            [0.225367, -0.112458],
            [0.064284, -0.138994],
            [0.136689, -0.202966],
            [0.180533, -0.039111],
        ])
        np.testing.assert_allclose(med.values, expected, rtol=1e-4)

    def test_gaussian_density_first_draw_pinned(self, fitted_constant):
        """Bit-level sentinel on the first consumed innovation."""
        result = fitted_constant.forecast(steps=5, seed=42)
        draws = result.idata.posterior_predictive["forecast"].values
        assert draws.shape == (2, 50, 5, 2)
        np.testing.assert_allclose(draws[0, 0, 0, :], [0.38093296, -0.79446185], rtol=1e-7)


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


class TestStudentTDensityForecast:
    """Density forecasting under Student-t observation errors (issue #152)."""

    def test_shape_and_finiteness(self, fitted_constant_t):
        result = fitted_constant_t.forecast(steps=5, seed=42)
        draws = result.idata.posterior_predictive["forecast"].values
        assert draws.shape == (2, 50, 5, 2)
        assert np.isfinite(draws).all()
        assert result.mode == "density"

    def test_reproducible_under_a_matched_seed(self, fitted_constant_t):
        r1 = fitted_constant_t.forecast(steps=5, seed=42)
        r2 = fitted_constant_t.forecast(steps=5, seed=42)
        np.testing.assert_array_equal(
            r1.idata.posterior_predictive["forecast"].values,
            r2.idata.posterior_predictive["forecast"].values,
        )

    def test_differs_across_seeds(self, fitted_constant_t):
        r1 = fitted_constant_t.forecast(steps=5, seed=42)
        r2 = fitted_constant_t.forecast(steps=5, seed=99)
        assert not np.array_equal(
            r1.idata.posterior_predictive["forecast"].values,
            r2.idata.posterior_predictive["forecast"].values,
        )

    def test_tails_are_fatter_but_the_bulk_is_not(self, fitted_constant, fitted_constant_t):
        """Matched posterior + seed: t widens the *tails*, not the whole band.

        Checking only "wider" would pass for any change that inflates the
        scale. The IQR ratio guards against that false positive: a heavier
        tail leaves the interquartile range essentially alone.
        """
        g = fitted_constant.forecast(steps=5, seed=7).idata.posterior_predictive["forecast"].values
        t = fitted_constant_t.forecast(steps=5, seed=7).idata.posterior_predictive["forecast"].values

        g0, t0 = g[:, :, 0, :].ravel(), t[:, :, 0, :].ravel()
        g_tail = np.quantile(g0, 0.999) - np.quantile(g0, 0.001)
        t_tail = np.quantile(t0, 0.999) - np.quantile(t0, 0.001)
        assert t_tail > g_tail

        g_iqr = np.quantile(g0, 0.75) - np.quantile(g0, 0.25)
        t_iqr = np.quantile(t0, 0.75) - np.quantile(t0, 0.25)
        assert 0.8 < t_iqr / g_iqr < 1.3

    def test_mean_mode_is_identical_across_error_distributions(self, fitted_constant, fitted_constant_t):
        """Mean mode consumes no randomness, so the error law cannot matter."""
        g = fitted_constant.forecast(steps=5, include_shock_uncertainty=False)
        t = fitted_constant_t.forecast(steps=5, include_shock_uncertainty=False)
        np.testing.assert_array_equal(
            g.idata.posterior_predictive["forecast"].values,
            t.idata.posterior_predictive["forecast"].values,
        )

    def test_missing_nu_in_posterior_is_a_clear_error(self, fitted_constant):
        """A t-labelled FittedVAR over a Gaussian posterior fails loudly."""
        from impulso.observation import StudentT

        mislabelled = FittedVAR(
            idata=fitted_constant.idata,
            n_lags=fitted_constant.n_lags,
            data=fitted_constant.data,
            var_names=fitted_constant.var_names,
            volatility=Constant(),
            error_dist=StudentT(nu=5.0),
        )
        with pytest.raises(ValueError, match="no 'nu' variable"):
            mislabelled.forecast(steps=3, seed=1)
