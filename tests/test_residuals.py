"""Tests for reduced-form residual reconstruction (`_residuals.py`)."""

import numpy as np
import pandas as pd
import pytest

from impulso._residuals import reduced_form_residuals
from impulso.data import VARData


@pytest.fixture
def var_data_2v_short():
    """Small 2-var VARData matching the synthetic_idata_2v posterior shapes."""
    rng = np.random.default_rng(7)
    T, n = 30, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


class TestReducedFormResiduals:
    def test_shape(self, synthetic_idata_2v, var_data_2v_short):
        resid = reduced_form_residuals(synthetic_idata_2v.posterior, var_data_2v_short, n_lags=1)
        assert resid.shape == (2, 50, 29, 2)  # (C, D, T - p, n)

    def test_golden_equality_with_inline_computation(self, synthetic_idata_2v, var_data_2v_short):
        """Helper reproduces the inline computation it replaced in
        historical_decomposition, exactly."""
        posterior = synthetic_idata_2v.posterior
        n_lags = 1
        B_draws = posterior["B"].values
        intercept_draws = posterior["intercept"].values
        y = var_data_2v_short.endog
        T = y.shape[0]
        x_lag = np.concatenate(
            [y[n_lags - lag : T - lag] for lag in range(1, n_lags + 1)],
            axis=1,
        )
        y_hat = intercept_draws[:, :, np.newaxis, :] + np.einsum("cdij,tj->cdti", B_draws, x_lag)
        expected = y[n_lags:][np.newaxis, np.newaxis, :, :] - y_hat

        actual = reduced_form_residuals(posterior, var_data_2v_short, n_lags)
        np.testing.assert_array_equal(actual, expected)

    def test_zero_residuals_when_posterior_matches_dgp(self):
        """Data generated exactly from (intercept, B) gives zero residuals."""
        import arviz as az
        import xarray as xr

        rng = np.random.default_rng(0)
        T, n = 40, 2
        B_true = np.array([[0.5, 0.1], [0.0, 0.4]])
        c_true = np.array([0.2, -0.1])
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = c_true + B_true @ y[t - 1]
        y[0] = rng.standard_normal(n)
        # Re-simulate with the random start propagated
        for t in range(1, T):
            y[t] = c_true + B_true @ y[t - 1]

        data = VARData(
            endog=y,
            endog_names=["a", "b"],
            index=pd.date_range("2000-01-01", periods=T, freq="MS"),
        )
        posterior = xr.Dataset({
            "B": xr.DataArray(
                np.broadcast_to(B_true, (1, 3, n, n)).copy(),
                dims=["chain", "draw", "var", "coeff"],
            ),
            "intercept": xr.DataArray(
                np.broadcast_to(c_true, (1, 3, n)).copy(),
                dims=["chain", "draw", "var"],
            ),
        })
        idata = az.InferenceData(posterior=posterior)
        resid = reduced_form_residuals(idata.posterior, data, n_lags=1)
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)
