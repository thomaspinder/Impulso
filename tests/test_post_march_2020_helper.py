"""Tests for the post-March-2020 conditional-forecast helper.

The helper lives at ``docs/tutorials/_post_march_2020.py`` (underscored so the
docs build skips it), so it is loaded by file path rather than imported as a
package module.
"""

import importlib.util
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from impulso.fitted import FittedVAR
from impulso.volatility import Constant

_HELPER = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "_post_march_2020.py"
_spec = importlib.util.spec_from_file_location("_post_march_2020", _HELPER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
conditional_forecast = _mod.conditional_forecast


def _synthetic_fitted(n_chains, n_draws, n_vars, n_lags, seed=0):
    """Tiny FittedVAR with a fabricated NIW-style posterior (B, intercept, L).

    Mirrors conftest's ``synthetic_idata_2v`` group layout and attaches the real
    ``Constant`` volatility adapter, which reads ``posterior["L"]``.
    """
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.2
    intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.1
    l_chol = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            a = rng.standard_normal((n_vars, n_vars)) * 0.5
            l_chol[c, d] = np.linalg.cholesky(a @ a.T + np.eye(n_vars))
    posterior = xr.Dataset({
        "B": xr.DataArray(b, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "L": xr.DataArray(l_chol, dims=["chain", "draw", "var1", "var2"]),
    })
    idata = az.InferenceData(posterior=posterior)
    return FittedVAR.model_construct(idata=idata, n_lags=n_lags, volatility=Constant())


def test_conditional_forecast_reproduces_path_var1():
    """VAR(1), 2 vars: conditioned row hits the imposed path; shape + finite."""
    n_chains, n_draws, n_vars, n_lags = 2, 5, 2, 1
    horizon, index = 6, 0
    fitted = _synthetic_fitted(n_chains, n_draws, n_vars, n_lags)
    history = np.random.default_rng(1).standard_normal((n_lags, n_vars))
    unemployment_path = np.linspace(1.0, 3.0, horizon)

    out = conditional_forecast(fitted, history, unemployment_path, index=index, seed=7)

    assert out.shape == (n_chains, n_draws, horizon, n_vars)
    assert np.all(np.isfinite(out))
    imposed = np.broadcast_to(unemployment_path, (n_chains, n_draws, horizon))
    np.testing.assert_allclose(out[:, :, :, index], imposed, atol=1e-8)


def test_conditional_forecast_reproduces_path_var2_multivar():
    """VAR(2), 3 vars, non-zero index: exercises the multi-lag MA recursion."""
    n_chains, n_draws, n_vars, n_lags = 1, 4, 3, 2
    horizon, index = 5, 1
    fitted = _synthetic_fitted(n_chains, n_draws, n_vars, n_lags, seed=3)
    # More history rows than n_lags: only the last n_lags must be used.
    history = np.random.default_rng(2).standard_normal((10, n_vars))
    unemployment_path = np.array([2.0, 1.5, 1.0, 0.8, 0.6])

    out = conditional_forecast(fitted, history, unemployment_path, index=index, seed=11)

    assert out.shape == (n_chains, n_draws, horizon, n_vars)
    assert np.all(np.isfinite(out))
    imposed = np.broadcast_to(unemployment_path, (n_chains, n_draws, horizon))
    np.testing.assert_allclose(out[:, :, :, index], imposed, atol=1e-8)
