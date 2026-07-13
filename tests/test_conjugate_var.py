"""End-to-end tests for the conjugate VAR estimator (ConjugateVAR)."""

import numpy as np
import pandas as pd
import pytest

from impulso.conjugate import ConjugateVAR
from impulso.conjugate_volatility import PandemicBreak
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.priors import MinnesotaPrior, NIWPrior
from impulso.volatility import Constant

DRAWS = 40  # small, for speed


def _synthetic_var_data(
    n_obs: int,
    seed: int,
    spike_start_level: int | None = None,
    spike_scale: float = 8.0,
) -> VARData:
    """Stationary VAR(1) DGP with 2 variables, optional 3-period residual spike.

    ``spike_start_level`` is the (0-based) level index at which a burst of three
    inflated residuals begins; ``None`` gives the homoscedastic baseline.
    """
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    l_base = np.array([[0.5, 0.0], [0.2, 0.4]])
    y = np.zeros((n_obs, 2))
    for t in range(1, n_obs):
        eps = l_base @ rng.standard_normal(2)
        if spike_start_level is not None and spike_start_level <= t <= spike_start_level + 2:
            eps = eps * spike_scale
        y[t] = A @ y[t - 1] + eps
    index = pd.date_range("2010-01-01", periods=n_obs, freq="MS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


def test_no_break_fit_forecast_and_irf():
    data = _synthetic_var_data(60, seed=0)
    model = ConjugateVAR(lags=1, prior=NIWPrior(), draws=DRAWS, tune=DRAWS, seed=0)

    fitted = model.fit(data)
    assert isinstance(fitted, FittedVAR)
    assert isinstance(fitted.volatility, Constant)

    post = fitted.idata.posterior
    assert post["B"].shape == (1, DRAWS, 2, 2)
    assert post["intercept"].shape == (1, DRAWS, 2)
    assert post["L"].shape == (1, DRAWS, 2, 2)

    forecast = fitted.forecast(steps=6, seed=0).idata.posterior_predictive["forecast"].values
    assert forecast.shape == (1, DRAWS, 6, 2)
    assert np.isfinite(forecast).all()

    identified = fitted.set_identification_strategy(Cholesky(ordering=data.endog_names))
    irf = identified.impulse_response(horizon=8).idata.posterior_predictive["irf"].values
    assert irf.shape == (1, DRAWS, 9, 2, 2)
    assert np.isfinite(irf).all()


def test_pandemic_break_fit_forecast_and_irf():
    start = 55  # lag-trimmed index of t*; level residual spike begins at start + n_lags
    data = _synthetic_var_data(70, seed=1, spike_start_level=start + 1)
    model = ConjugateVAR(
        lags=1,
        prior=NIWPrior(),
        volatility=PandemicBreak(start=start),
        draws=DRAWS,
        tune=DRAWS,
        seed=1,
    )

    fitted = model.fit(data)
    assert isinstance(fitted, FittedVAR)
    assert isinstance(fitted.volatility, PandemicBreak)

    post = fitted.idata.posterior
    for key in ("s_march", "s_april", "s_may", "rho"):
        assert post[key].shape == (1, DRAWS)
        assert np.isfinite(post[key].values).all()

    forecast = fitted.forecast(steps=6, seed=1).idata.posterior_predictive["forecast"].values
    assert forecast.shape == (1, DRAWS, 6, 2)
    assert np.isfinite(forecast).all()

    identified = fitted.set_identification_strategy(Cholesky(ordering=data.endog_names))
    irf = identified.impulse_response(horizon=8, at=10).idata.posterior_predictive["irf"].values
    assert irf.shape == (1, DRAWS, 9, 2, 2)
    assert np.isfinite(irf).all()


def test_rejects_non_niw_prior():
    with pytest.raises(ValueError, match="VAR"):
        ConjugateVAR(lags=1, prior=MinnesotaPrior())


def test_rejects_pymc_volatility():
    with pytest.raises(ValueError, match="VAR"):
        ConjugateVAR(lags=1, prior=NIWPrior(), volatility=Constant())
