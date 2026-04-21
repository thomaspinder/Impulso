"""Slow DGP recovery tests for the SV model."""

import numpy as np
import pytest
from scipy import stats

from impulso.samplers import NUTSSampler
from impulso.sv.spec import StochasticVolatility


@pytest.mark.slow
def test_rw_recovery(sv_data_rw, sv_series_rw):
    """Posterior should recover the true log-vol path and sigma_eta."""
    sampler = NUTSSampler(
        draws=500,
        tune=500,
        chains=2,
        cores=1,
        target_accept=0.95,
        random_seed=7,
    )
    fitted = StochasticVolatility(dynamics="random_walk").fit(sv_data_rw, sampler=sampler)

    # Rank correlation of posterior-median h_t with truth
    h_post = fitted.log_volatility.reshape(-1, fitted.log_volatility.shape[-1])
    h_med = np.median(h_post, axis=0)
    rho, _ = stats.spearmanr(h_med, sv_series_rw["h_true"])
    assert rho > 0.7, f"Spearman correlation {rho:.3f} < 0.7"

    # sigma_eta 89% HDI covers truth
    import arviz as az

    hdi = az.hdi(fitted.idata, hdi_prob=0.89)
    lo = float(hdi["sigma_eta"].sel(hdi="lower"))
    hi = float(hdi["sigma_eta"].sel(hdi="higher"))
    true_sigma = sv_series_rw["sigma_eta_true"]
    assert lo <= true_sigma <= hi, f"sigma_eta HDI [{lo:.3f}, {hi:.3f}] misses {true_sigma}"

    # Pointwise 89% HDI covers truth at >= 80% of time points
    h_hdi = az.hdi(fitted.idata, hdi_prob=0.89)["h"].values  # (T, 2)
    lo_h, hi_h = h_hdi[:, 0], h_hdi[:, 1]
    covered = ((lo_h <= sv_series_rw["h_true"]) & (sv_series_rw["h_true"] <= hi_h)).mean()
    assert covered >= 0.80, f"Coverage {covered:.2f} < 0.80"


@pytest.fixture
def sv_series_ar1():
    """1-D series simulated from AR(1) log-volatility SV DGP."""
    rng = np.random.default_rng(11)
    T = 500
    phi = 0.95
    alpha = 0.0
    sigma_eta = 0.1
    mu = 0.0
    h = np.zeros(T)
    for t in range(1, T):
        h[t] = alpha + phi * (h[t - 1] - alpha) + sigma_eta * rng.standard_normal()
    y = mu + np.exp(0.5 * h) * rng.standard_normal(T)
    return {"y": y, "h_true": h, "phi_true": phi, "sigma_eta_true": sigma_eta}


@pytest.mark.slow
def test_ar1_recovery(sv_series_ar1):
    import pandas as pd

    from impulso.sv.data import SVData

    y = sv_series_ar1["y"]
    index = pd.date_range("1980-01-01", periods=len(y), freq="MS")
    data = SVData(y=y, name="sim", index=index)
    sampler = NUTSSampler(
        draws=500,
        tune=500,
        chains=2,
        cores=1,
        target_accept=0.95,
        random_seed=8,
    )
    fitted = StochasticVolatility(dynamics="ar1").fit(data, sampler=sampler)

    h_post = fitted.log_volatility.reshape(-1, fitted.log_volatility.shape[-1])
    h_med = np.median(h_post, axis=0)
    rho, _ = stats.spearmanr(h_med, sv_series_ar1["h_true"])
    assert rho > 0.7, f"Spearman correlation {rho:.3f} < 0.7"

    import arviz as az

    hdi = az.hdi(fitted.idata, hdi_prob=0.89)
    for key, truth in [
        ("phi", sv_series_ar1["phi_true"]),
        ("sigma_eta", sv_series_ar1["sigma_eta_true"]),
    ]:
        lo = float(hdi[key].sel(hdi="lower"))
        hi = float(hdi[key].sel(hdi="higher"))
        assert lo <= truth <= hi, f"{key} HDI [{lo:.3f}, {hi:.3f}] misses {truth}"
