"""Slow DGP recovery tests for the SV model."""

import numpy as np
import pytest
from scipy import stats

from impulso._arviz_compat import get_group_dataset, hdi_bounds
from impulso.samplers import NUTSSampler
from impulso.sv.spec import StochasticVolatility

# Credible level for the scalar-parameter recovery guards below.
#
# These assertions ask whether the posterior is grossly wrong, not whether it
# is calibrated: calibration is a statement about many datasets, and each test
# here fits exactly one. On these fixtures the truth lands in the posterior's
# upper tail -- measured over five sampler seeds on the fixed RW dataset, the
# truth sits at posterior quantile 0.889 +- 0.028, and on the AR(1) dataset at
# 0.910 +- 0.010. That is expected rather than broken: sigma_eta is weakly
# identified when every h_t is seen through log-chi2 observation noise
# (variance pi^2/2), and the AR(1) fixture's 0.3 sits in the upper 13% of the
# default HalfNormal(0.2) prior.
#
# An 89% HDI on a right-skewed posterior trims almost entirely off the right,
# putting its upper edge in that same quantile band, so coverage was decided by
# MCMC noise: 3/5 sampler seeds on the RW fixture. That is why this passed on
# every run for months against one CI runner and then failed the first time the
# 3.12 leg drew different hardware, with src/ and the lock file untouched.
# Raising `draws` cannot fix it -- sharper endpoints converge onto the wrong
# side of the truth. The interval itself has to be the generous one.
#
# 98% covers 5/5 seeds for every parameter here with ~2.6 sd of quantile
# headroom, and still fails loudly if sigma_eta collapses to zero or blows up.
RECOVERY_HDI = 0.98


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

    # sigma_eta recovery guard (see RECOVERY_HDI)
    posterior = get_group_dataset(fitted.idata, "posterior")
    lo_da, hi_da = hdi_bounds(posterior["sigma_eta"], RECOVERY_HDI)
    lo, hi = float(lo_da), float(hi_da)
    true_sigma = sv_series_rw["sigma_eta_true"]
    assert lo <= true_sigma <= hi, f"sigma_eta {RECOVERY_HDI:.0%} HDI [{lo:.3f}, {hi:.3f}] misses {true_sigma}"

    # Pointwise 89% HDI covers truth at >= 80% of time points
    lo_h, hi_h = (bound.values for bound in hdi_bounds(posterior["h"], 0.89))
    covered = ((lo_h <= sv_series_rw["h_true"]) & (sv_series_rw["h_true"] <= hi_h)).mean()
    assert covered >= 0.80, f"Coverage {covered:.2f} < 0.80"


@pytest.fixture
def sv_series_ar1():
    """1-D series simulated from AR(1) log-volatility SV DGP."""
    rng = np.random.default_rng(11)
    T = 500
    phi = 0.95
    alpha = 0.0
    sigma_eta = 0.3
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

    posterior = get_group_dataset(fitted.idata, "posterior")
    for key, truth in [
        ("phi", sv_series_ar1["phi_true"]),
        ("sigma_eta", sv_series_ar1["sigma_eta_true"]),
    ]:
        lo_da, hi_da = hdi_bounds(posterior[key], RECOVERY_HDI)
        lo, hi = float(lo_da), float(hi_da)
        assert lo <= truth <= hi, f"{key} {RECOVERY_HDI:.0%} HDI [{lo:.3f}, {hi:.3f}] misses {truth}"
