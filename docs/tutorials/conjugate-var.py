# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The conjugate VAR: fast Bayesian estimation
# ## When to reach for `ConjugateVAR` instead of the NUTS VAR
#
# Impulso estimates a reduced-form VAR two ways. Both return the *same* `FittedVAR`, so
# identification, impulse responses, FEVDs, and forecasts are byte-for-byte identical
# downstream. What differs is the **mode of inference**:
#
# - **The NUTS VAR** (`VAR`) places *independent-Normal* priors on the coefficients and
#   samples the full posterior with Hamiltonian Monte Carlo. Maximally flexible — it admits
#   per-equation shrinkage, stochastic volatility, sign restrictions, and external
#   instruments — but every coefficient is a sampled latent, so large systems are slow.
# - **The conjugate VAR** (`ConjugateVAR`) places a *Normal-Inverse-Wishart* prior, which is
#   conjugate to the VAR likelihood. The coefficient/covariance posterior is then available
#   in **closed form**: we draw $(\beta, \Sigma)$ analytically and reserve Monte Carlo for a
#   single low-dimensional hyperparameter — the Minnesota tightness $\lambda$ — which the
#   data *selects* by marginal likelihood ({cite:t}`giannoneLenzaPrimiceri2015`).
#
# This notebook fits both on the same series, shows they reach the same structural
# conclusions, times them, and ends with a rule for choosing between them. We use an
# **environmental** system — a German climate–energy VAR — rather than the usual
# macro data, to show the machinery is domain-agnostic.
#
# :::{admonition} Scope
# :class: note
# This is the estimator-first tour. For the conjugate VAR wearing a *deterministic
# volatility break* — the COVID application it was built for — see
# [Estimating a VAR after March 2020](post-march-2020.py). Why a conjugate estimator is a
# *sibling* of `VAR` rather than a mode of it is recorded in ADR 0004.
# :::

# %% tags=["remove-cell"]
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pytensor").setLevel(logging.ERROR)

# %% tags=["remove-cell"]
import os

# Smoke-render flag: IMPULSO_DOCS_CI=1 shrinks MCMC for fast CI builds.
ci = os.environ.get("IMPULSO_DOCS_CI") == "1"

# %%
import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from impulso import Cholesky, ConjugateVAR, MinnesotaPrior, NIWPrior, VAR, VARData
from impulso.samplers import NUTSSampler

# %% [markdown]
# ## Data: a German climate–energy system
#
# Weather is where the environment meets the economy: temperature drives heating and
# cooling demand, sunshine and wind set renewable supply, and rainfall feeds hydro and
# agriculture. We assemble a four-variable monthly system for Berlin (1980–2024) from the
# **ERA5 reanalysis** ({cite:t}`hersbach2020`), served by the
# [Open-Meteo](https://open-meteo.com/) historical archive.
#
# | Variable | ERA5 series (monthly mean of daily) | Unit | Economic reading |
# |----------|-------------------------------------|------|------------------|
# | `temperature` | `temperature_2m_mean` | °C | heating / cooling demand |
# | `radiation` | `shortwave_radiation_sum` | MJ/m² | solar-PV potential |
# | `wind` | `wind_speed_10m_mean` | km/h | wind-power potential |
# | `precipitation` | `precipitation_sum` | mm/day | hydro / runoff |
#
# The committed CSV is produced once by `scripts/fetch_berlin_climate.py` and read offline
# here — no network call at render time. That script targets Open-Meteo's free archive
# endpoint, so anyone can reproduce the file without credentials.

# %% mystnb={"figure": {"caption": "Raw monthly ERA5 series for Berlin, 1980–2024. Temperature and radiation are dominated by the seasonal cycle.", "name": "climate-raw"}} tags=["remove-input"]
raw = pd.read_csv("data/berlin_climate.csv", index_col="date", parse_dates=True)

fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
units = {"temperature": "°C", "radiation": "MJ/m²", "wind": "km/h", "precipitation": "mm/day"}
for ax, col in zip(axes, raw.columns, strict=True):
    ax.plot(raw.index, raw[col], linewidth=0.7, color="0.3")
    ax.set_ylabel(f"{col}\n({units[col]})", fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_title("Berlin climate — raw monthly means (1980–2024)")
fig.tight_layout()

# %% [markdown]
# The raw series are overwhelmingly *seasonal* — a VAR fit on them would spend its
# coefficients re-learning the calendar. We model **anomalies** instead: each observation
# minus its month-of-year climatological mean, standardised to unit variance. The result is
# stationary, comparable across variables, and lets impulse responses read in standard
# deviations.

# %%
climatology = raw.groupby(raw.index.month).transform("mean")
anomalies = raw - climatology
anomalies = (anomalies - anomalies.mean()) / anomalies.std()
anomalies.describe().round(2)

# %% mystnb={"figure": {"caption": "Standardised monthly anomalies — the seasonal cycle removed. This is what the VAR sees.", "name": "climate-anomalies"}} tags=["remove-input"]
fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
for ax, col in zip(axes, anomalies.columns, strict=True):
    ax.plot(anomalies.index, anomalies[col], linewidth=0.6, color="C0")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_ylabel(col, fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_title("Berlin climate — standardised anomalies")
fig.tight_layout()

# %% [markdown]
# ## Fitting the conjugate VAR
#
# We use twelve lags — enough to capture up to a year of dynamic feedback in monthly data —
# giving $4 \times 12 = 48$ coefficients per equation. The prior is the conjugate Minnesota
# prior `NIWPrior`; `select=True` asks the estimator to choose the overall tightness
# $\lambda$ by maximising the marginal likelihood and then sample its posterior, rather than
# fixing it by hand ({cite:t}`giannoneLenzaPrimiceri2015`; the Minnesota shrinkage idea goes
# back to {cite:t}`doan1984` and {cite:t}`litterman1986`).

# %%
LAGS = 12
data = VARData.from_df(anomalies, endog=list(anomalies.columns))

conjugate_prior = NIWPrior(select=True, decay=2.0, cross_shrinkage=1.0)
start = time.perf_counter()
fitted_conjugate = ConjugateVAR(lags=LAGS, prior=conjugate_prior, draws=2000, tune=1000, seed=0).fit(data)
conjugate_seconds = time.perf_counter() - start

lambda_hat = float(fitted_conjugate.idata.posterior["lambda_"].median())
print(f"data-selected Minnesota tightness  lambda = {lambda_hat:.3f}")
print(f"conjugate fit wall-clock                  = {conjugate_seconds:.2f} s")

# %% [markdown]
# The estimator reports a posterior for $\lambda$ (not a fixed value): the data speak to how
# much shrinkage the system needs. Everything downstream — coefficients, covariance, the base
# Cholesky factor — was drawn in closed form conditional on those hyperparameter draws.
#
# ## The same model by NUTS
#
# To make this a clean *inference-mode* comparison, we fit the NUTS VAR at the **same**
# tightness the conjugate estimator just selected (`MinnesotaPrior(tightness=lambda_hat)`).
# Now the only differences are the prior family (independent-Normal vs conjugate NIW) and
# the sampler — not the amount of shrinkage.

# %%
if ci:
    sampler = NUTSSampler(draws=10, tune=50, chains=1, cores=1, random_seed=0)
else:
    sampler = NUTSSampler(draws=1000, tune=1500, chains=2, cores=1, random_seed=0)

start = time.perf_counter()
fitted_nuts = VAR(lags=LAGS, prior=MinnesotaPrior(tightness=lambda_hat)).fit(data, sampler=sampler)
nuts_seconds = time.perf_counter() - start
print(f"NUTS fit wall-clock = {nuts_seconds:.2f} s")
az.summary(fitted_nuts.idata, var_names=["intercept"], kind="diagnostics")

# %% [markdown]
# ### Speed
#
# Both estimators fit the identical 4-variable, 12-lag system. The conjugate path spends
# Monte Carlo only on a single scalar; NUTS explores a ~200-dimensional coefficient
# posterior. At full render the gap is an order of magnitude or more.

# %%
if ci:
    print("CI smoke mode: NUTS is shrunk to a few draws — these timings are NOT representative.")
else:
    print(
        f"conjugate: {conjugate_seconds:.2f} s | "
        f"NUTS: {nuts_seconds:.2f} s | "
        f"speed-up: {nuts_seconds / conjugate_seconds:.0f}x"
    )

# %% [markdown]
# ## Downstream parity: identical structural machinery
#
# Because both estimators return a `FittedVAR`, identification is the same call on each. We
# apply a Cholesky scheme with the ordering `radiation → temperature → wind → precipitation`
# (solar forcing is the most exogenous; rainfall the most responsive). The ordering encodes
# real assumptions — see [Monetary Policy Analysis](monetary-policy.py) for how much it can
# matter — but here we hold it fixed and vary only the estimator.

# %%
ordering = ["radiation", "temperature", "wind", "precipitation"]
irf_conjugate = fitted_conjugate.set_identification_strategy(Cholesky(ordering=ordering)).impulse_response(horizon=24)
irf_nuts = fitted_nuts.set_identification_strategy(Cholesky(ordering=ordering)).impulse_response(horizon=24)

# %% mystnb={"figure": {"caption": "Conjugate-VAR impulse responses (Cholesky). Column shock → row response, over 24 months.", "name": "irf-conjugate"}} tags=["remove-input"]
fig = irf_conjugate.plot()
_ = fig.suptitle("Conjugate VAR — impulse responses", y=1.02)

# %% [markdown]
# Now overlay the two estimators on the same axes. If the conjugate VAR is a legitimate
# estimator and not a shortcut, its responses should track the NUTS responses in shape and
# sign, with band widths of the same order.

# %%
def irf_band(irf_result, shock, response, prob=0.9):
    """Return (horizons, median, hdi_low, hdi_high) for one shock→response pair."""
    draws = irf_result.idata.posterior_predictive["irf"].sel(shock=shock, response=response)
    median = draws.median(dim=("chain", "draw")).values
    hdi = az.hdi(draws, hdi_prob=prob)["irf"]
    return np.arange(median.shape[0]), median, hdi.sel(hdi="lower").values, hdi.sel(hdi="higher").values


# %% mystnb={"figure": {"caption": "Conjugate vs NUTS impulse responses at the same tightness. Medians (lines) and 90% bands (shaded).", "name": "irf-overlay"}} tags=["remove-input"]
pairs = [("radiation", "temperature"), ("temperature", "wind")]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (shock, response) in zip(axes, pairs, strict=True):
    for result, color, label in [(irf_conjugate, "C0", "conjugate"), (irf_nuts, "C1", "NUTS")]:
        horizons, median, low, high = irf_band(result, shock, response)
        ax.plot(horizons, median, color=color, label=label)
        ax.fill_between(horizons, low, high, color=color, alpha=0.2)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_title(f"{shock} shock → {response}")
    ax.set_xlabel("months")
    ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# The two estimators tell the same structural story: a positive radiation (sunshine) shock
# warms temperature; a warmth shock is followed by calmer winds. The medians track closely
# and the bands overlap. They are *not* identical — the conjugate NIW prior imposes a
# symmetric Kronecker structure across equations while the NUTS prior is independent-Normal —
# and that is exactly the point: the inference mode is a modelling choice, not a source of
# contradiction.

# %%
if not ci:
    correlations = [
        np.corrcoef(irf_band(irf_conjugate, shock, response)[1], irf_band(irf_nuts, shock, response)[1])[0, 1]
        for shock in ordering
        for response in ordering
    ]
    print(f"median-IRF shape correlation across all 16 shock/response pairs: {np.nanmean(correlations):.2f}")

# %% [markdown]
# ## When to reach for which
#
# Both estimators share the entire post-fitting pipeline — identification, IRFs, FEVDs,
# forecasts — so the choice is purely about the estimation path.
#
# | Reach for the **conjugate VAR** when… | Reach for the **NUTS VAR** when… |
# |----------------------------------------|-----------------------------------|
# | speed matters — hyperparameter selection, model comparison, or many refits | you need per-equation or asymmetric cross-variable shrinkage |
# | you want the tightness $\lambda$ chosen by the data (hierarchical) | you need stochastic volatility, sign restrictions, or external instruments |
# | the conjugate NIW (symmetric, Kronecker) prior suits the problem | you need arbitrary or non-conjugate priors |
# | the system is large and full MCMC over every coefficient is costly | you want full HMC convergence diagnostics on all coefficients |
#
# The conjugate VAR trades flexibility for closed-form speed and a data-driven prior. When
# your problem fits inside that trade — as macro and climate systems with symmetric Minnesota
# shrinkage usually do — it is the sharper tool. When you need volatility that moves or priors
# that bend per equation, the NUTS VAR is there, and everything you build on top is the same.
#
# <section class="consulting-cta">
#     <p>We currently have some <strong>availability for consulting</strong> on how Bayesian modelling, vector autoregressions, and impulso can be integrated into your team's macroeconomic, financial, and environmental forecasting work. If this sounds relevant, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation.</p>
# </section>
#
# ## References
#
# The works cited above are collected on the [project bibliography](../references.md) page.
