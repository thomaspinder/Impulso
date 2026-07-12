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
#     path: /Users/thomaspinder/Library/Jupyter/kernels/python3
# ---

# %% [markdown]
# # Oil supply news with an external instrument
#
# An OPEC announcement can move the oil price today even when it does not change production today. On 14 December 2006, for example, OPEC announced a cut of 500,000 barrels per day that would take effect the following February. The oil price rose by about 2% on the announcement day. The market was reacting to news about future supply.
#
# Känzig (2021) uses this timing to ask what happens after expectations of future oil supply deteriorate. The distinction from an unexpected loss of current production is important. A shortage today forces users to draw down inventories. News of a shortage tomorrow instead gives them a reason to build inventories while oil is still available.
#
# | Response today | Unexpected supply loss today | News of lower supply in the future |
# |---|---|---|
# | Oil price | Rises | Rises |
# | Oil production | Falls immediately | Changes little at first |
# | Oil inventories | Fall as stocks are used | Rise as stocks are accumulated |
#
# This tutorial reproduces Känzig's six-variable oil-market VAR and then estimates a Bayesian version with Impulso. The identifying information comes from changes in oil futures prices around OPEC announcements. In the structural VAR literature, a separate series used in this way is called an **external instrument**, or **proxy**.
#
# The aim is deliberately narrow. The proxy identifies the shock associated with news about future oil supply; it does not turn every oil-price movement into an oil supply shock, nor does it identify all the other shocks in the system.

# %% tags=["remove-cell"]
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pytensor").setLevel(logging.ERROR)

# %% tags=["remove-cell"]
import os

# Smoke-render flag: set IMPULSO_DOCS_CI=1 to shrink MCMC for fast CI builds.
ci = os.environ.get("IMPULSO_DOCS_CI") == "1"

# %% [markdown]
# ## How the announcement surprise identifies one shock
#
# A VAR first removes the movements that can be predicted from past data. What remains is a vector of residuals, $u_t$. These residuals are still mixtures: an unexpectedly high oil price in a given month could reflect supply news, demand news, geopolitical events, or several shocks at once. A structural VAR writes that mixture as
#
# $$
# u_t = P \varepsilon_t .
# $$
#
# Here $\varepsilon_t$ collects the unobserved economic shocks, while each column of $P$ records how one of those shocks moves the six observed variables on impact.
#
# Let $z_t$ denote the futures-price change around an OPEC announcement. It can isolate the oil supply news shock, $\varepsilon_{1t}$, under two assumptions:
#
# - **Relevance:** the announcement surprise moves with oil supply news, so $\mathbb{E}[z_t \varepsilon_{1t}] = \phi \neq 0$.
# - **Exogeneity:** it does not move systematically with the other structural shocks, so $\mathbb{E}[z_t \varepsilon_{jt}] = 0$ for $j \neq 1$.
#
# Under these assumptions,
#
# $$
# \mathbb{E}[z_t u_t] = P\, \mathbb{E}[z_t \varepsilon_t] = \phi\, p_1 .
# $$
#
# Once the VAR has been estimated, we can estimate the left-hand side with the sample covariance between the proxy and the residuals. The right-hand side says that this covariance points in the same direction as $p_1$, the impact effects of oil supply news. The unknown $\phi$ means that the scale is not identified, so we must choose a normalisation. Following Känzig, we scale the shock to raise the real oil price by 10% on impact. The fitted VAR then traces the response over subsequent months.
#
# This argument identifies only $p_1$. It requires no causal ordering among the six variables and imposes no signs on their responses. The rise in inventories and delayed fall in production are therefore evidence for the news interpretation, not restrictions built into it.

# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from impulso import VAR, VARData
from impulso.identification import ProxySVAR
from impulso.samplers import NUTSSampler

# %% [markdown]
# ## Data: OPEC announcements and monthly macro variables
#
# Känzig builds the proxy from 119 OPEC announcements between 1983 and 2017. For each announcement, he measures the change in West Texas Intermediate (WTI) futures settlement prices from the last trading day before the announcement to the announcement day. OPEC production decisions usually take effect with a delay, so futures prices are a natural place to look for the market's response. If the futures risk premium does not change within the one-day window, the price change measures a revision in the expected future oil price.
#
# The one-day window is a compromise. It gives markets time to interpret an OPEC statement, whose precise release time is often unavailable and whose contents may leak before publication. But a full day also leaves room for unrelated news to move oil prices. The proxy is therefore not automatically valid because it is measured at high frequency. Its interpretation still rests on the exogeneity assumption above. Känzig studies this concern with control days, an alternative heteroskedasticity-based estimator, and several other checks; this tutorial reproduces his baseline external-instrument specification.
#
# The files come from Känzig's public replication repository ([dkaenzig/replicationOilSupplyNews](https://github.com/dkaenzig/replicationOilSupplyNews)) and serve two different purposes:
#
# - **VAR data (monthly, 1974M01-2017M12):** the real oil price, world oil production, world oil inventories, world industrial production, U.S. industrial production, and the U.S. consumer price index (CPI). The series enter as $100 \times \log(\cdot)$, and the oil price is deflated by U.S. CPI.
# - **Proxy (indexed 1975M01-2017M12):** the first principal component of announcement-day changes in WTI futures prices at maturities from 1 to 12 months. The underlying futures data begin in 1983. Surprises from multiple announcements in the same month are added together; a month without an announcement is recorded as zero, not as missing.

# %%
data_df = pd.read_csv("data/kaenzig_data.csv", index_col=0, parse_dates=True)
instrument = pd.read_csv("data/kaenzig_instrument.csv", index_col=0, parse_dates=True)["oil_surprise"]

var_names_paper = [
    "Real oil price", "World oil production", "World oil inventories",
    "World industrial production", "U.S. industrial production", "U.S. CPI",
]
data = VARData(endog=data_df.values, endog_names=list(data_df.columns), index=data_df.index)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(instrument.index, instrument.values, lw=0.8)
ax.set_ylabel("Oil price expectation revision [%]")
ax.set_title("Oil futures surprises around OPEC announcements")
plt.tight_layout()

# %% [markdown]
# The proxy is sparse because OPEC does not announce a new production decision every month. A positive value means that futures prices rose around the announcement. In Känzig's interpretation, the market had received adverse news about future supply. The proxy is not used as the shock itself: it misses supply news that arrives outside OPEC announcements and contains measurement noise. Instead, it selects the part of the VAR residuals that co-moves with OPEC news.
#
# ## First recover the published benchmark
#
# Before changing the estimator, we reproduce the paper's benchmark in NumPy. `_kaenzig_frequentist.py` follows the same sequence: estimate a 12-lag VAR in levels by ordinary least squares, use the announcement surprise as an instrument for the oil-price residual, and construct confidence bands with a moving-block bootstrap. Matching this result gives the Bayesian comparison a known reference point.

# %%
from _kaenzig_frequentist import first_stage_f, proxy_var_kaenzig, var_ols

nsim = 50 if ci else 1000
freq = proxy_var_kaenzig(data_df.values, instrument.values, p=12, horizon=50, shock_size=10.0, nsim=nsim, seed=0)

_, U_ols = var_ols(data_df.values, 12)
print(f"First stage: F = {first_stage_f(instrument.values, U_ols[:, 0]):.2f}, "
      f"robust F = {first_stage_f(instrument.values, U_ols[:, 0], robust=True):.2f}")

# %% [markdown]
# Känzig reports a conventional first-stage F-statistic of 22.67 for the composite proxy and a heteroskedasticity-robust value of 10.55; the local calculation recovers these figures. The proxy explains 4.22% of the monthly oil-price residual. That is enough signal to be useful, but not so much that instrument strength can be ignored. The familiar $F>10$ rule is only a screening device, and the robust result sits just above it. An F-statistic also says nothing about exogeneity: a strong instrument can still be invalid.
#
# ## What changes in the Bayesian version
#
# The economic design and the dynamic specification stay fixed. We still use the same six variables, 12 monthly lags, a constant, and the same proxy. What changes is how the reduced-form VAR and its uncertainty are estimated.
#
# | Step | Känzig's benchmark | Impulso version |
# |---|---|---|
# | VAR coefficients | Ordinary least squares | Bayesian posterior with a Minnesota prior |
# | Identification | One impact vector from the external instrument | The external-instrument impact vector, estimated for every posterior draw |
# | Uncertainty | Moving-block bootstrap | Posterior credible intervals, conditional on the observed proxy series |
#
# This distinction matters for the comparison below. The Bayesian analysis is not simply Känzig's estimator with different plotting code: the Minnesota prior shrinks a large system toward persistent, parsimonious dynamics. With six variables and 12 lags, each equation contains 72 lag coefficients. The low-rank mass-matrix setting helps NUTS sample their strong posterior correlations; it is a computational setting, not an additional economic assumption.

# %%
if ci:
    sampler = NUTSSampler(draws=10, tune=50, chains=1, cores=1, random_seed=42)
else:
    sampler = NUTSSampler(
        draws=1000,
        tune=1500,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=42,
        nuts_sampler_kwargs={"low_rank_modified_mass_matrix": True},
    )

fitted = VAR(lags=12, prior="minnesota").fit(data, sampler=sampler)

summ = az.summary(fitted.idata, var_names=["B"], kind="diagnostics")
print(f"B coefficients: min ESS = {summ['ess_bulk'].min():.0f}, "
      f"median ESS = {summ['ess_bulk'].median():.0f}, "
      f"max r_hat = {summ['r_hat'].max():.3f}")
print(f"divergences: {int(fitted.idata.sample_stats['diverging'].sum())}")

# %% [markdown]
# The full run reports no divergences, a minimum bulk effective sample size above 1,500, and a maximum $\hat R$ of 1.01. Those diagnostics give us a sound set of posterior draws for the comparison. They are worth checking before identification: a proxy cannot rescue a VAR whose posterior has not been sampled reliably.
#
# Once the VAR has been fitted, identification is one additional step. For each posterior draw, `ProxySVAR` reconstructs the monthly residuals and calculates how they co-move with the announcement surprise. Dates are matched through the two `DatetimeIndex` objects. A month absent from the proxy is dropped; a month present with a value of zero remains in the calculation and means that no OPEC announcement occurred.
#
# We orient the shock so that adverse supply news raises the oil price. Setting `scale=10.0` then applies Känzig's normalisation: the real oil price rises by 10% on impact.

# %%
scheme = ProxySVAR(
    instrument=instrument,
    policy_variable="real_oil_price",
    shock_name="oil_supply_news",
    scale=10.0,
)
ivar = fitted.set_identification_strategy(scheme)
irf = ivar.impulse_response(horizon=50)

# %% [markdown]
# The structural shock matrix stores a summary of the first-stage diagnostics. We can also inspect the full distribution. The instrument is fixed across draws, but the reconstructed oil-price residual changes with the VAR coefficients. Each draw therefore implies a different first-stage F-statistic:

# %%
sm = ivar.shock_matrix()
{k: round(v, 2) for k, v in sm.attrs.items()}

# %%
f_draws = scheme.first_stage(fitted.idata.posterior, data, n_lags=12).ravel()

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(f_draws, bins=40, color="C0", alpha=0.75)
ax.axvline(10, color="k", ls="--", lw=1, label="F = 10 rule of thumb")
ax.axvline(np.median(f_draws), color="C1", lw=1.5, label=f"posterior median = {np.median(f_draws):.1f}")
ax.set_xlabel("First-stage F")
ax.set_title("Posterior of instrument relevance")
ax.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# Most of the posterior mass lies above the $F=10$ rule of thumb. This is reassuring about relevance, but it does not settle the exclusion question. Nor is this distribution a bootstrap of the announcement series: it records uncertainty about the fitted VAR while treating the observed proxy as fixed.
#
# ## What follows adverse oil supply news?
#
# The figure overlays two analyses of the same six-variable system. Blue shows the Impulso posterior median with 68% and 90% credible intervals. Orange shows a local reproduction of Känzig's OLS point estimate and moving-block-bootstrap confidence bands. The paper uses 10,000 bootstrap replications; the full notebook uses 1,000 to keep the render manageable. These intervals have different interpretations, so their widths should not be read as a contest between methods. The useful comparison is whether the estimated paths tell the same economic story.

# %%
irf_draws = irf.idata.posterior_predictive["irf"].sel(shock="oil_supply_news")
med = irf_draws.median(dim=("chain", "draw")).values
q = {p: irf_draws.quantile(p, dim=("chain", "draw")).values for p in (0.05, 0.16, 0.84, 0.95)}

horizon = np.arange(51)
fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True)
for j, (ax, name) in enumerate(zip(axes.ravel(), var_names_paper)):
    ax.fill_between(horizon, q[0.05][:, j], q[0.95][:, j], alpha=0.18, color="C0", label="Impulso 90%")
    ax.fill_between(horizon, q[0.16][:, j], q[0.84][:, j], alpha=0.35, color="C0", label="Impulso 68%")
    ax.plot(horizon, med[:, j], color="C0", lw=1.8, label="Impulso median")
    ax.plot(horizon, freq.irf[:, j], color="C1", lw=1.5, ls="--", label="Känzig point")
    for lo, hi in (freq.bands68, freq.bands90):
        ax.plot(horizon, lo[:, j], color="C1", lw=0.8, ls=":")
        ax.plot(horizon, hi[:, j], color="C1", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(name, fontsize=11)
    ax.set_xlim(0, 50)
    if j >= 3:
        ax.set_xlabel("Months")
    if j % 3 == 0:
        ax.set_ylabel("%")
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("Response to an oil supply news shock raising the real oil price by 10%", y=1.0)
plt.tight_layout()

# %% [markdown]
# The first three panels contain the central economic result. The oil price rises by construction. Oil production, however, changes little on impact and declines only later, while inventories begin to rise immediately. That is the timing predicted by news of a future shortfall: firms store oil before supply tightens. An unexpected shortfall today would instead force production down and inventories to be used. Because the model imposes none of these signs, the pattern is evidence in favour of the interpretation rather than a restatement of the identifying assumptions.
#
# The news also propagates beyond the oil market. World industrial production is nearly unchanged during the first year and then falls. U.S. industrial production declines sooner and more sharply, while the U.S. price level rises. In Känzig's estimates, a shock that raises the oil price by 10% eventually lowers world oil production by about 0.7%, raises inventories by 1.2%, lowers world and U.S. industrial production by 0.6% and 1%, and raises U.S. CPI by about 0.4%. The Impulso medians closely reproduce those magnitudes and dynamics.
#
# The blue credible intervals are generally narrower than the orange bootstrap bands, especially at long horizons. Two differences matter. The Minnesota prior regularises the 12-lag system, whereas Känzig estimates its coefficients by OLS. Impulso's intervals also condition on the observed proxy, while the moving-block bootstrap resamples the time-series observations used for identification. Narrower blue bands therefore do not establish a general precision advantage for Bayesian estimation.
#
# The clearest difference appears in U.S. CPI after about 30 months. Impulso's posterior median decays more slowly than Känzig's point estimate, though the published path remains inside the 90% credible interval. Random-walk shrinkage of the persistent CPI series is a plausible explanation, but establishing that claim would require a prior-sensitivity exercise. This comparison also covers only the paper's baseline external-instrument design; Känzig's checks for event-window noise and alternative specifications remain important evidence for the economic interpretation.
#
# ## Extension: allow the shock scale to change over time
#
# Känzig's benchmark uses one residual covariance matrix, $\Sigma$, for the full sample. The next analysis is an extension rather than part of the replication. Impulso can replace that constant matrix with a sequence, $\Sigma_t$, so the model can represent calm and volatile periods differently. The proxy continues to determine which direction in the residuals represents oil supply news; the time-varying covariance determines the size of a one-standard-deviation shock in each month.
#
# To keep the notebook practical to render, we fit this stochastic-volatility model to four variables: the real oil price, world oil production, U.S. industrial production, and U.S. CPI. The same API accepts the six-variable system, but that model contains more than 3,000 latent volatility states. Because the system is smaller, the results below demonstrate the extension and should not be treated as a direct robustness check of the six-variable benchmark.

# %%
core = ["real_oil_price", "world_oil_production", "us_ip", "us_cpi"]
data_sv = VARData(endog=data_df[core].values, endog_names=core, index=data_df.index)

if ci:
    sv_sampler = NUTSSampler(draws=10, tune=50, chains=1, cores=1, random_seed=42)
else:
    sv_sampler = NUTSSampler(
        draws=500,
        tune=1000,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=42,
        nuts_sampler_kwargs={"low_rank_modified_mass_matrix": True},
    )

fitted_sv = VAR(lags=12, volatility="sv").fit(data_sv, sampler=sv_sampler)

summ_sv = az.summary(fitted_sv.idata, var_names=["B"], kind="diagnostics")
print(f"B coefficients: min ESS = {summ_sv['ess_bulk'].min():.0f}, "
      f"max r_hat = {summ_sv['r_hat'].max():.3f}, "
      f"divergences: {int(fitted_sv.idata.sample_stats['diverging'].sum())}")

# %% [markdown]
# The stochastic-volatility fit is less clean than the baseline: the full render reports a small number of divergences (see the count printed above) and a maximum $\hat R$ of about 1.02. That is adequate for demonstrating how the interfaces compose, but not for a final empirical analysis. A substantive application should run longer and resolve the divergences before interpreting the time-varying scale.
#
# With time-varying volatility, `shock_matrix(at="all")` returns an impact matrix for every month. Keeping the 10% normalisation would deliberately make the oil-price impact constant and hide the variation we want to inspect. We therefore set `scale=None` and plot the model-implied impact of a one-standard-deviation oil supply news shock on the real oil price.

# %%
scheme_sd = ProxySVAR(instrument=instrument, policy_variable="real_oil_price", shock_name="oil_supply_news")
ivar_sd = fitted_sv.set_identification_strategy(scheme_sd)
sd_path = ivar_sd.shock_matrix(at="all").sel(response="real_oil_price", shock="oil_supply_news")
sd_med = sd_path.median(dim=("chain", "draw"))
sd_lo = sd_path.quantile(0.16, dim=("chain", "draw"))
sd_hi = sd_path.quantile(0.84, dim=("chain", "draw"))

fig, ax = plt.subplots(figsize=(10, 3.5))
t = sd_path.coords["time"].values
ax.fill_between(t, sd_lo, sd_hi, alpha=0.3, color="C0")
ax.plot(t, sd_med, color="C0", lw=1.2)
ax.set_ylabel("% oil price impact")
ax.set_title("One-standard-deviation oil supply news shock, period by period")
plt.tight_layout()

# %% [markdown]
# The scale varies substantially over the sample. Read this plot carefully: it is a path of conditional shock sizes implied by $\Sigma_t$, not a historical decomposition and not a record of when oil supply news occurred. A high value means that one standard deviation in the identified direction corresponds to a larger contemporaneous oil-price movement in that month. It does not show that oil supply news caused the surrounding episode. A constant-$\Sigma$ model replaces this entire path with one average scale.
#
# ## Respect the boundary of the identification
#
# The proxy identifies one column of the impact matrix. Some downstream calculations require a square, invertible matrix, so Impulso supplies an orthogonal completion for the other columns and labels them `unidentified_*`. Those columns are computational placeholders: rotating them would leave the identified oil supply news shock and the reduced-form covariance unchanged.
#
# The result methods preserve that distinction:
#
# - `fevd()` asks how much forecast-error variance each shock explains. It reports the share attributable to oil supply news but returns NaN for each `unidentified_*` share, because those individual shares change under an arbitrary rotation. FEVD also requires one-standard-deviation shocks, so use `scale=None` rather than the 10% normalisation.
# - `historical_decomposition()` attributes the observed path to shocks over time. It combines the unidentified columns into `unidentified_remainder`: their separate paths are arbitrary, but their sum is the part not attributed to oil supply news.
# - `ProxySVAR` emits a warning when the posterior-median first-stage F-statistic is below 10. The warning flags weak relevance; it cannot diagnose a violation of exogeneity.
#
# ## Reproducing this notebook
#
# The data CSVs ship with the documentation. To rebuild them from the original files, clone [dkaenzig/replicationOilSupplyNews](https://github.com/dkaenzig/replicationOilSupplyNews) and follow `mainAnalysisOilSupplyNews.m`: transform the variables to $100 \times \log(\cdot)$, deflate the oil price by CPI, and use column 15 for the principal-component futures surprise. A full render fits two MCMC models and takes about fifteen minutes on a laptop.
#
# ## References
#
# - Känzig, D. R. (2021). The macroeconomic effects of oil supply news: Evidence from OPEC announcements. *American Economic Review*, 111(4), 1092-1125.
# - Mertens, K., and Ravn, M. O. (2013). The dynamic effects of personal and corporate income tax changes in the United States. *American Economic Review*, 103(4), 1212-1247.
# - Stock, J. H., and Watson, M. W. (2018). Identification and estimation of dynamic causal effects in macroeconomics using external instruments. *The Economic Journal*, 128(610), 917-948.
