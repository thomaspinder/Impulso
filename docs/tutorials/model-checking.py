# ---
# jupyter:
#   jupytext:
#     default_lexer: python
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Model Checks and Validation

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
# A fitted model is a claim, and every claim deserves an audit. This tutorial walks the full
# checking workflow on real U.S. macroeconomic data: pretest the data *before* specifying the
# model, check the prior *before* sampling, check the sampler *after* it runs, and check the
# fitted model against the data it was fitted to. The sequence follows the Bayesian workflow
# described by {cite:t}`gelman2020` and {cite:t}`gabry2019` — each step is cheap, and each one
# catches a different way of being wrong.
#
# Two how-to guides cover the same APIs in reference form:
# [Testing for Stationarity and Cointegration](../how-to/stationarity-testing.md) and
# [Prior and Posterior Predictive Checks](../how-to/predictive-checks.md). This tutorial shows
# what running them on genuine data looks like, including the parts where the answers are messy.
#
# The stationarity pretests need the `diagnostics` extra
# (`pip install "impulso[diagnostics]"`), which pulls in `statsmodels`.

# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from impulso import (
    VAR,
    VARData,
    adf_test,
    integration_order,
    johansen_test,
    kpss_test,
    select_lag_order,
)
from impulso.samplers import NUTSSampler

# %% [markdown]
# ## The data
#
# We reuse the three-variable U.S. monetary policy system from the
# [Monetary Policy Analysis tutorial](monetary-policy.py): industrial production (`output`)
# and the consumer price index (`prices`), both entering as $100 \times \log$, and the
# effective federal funds rate (`rate`) in percentage points. The sample runs from January
# 1965 to December 2007 — monthly data, 516 observations. That tutorial explains why these
# variables and this sample; here they are simply a realistic test bed, with all the
# awkwardness real macro data brings.

# %% mystnb={"figure": {"caption": "The three series. Output and prices trend strongly upward; the funds rate is bounded but highly persistent.", "name": "checking-raw-data"}} tags=["remove-input"]
df = pd.read_csv("data/monetary_policy.csv", index_col="date", parse_dates=True)
df = df.loc[:"2007-12"]

fig, axes = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)
labels = {
    "output": "Log Industrial Production (x100)",
    "prices": "Log CPI (x100)",
    "rate": "Federal Funds Rate (%)",
}
for ax, col in zip(axes, df.columns, strict=True):
    ax.plot(df.index, df[col], linewidth=1, color="0.3")
    ax.set_ylabel(labels[col], fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_title("U.S. Monetary Policy Data (1965-2007)")
fig.tight_layout()

# %%
data = VARData.from_df(df, endog=["output", "prices", "rate"])

# %% [markdown]
# ## Step 1: pretest the data
#
# A VAR in levels and a VAR in differences answer different questions, and the choice should
# be made deliberately. The pretests report; they do not decide — the modelling decision at
# the end of this section is ours.
#
# ### Unit-root tests
#
# The Augmented Dickey-Fuller test ({cite:t}`dickeyFuller1979`) takes a unit root as its null,
# so a small p-value argues for stationarity. Because `output` and `prices` trend visibly, we
# allow a linear trend under the alternative with `regression="ct"` — testing a trending
# series against a constant-only alternative would stack the deck towards non-stationarity.

# %%
adf = adf_test(data, regression="ct")
adf.summary()

# %% [markdown]
# ADF rejects a unit root for `output` and `rate` at the 5% level, but not for `prices`.
#
# The KPSS test ({cite:t}`kwiatkowski1992`) flips the burden of proof: its null is
# stationarity, so a rejection argues for a unit root. Running both is standard practice
# precisely because ADF has poor power against persistent-but-stationary alternatives.

# %%
kpss = kpss_test(data, regression="ct")
kpss.summary()

# %% [markdown]
# KPSS rejects stationarity for **all three** series — note the `pvalue_bounded` column: the
# reported p-values sit at the edge of the published lookup table, so they are bounds, not
# estimates. The two tests now disagree on `output` and `rate`. That is not a malfunction; it
# is the classic signature of highly persistent series in a finite sample, where "unit root"
# and "root of 0.98" are close to observationally equivalent. What to do when the tests
# disagree is a topic of its own — see
# [Stationarity pitfalls](../how-to/climate-pitfalls.md) for a worked discussion.
#
# ### Integration order
#
# `integration_order` automates the difference-and-retest loop, with ADF driving the stopping
# rule and KPSS recorded alongside as a cross-check:

# %%
orders = integration_order(data, max_order=2, regression="ct")
print(f"order: {orders.order}")
print(f"d_max: {orders.d_max}")
print(f"inconclusive: {orders.inconclusive}")
orders.summary()

# %% [markdown]
# Every variable lands on the `inconclusive` list, each for an instructive reason:
#
# - `output` and `rate` stop at $d = 0$ with `joint_status = "conflicting"` — both tests
#   reject, which is the ADF/KPSS disagreement from above.
# - `prices` fails to reject a unit root even in first differences at $\alpha = 0.05$
#   (monthly inflation is itself very persistent) and only stops at $d = 2$. Whether the
#   price level is I(1) or I(2) is a genuinely unsettled question in the unit-root
#   literature, and a 43-year sample does not settle it either.
#
# The honest summary: `prices` is clearly non-stationary, `output` and `rate` are too
# persistent to classify cleanly. Treat the `order` numbers as a table to read, not a verdict
# to obey.
#
# ### Cointegration
#
# If the series are individually non-stationary they may still share long-run relationships,
# in which case differencing every series would throw those relationships away. The Johansen
# procedure ({cite:t}`johansen1991`) estimates how many such relationships exist. It needs a
# lag order first — `k_ar_diff` counts lagged *differences*, so it is $p - 1$ for a VAR($p$)
# in levels:

# %%
ic = select_lag_order(data, max_lags=12)
print(f"AIC selects {ic.aic}, BIC selects {ic.bic}, HQ selects {ic.hq}")

p = ic.hq
johansen = johansen_test(data, det_order=0, k_ar_diff=p - 1)
print(
    f"rank (trace): {johansen.rank}, rank (max eigenvalue): {johansen.rank_max_eigen}"
)
johansen.summary()

# %% [markdown]
# The criteria disagree in their usual pattern — AIC generous at 11, BIC parsimonious at 2 —
# and we take Hannan-Quinn's compromise of 3 lags. Both Johansen statistics then agree on a
# cointegration rank of **2**: two long-run relationships tie the three series together,
# leaving a single common stochastic trend.
#
# ### The decision: fit in levels
#
# Three facts point the same way. The series are non-stationary or nearly so; they are
# cointegrated, so differencing everything would discard the long-run structure; and
# {cite:t}`simsStockWatson1990` showed that Bayesian inference in a levels VAR is valid
# whether or not unit roots are present. The Minnesota prior ({cite:t}`doan1984`) is built
# for exactly this situation — it shrinks each equation towards a random walk, which is a
# good description of these series. So: a VAR(3) in levels with a Minnesota prior.

# %%
spec = VAR(lags=3, prior="minnesota")

# %% [markdown]
# ## Step 2: check the prior before sampling
#
# A prior predictive check asks whether the model *before seeing the likelihood* can produce
# data on the same planet as the data we have ({cite:t}`gelmanMengStern1996`;
# {cite:t}`gabry2019`). `VAR.prior_predictive` draws from exactly the graph `fit` will
# sample, so there is no risk of checking a different prior than the one we use:

# %%
prior = spec.prior_predictive(data, draws=500, random_seed=0)
axes = az.plot_ppc(prior, group="prior", num_pp_samples=50, coords={"var": ["rate"]})

# %% [markdown]
# The pooled density for the funds rate shows the prior band comfortably containing the
# observed distribution, without being absurdly wider. A quantile coverage check makes that
# quantitative across all three variables. Quantiles, not mean $\pm k \cdot$ sd: the default
# scale prior is HalfCauchy, which has no finite moments, so a prior predictive mean is
# meaningless — see the [predictive-checks how-to](../how-to/predictive-checks.md).

# %%
prior_draws = prior.prior_predictive["obs"].values[0]  # (draws, time, var)
lower, upper = np.quantile(prior_draws, [0.025, 0.975], axis=0)
observed = prior.observed_data["obs"].values

coverage = ((observed >= lower) & (observed <= upper)).mean()
width = np.median(upper - lower)
print(
    f"95% prior band covers {coverage:.1%} of the data; median band width {width:.0f}"
)

# %% [markdown]
# Full coverage, with a band roughly 210 units wide against series that span roughly 1 to
# 535. The prior is loose — as a shrinkage prior should be — but not so diffuse that it burns
# sampling effort in absurd regions. If the band had *excluded* chunks of the data, the prior
# would be fighting the likelihood, and we would revisit the Minnesota hyperparameters before
# spending a single MCMC second (see
# [The Minnesota Prior, From Scratch](minnesota-prior.py) for tuning `tightness`).
#
# ## Step 3: fit, then interrogate the sampler
#
# `fit` samples the posterior with NUTS, the adaptive Hamiltonian Monte Carlo variant of
# {cite:t}`hoffmanGelman2014`. HMC is efficient, and — just as valuable — it is *loud* when it
# fails: its diagnostics flag problems that would pass silently in older samplers
# ({cite:t}`betancourt2017`). Loud only helps if you listen, so after every fit we check, in
# order: divergences, $\widehat{R}$, effective sample size, and the energy diagnostic.
#
# We start with production-scale settings and default mass-matrix adaptation:

# %%
if ci:
    sampler = NUTSSampler(
        draws=50, tune=500, chains=2, cores=1, target_accept=0.9, random_seed=123
    )
else:
    sampler = NUTSSampler(
        draws=1500,
        tune=1500,
        chains=4,
        cores=4,
        target_accept=0.9,
        random_seed=123,
        nuts_sampler="nutpie",
    )

fitted = spec.fit(data, sampler=sampler)

# %% [markdown]
# :::{admonition} Smoke renders
# :class: warning
# In CI this notebook runs with drastically shrunk MCMC (`IMPULSO_DOCS_CI=1`), so the printed
# diagnostics in a smoke build are not representative. The narrative describes the
# full-fidelity render.
# :::
#
# ### Divergences
#
# A divergence is a numerical failure of the leapfrog integrator, and it is the sharpest
# signal HMC emits: divergences concentrate exactly where the posterior has geometry the
# sampler cannot traverse, so the affected region is *under-explored* and estimates that pass
# through it are biased ({cite:t}`betancourt2017`). Zero is the only acceptable count:

# %%
print(f"divergences: {int(fitted.idata.sample_stats.diverging.sum())}")

# %% [markdown]
# None — but zero divergences is necessary, not sufficient. The next checks catch failures
# that never trip the integrator.
#
# ### $\widehat{R}$ and effective sample size
#
# `az.summary(kind="diagnostics")` reports, per parameter:
#
# - **`r_hat`** — the rank-normalised split-$\widehat{R}$ of {cite:t}`vehtari2021`, the
#   modern refinement of {cite:t}`gelmanRubin1992`. It compares within-chain and
#   between-chain variance; at convergence it is 1, and values above **1.01** mean the chains
#   have not agreed on what the posterior looks like.
# - **`ess_bulk`** — the effective number of independent draws for the centre of the
#   posterior, which governs how well means and medians are estimated.
# - **`ess_tail`** — the same for the 5% and 95% quantiles, which governs credible-interval
#   endpoints. A chain can estimate the centre well and the tails badly.
# - **`mcse_mean` / `mcse_sd`** — the Monte Carlo standard error those ESS values imply.
#
# {cite:t}`vehtari2021` recommend at least 400 effective draws (100 per chain on four
# chains) before trusting the estimates. Rather than eyeball hundreds of rows, filter for
# violations:

# %%
summ = az.summary(fitted.idata, kind="diagnostics")
flagged = summ[(summ["r_hat"] > 1.01) | (summ["ess_bulk"] < 400)]
print(f"{len(flagged)} of {len(summ)} parameters flagged")
flagged.sort_values("r_hat", ascending=False).head(8)

# %% [markdown]
# The fit fails the check. In the full render, $\widehat{R}$ reaches 1.13 with bulk ESS in
# the low dozens, and the flagged rows are not random: they concentrate in the `prices`
# equation's own lags (`B[prices, L1.prices]`, `L2.prices`, `L3.prices`) and their
# neighbours. A trace plot shows what that means in the raw draws — the four chains disagree
# about the coefficient's location and wander slowly within it, instead of overlapping as
# indistinguishable noise:

# %%
az.plot_trace(
    fitted.idata,
    var_names=["B"],
    coords={"var": ["prices"], "coeff": ["L1.prices", "L2.prices"]},
)
plt.tight_layout()

# %% [markdown]
# ### The energy diagnostic
#
# `az.plot_energy` overlays the distribution of the Hamiltonian energy with the distribution
# of its step-to-step changes; when the two differ badly (Bayesian fraction of missing
# information, BFMI, below about 0.3) the sampler cannot move between energy levels fast
# enough to explore the tails ({cite:t}`betancourt2017`):

# %%
az.plot_energy(fitted.idata)

# %% [markdown]
# The two distributions overlap and BFMI is near 1 — this check passes. That is the point of
# running the checks as a battery: divergences and energy are clean while $\widehat{R}$ and
# ESS fail, and only the full battery tells you the problem is slow mixing rather than
# pathological geometry.
#
# ### Fixing it
#
# Why does mixing fail here? The regressors of a levels VAR are lagged copies of
# near-random-walk series — `prices` at lags one, two, and three are almost the same
# variable. Individually each coefficient is weakly identified even though their sum is
# well determined, so the posterior has long, narrow, correlated ridges, and NUTS's default
# *diagonal* mass matrix cannot be scaled to match. This is a known pathology of large
# VARs, and nutpie ships a remedy: a low-rank modified mass matrix that adapts to exactly
# this kind of ill-conditioning.

# %%
if ci:
    sampler = NUTSSampler(
        draws=50,
        tune=500,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=123,
        nuts_sampler_kwargs={"low_rank_modified_mass_matrix": True},
    )
else:
    sampler = NUTSSampler(
        draws=1500,
        tune=1500,
        chains=4,
        cores=4,
        target_accept=0.9,
        random_seed=123,
        nuts_sampler="nutpie",
        nuts_sampler_kwargs={"low_rank_modified_mass_matrix": True},
    )

fitted = spec.fit(data, sampler=sampler)

summ = az.summary(fitted.idata, kind="diagnostics")
flagged = summ[(summ["r_hat"] > 1.01) | (summ["ess_bulk"] < 400)]
print(f"divergences: {int(fitted.idata.sample_stats.diverging.sum())}")
print(f"{len(flagged)} of {len(summ)} parameters flagged")
print(
    f"max r_hat: {summ['r_hat'].max():.3f}, min ess_bulk: {summ['ess_bulk'].min():.0f}"
)

# %% [markdown]
# In the full render every flag clears: no divergences, $\widehat{R} = 1.00$ throughout, and
# a minimum bulk ESS in the thousands. A rank plot — {cite:t}`vehtari2021`'s preferred
# convergence visual — confirms it on the previously worst coefficient. When chains mix, each
# chain's draws are spread uniformly across the pooled ranking, so all histograms should look
# flat:

# %%
az.plot_rank(
    fitted.idata,
    var_names=["B"],
    coords={"var": ["prices"], "coeff": ["L1.prices", "L2.prices"]},
)
plt.tight_layout()

# %% [markdown]
# :::{admonition} Going deeper
# :class: note
# This section covers the checks you should never skip. For the full menu — MCSE-aware
# reporting, chain-splitting subtleties, folded $\widehat{R}$ — see the
# [ArviZ API reference](https://python.arviz.org/en/stable/api.html)
# ({cite:t}`kumar2019`), {cite:t}`vehtari2021`, and {cite:t}`betancourt2017`.
# :::
#
# ## Step 4: check the fit against the data
#
# The sampler converged; now, did it converge to a model worth having? A posterior predictive
# check replicates the estimation sample from the posterior and compares the replicates with
# what actually happened ({cite:t}`gelmanMengStern1996`). Each replicate is one-step-ahead
# conditioned on the observed lags — the standard predictive object for a conditional model,
# and the one `az.plot_ppc` expects:

# %%
ppc = fitted.posterior_predictive(seed=0)
axes = az.plot_ppc(ppc, num_pp_samples=100, coords={"var": ["rate"]})

# %% [markdown]
# The replicate densities track the observed density closely — compare with the prior
# predictive version of this plot above, where the band was hundreds of units wide. The same
# quantile coverage check, now on the posterior:

# %%
rep = ppc.posterior_predictive["obs"].values  # (chain, draw, time, var)
flat = rep.reshape(-1, *rep.shape[2:])  # (chain*draw, time, var)
lower, upper = np.quantile(flat, [0.025, 0.975], axis=0)
observed = ppc.observed_data["obs"].values

pooled = ((observed >= lower) & (observed <= upper)).mean()
per_var = ((observed >= lower) & (observed <= upper)).mean(axis=0)
print(f"95% band covers {pooled:.1%} of observations")
for name, cov in zip(data.endog_names, per_var, strict=True):
    print(f"  {name}: {cov:.1%}")

# %% mystnb={"figure": {"caption": "Observed series against the 95% posterior predictive band. The band is one-step-ahead, so it hugs the data; the check is whether the right fraction of points escape it.", "name": "checking-ppc-bands"}} tags=["remove-input"]
time = df.index[3:]
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for i, (ax, name) in enumerate(zip(axes, data.endog_names, strict=True)):
    ax.fill_between(
        time,
        lower[:, i],
        upper[:, i],
        color="C0",
        alpha=0.3,
        label="95% predictive band",
    )
    ax.plot(time, observed[:, i], color="0.2", linewidth=0.8, label="observed")
    ax.set_ylabel(labels[name], fontsize=9)
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper left", fontsize=8)
axes[0].set_title("Posterior predictive check, one-step-ahead")
fig.tight_layout()

# %% [markdown]
# In the full render the band covers about 95% of observations overall and per variable —
# the model is neither over-confident (coverage well below 95%) nor over-dispersed (coverage
# near 100%). One caveat worth carrying forward: the covariance here is constant through
# time, while the funds-rate panel visibly is not — the early-1980s Volcker period strains
# the band. The [Stochastic Volatility tutorial](stochastic-volatility.py) relaxes exactly
# that assumption.
#
# For residual diagnostics — subtracting the conditional mean under parameter uncertainty via
# `posterior_predictive(simulate_innovations=False)` — and for attaching the replicates to
# `fitted.idata`, see the [predictive-checks how-to](../how-to/predictive-checks.md).
#
# ## The checklist
#
# | When | Question | Tool |
# | --- | --- | --- |
# | Before specifying | Levels or differences? Shared trends? | `adf_test`, `kpss_test`, `integration_order`, `johansen_test` |
# | Before sampling | Can the prior produce plausible data? | `VAR.prior_predictive` + `az.plot_ppc(group="prior")` |
# | After sampling | Did the sampler explore the posterior? | divergences, `az.summary(kind="diagnostics")`, `az.plot_trace`, `az.plot_rank`, `az.plot_energy` |
# | After sampling | Does the model reproduce the sample? | `FittedVAR.posterior_predictive` + `az.plot_ppc`, quantile coverage |
#
# None of these checks proves the model right; each can prove it wrong cheaply. Run all four
# before you forecast, identify shocks, or hand results to anyone.
#
# ## What's next
#
# - **Forecast** with the checked model — the [Forecasting tutorial](forecasting.py)
# - **Identify structural shocks** — the [Structural Analysis tutorial](structural-analysis.py)
# - **Time-varying volatility**, for the constant-covariance caveat above — the
#   [Stochastic Volatility tutorial](stochastic-volatility.py)
#
# <section class="consulting-cta">
#     <p>We currently have some <strong>availability for consulting</strong> on how Bayesian modelling, vector autoregressions, and impulso can be integrated into your team's macroeconomic and financial forecasting work. If this sounds relevant, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation.</p>
# </section>
