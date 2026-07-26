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
# # The Minnesota Prior, From Scratch

# %% tags=["remove-cell"]
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pytensor").setLevel(logging.ERROR)

# %% tags=["remove-cell"]
import os

# Smoke-render flag: IMPULSO_DOCS_CI=1 shrinks MCMC for fast CI builds.
ci = os.environ.get("IMPULSO_DOCS_CI") == "1"

# %% [markdown]
# Every Bayesian VAR in Impulso ships with `prior="minnesota"` switched on by default. This
# tutorial explains what that default is doing to your model: the arithmetic behind it, the
# three numbers you can turn, and what each one costs you.
#
# It assumes you know what a VAR is and have seen a normal distribution. It does *not*
# assume you have met a Minnesota prior before. If you want the compressed reference version
# instead, read the [Minnesota prior explanation](../explanation/minnesota-prior.md).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from impulso import VAR, MinnesotaPrior, VARData
from impulso.samplers import NUTSSampler

# %% [markdown]
# ## The problem: a VAR runs out of data fast
#
# A VAR with $n$ variables and $p$ lags has $n^2 p$ slope coefficients plus $n$ intercepts.
# The quadratic term is the trouble. Adding one variable to a 5-variable VAR(4) adds 44 slope
# coefficients, not 4. Meanwhile the sample gives you $nT$ numbers to estimate them from,
# which grows only linearly.
#
# {numref}`minnesota-parameter-count` plots both sides for a quarterly sample of $T = 200$,
# roughly fifty years of data — about as long a macroeconomic series as you will ever be
# handed.

# %% mystnb={"figure": {"caption": "Free VAR coefficients against available data points for a sample of $T = 200$. The parameter count grows quadratically in the number of variables; the data does not.", "name": "minnesota-parameter-count"}} tags=["remove-input"]
n_grid = np.arange(2, 13)
T_ref = 200

fig, ax = plt.subplots(figsize=(7, 4))
for p, style in zip([1, 2, 4, 8], ["-", "--", "-.", ":"], strict=True):
    ax.plot(n_grid, n_grid**2 * p + n_grid, style, color="crimson", label=f"coefficients, $p={p}$")
ax.plot(n_grid, n_grid * T_ref, color="0.3", linewidth=2, label=f"data points ($nT$, $T={T_ref}$)")
ax.set_yscale("log")
ax.set_xlabel("number of variables $n$")
ax.set_ylabel("count (log scale)")
ax.set_title("Parameters grow quadratically, data does not")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
# An 8-variable VAR(8) has 520 coefficients against 1,600 data points. Ordinary least
# squares will happily fit it, and the fit will be mostly noise: the estimates have enormous
# sampling variance, and the implied dynamics are frequently explosive. This is the *curse of
# dimensionality* that {cite:t}`sims1980` flagged in the paper that introduced VARs to
# macroeconomics, and it is the reason {cite:t}`doan1984` and {cite:t}`litterman1986` — then
# at the Federal Reserve Bank of Minneapolis and the University of Minnesota — proposed
# fixing it with a prior rather than with more data.
#
# ## The idea: start from a random walk
#
# Write the VAR in the form Impulso estimates:
#
# $$
# y_t = c + \sum_{l=1}^{p} A_l\, y_{t-l} + u_t,
# \qquad u_t \sim \mathcal{N}(0, \Sigma)
# $$ (eq-minnesota-var)
#
# where $y_t \in \mathbb{R}^n$ and each $A_l$ is $n \times n$. Stack the lag matrices
# side by side into the single coefficient matrix Impulso samples,
#
# $$
# B = \begin{bmatrix} A_1 & A_2 & \cdots & A_p \end{bmatrix} \in \mathbb{R}^{n \times np},
# $$ (eq-minnesota-stack)
#
# and write $\beta_{ij}^{(l)}$ for the entry of $A_l$ in row $i$, column $j$: the response of
# variable $i$ to lag $l$ of variable $j$.
#
# The Minnesota prior is a normal distribution on every one of these entries, treated as
# independent:
#
# $$
# \beta_{ij}^{(l)} \sim \mathcal{N}\!\left( m_{ij}^{(l)},\; \big(s_{ij}^{(l)}\big)^2 \right).
# $$ (eq-minnesota-normal)
#
# All the content is in how $m$ and $s$ are chosen. The mean encodes a *guess*; the standard
# deviation encodes *how strongly you hold it*.
#
# ### The mean: each series is a random walk
#
# The guess is that every variable is a random walk and nothing predicts anything else:
#
# $$
# m_{ij}^{(l)} =
# \begin{cases}
# 1 & \text{if } i = j \text{ and } l = 1, \\
# 0 & \text{otherwise.}
# \end{cases}
# $$ (eq-minnesota-mean)
#
# So $A_1$ has ones on its diagonal, and every other coefficient in the model is centred at
# zero. Two things motivate this. Macroeconomic levels — output, prices, employment — really
# are close to unit-root processes, and a random walk is a famously hard forecast to beat at
# short horizons. And it is a *safe* default: shrinking toward it removes cross-variable
# dynamics rather than inventing them, so the prior can only cost you predictability you had
# real evidence for.
#
# :::{admonition} Working with growth rates or anomalies?
# :class: note
# The random-walk mean is a statement about *levels*. If you have already differenced,
# de-meaned, or standardised your series, a prior mean of one on the own first lag is too
# persistent — the honest centre is closer to zero. You can still use `MinnesotaPrior`, but
# the shrinkage target is then working against you rather than for you. Write a custom prior
# with a zero mean instead (see [Writing a Custom Prior](../how-to/custom-priors.md)); it is a
# ten-line class.
# :::
#
# ### The standard deviation: three knobs
#
# The prior mean above is deliberately naive. What stops it from dominating the data is the
# standard deviation, and this is where the tuning happens:
#
# $$
# s_{ij}^{(l)} = \underbrace{\lambda}_{\text{tightness}} \times
# \underbrace{d(l)}_{\text{lag decay}} \times
# \underbrace{\begin{cases} 1 & i = j \\ \kappa & i \neq j \end{cases}}_{\text{cross shrinkage}},
# \qquad
# d(l) = \begin{cases} 1/l & \texttt{"harmonic"} \\ 1/l^{2} & \texttt{"geometric"}. \end{cases}
# $$ (eq-minnesota-sd)
#
# Read the three factors as three separate beliefs:
#
# | Factor | Argument | Default | The belief it encodes |
# |--------|----------|---------|-----------------------|
# | $\lambda$ | `tightness` | `0.1` | How far *any* coefficient may stray from its prior mean. $\lambda \to 0$ freezes the model at the random walk; $\lambda \to \infty$ recovers OLS. |
# | $d(l)$ | `decay` | `"harmonic"` | Distant lags matter less than recent ones, so they get shrunk harder. |
# | $\kappa$ | `cross_shrinkage` | `0.5` | A variable's own history is more informative about it than other variables' histories. $\kappa = 0$ forbids cross-variable dynamics entirely; $\kappa = 1$ treats own and cross lags alike. |
#
# That is the whole prior. `MinnesotaPrior.build_priors` returns exactly
# {eq}`eq-minnesota-mean` and {eq}`eq-minnesota-sd` as two arrays, `B_mu` and `B_sigma`,
# which `VAR.fit` hands straight to PyMC as the mean and standard deviation of a normal
# prior on `B`.

# %%
prior = MinnesotaPrior()  # tightness=0.1, decay="harmonic", cross_shrinkage=0.5
params = prior.build_priors(n_vars=3, n_lags=4)
{k: v.shape for k, v in params.items()}

# %% [markdown]
# ## Seeing the prior
#
# Those two $3 \times 12$ arrays *are* the prior, so plotting them is the most direct way to
# understand it. Rows are equations (which variable is being explained); columns run over the
# 12 regressors in lag-major order — all three variables at lag 1, then all three at lag 2,
# and so on.

# %% mystnb={"figure": {"caption": "The default Minnesota prior for a 3-variable VAR(4), shown as the two arrays Impulso actually passes to PyMC. Left: prior means — ones on the own first lag, zero everywhere else. Right: prior standard deviations — brightest on own recent lags, dark on distant cross lags.", "name": "minnesota-heatmaps"}} tags=["remove-input"]
names = ["gdp", "infl", "rate"]
col_labels = [f"{names[c % 3]}\nL{c // 3 + 1}" for c in range(12)]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, key, title, cmap in [
    (axes[0], "B_mu", r"Prior mean $m_{ij}^{(l)}$", "RdBu_r"),
    (axes[1], "B_sigma", r"Prior standard deviation $s_{ij}^{(l)}$", "magma"),
]:
    arr = params[key]
    vmax = np.abs(arr).max()
    im = ax.imshow(arr, cmap=cmap, vmin=-vmax if key == "B_mu" else 0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12), col_labels, fontsize=6)
    ax.set_yticks(range(3), names, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("equation")
    for i in range(3):
        for j in range(12):
            r, g, b, _ = im.cmap(im.norm(arr[i, j]))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j,
                i,
                f"{arr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if luminance < 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.025)
fig.tight_layout()

# %% [markdown]
# The left panel is the random walk: a one wherever a variable meets its own first lag,
# zero everywhere else. The right panel is where the tuning lives. The largest prior standard
# deviation anywhere in the model is 0.10, on the own first lags — a coefficient the data
# must fight for even in the most permissive corner of the prior. By lag 4 the own-lag
# standard deviation is 0.025 and the cross-lag standard deviation is 0.0125, which is
# effectively a hard zero.
#
# ### How fast the lags die out
#
# The `decay` argument controls the slope of that decline. Harmonic decay ($1/l$) is gentle
# enough to leave seasonal or long-cycle dynamics some room; geometric decay ($1/l^2$) all
# but deletes anything past the second lag.

# %% mystnb={"figure": {"caption": "Prior standard deviation by lag, read out of `build_priors` for an 8-lag model. Geometric decay reaches near-zero by lag 3; harmonic decay leaves distant lags an order of magnitude more room.", "name": "minnesota-decay"}} tags=["remove-input"]
lags = np.arange(1, 9)
fig, ax = plt.subplots(figsize=(6.5, 4))
for decay, style in [("harmonic", "-"), ("geometric", "--")]:
    sd = MinnesotaPrior(decay=decay).build_priors(n_vars=3, n_lags=8)["B_sigma"]
    own = sd[0, (lags - 1) * 3 + 0]  # equation 0, own variable, each lag
    cross = sd[0, (lags - 1) * 3 + 1]  # equation 0, another variable, each lag
    ax.plot(lags, own, style, marker="o", color="crimson", label=f"own lags, {decay}")
    ax.plot(lags, cross, style, marker="s", color="0.35", label=f"cross lags, {decay}")
ax.set_yscale("log")
ax.set_xlabel("lag $l$")
ax.set_ylabel(r"prior standard deviation $s_{ij}^{(l)}$ (log scale)")
ax.set_title(r"Lag decay at $\lambda = 0.1$, $\kappa = 0.5$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
# The constant vertical gap between the red and grey curves is `cross_shrinkage`: a factor of
# $\kappa = 0.5$ applied uniformly across lags. Setting `cross_shrinkage=0`
# collapses the grey curves to zero and turns the VAR into $n$ independent autoregressions —
# useful as a forecasting benchmark, useless for structural work, since a variable that cannot
# respond to another variable's lags has no dynamic transmission to identify.
#
# ### What the prior looks like as a distribution
#
# {numref}`minnesota-densities` draws the same information as densities, which is how the
# sampler sees it. Each curve is the prior on a single coefficient before any data arrives.

# %% mystnb={"figure": {"caption": "Prior densities on two representative coefficients at three tightness settings. Left: the own first-lag coefficient, centred on the random walk. Right: a cross-variable first-lag coefficient, centred on zero and additionally shrunk by $\\kappa = 0.5$. The vertical scales are independent — compare the widths.", "name": "minnesota-densities"}} tags=["remove-input"]
grid = np.linspace(-1.0, 2.0, 800)


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
for lam, colour in zip([0.05, 0.1, 0.5], ["#7f1d1d", "crimson", "#f4a3a3"], strict=True):
    sd = MinnesotaPrior(tightness=lam).build_priors(n_vars=3, n_lags=4)["B_sigma"]
    axes[0].plot(grid, normal_pdf(grid, 1.0, sd[0, 0]), color=colour, label=rf"$\lambda = {lam}$")
    axes[1].plot(grid, normal_pdf(grid, 0.0, sd[0, 1]), color=colour, label=rf"$\lambda = {lam}$")
axes[0].axvline(1.0, color="0.5", linestyle=":", linewidth=1)
axes[1].axvline(0.0, color="0.5", linestyle=":", linewidth=1)
axes[0].set_title(r"own first lag $\beta_{ii}^{(1)}$ (prior mean 1)", fontsize=10)
axes[1].set_title(r"cross first lag $\beta_{ij}^{(1)}$ (prior mean 0)", fontsize=10)
for ax in axes:
    ax.set_xlabel("coefficient value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylabel("prior density")
# Panels use independent vertical scales; compare the widths, not the heights.
fig.tight_layout()

# %% [markdown]
# At $\lambda = 0.05$ the own first-lag coefficient is confined to roughly $[0.85, 1.15]$
# before the data speaks. At $\lambda = 0.5$ that same coefficient ranges over $[0, 2]$ and
# the prior has essentially stopped constraining anything.
#
# ## What the prior believes about the *world*
#
# Densities on individual coefficients are hard to price. What matters is the dynamics those
# coefficients imply, and you can look at that directly by drawing $B$ from the prior and
# simulating the system forward. This is a **prior predictive check**, and it costs nothing:
# no sampler, no data, just the prior arrays and some linear algebra.
#
# The summary statistic to watch is the **spectral radius**: the largest eigenvalue modulus of
# the VAR's companion matrix. Below one, shocks decay and the system is stationary. Above one,
# shocks compound and the system explodes. Each prior draw of $B$ gives us one of these
# numbers.

# %%
def companion_matrix(B: np.ndarray, n_vars: int, n_lags: int) -> np.ndarray:
    """Companion form of a VAR coefficient matrix stacked as [A_1 ... A_p]."""
    dim = n_vars * n_lags
    companion = np.zeros((dim, dim))
    companion[:n_vars] = B
    companion[n_vars:, : dim - n_vars] = np.eye(dim - n_vars)
    return companion


def spectral_radius(B: np.ndarray, n_vars: int, n_lags: int) -> float:
    """Largest eigenvalue modulus; < 1 means the VAR is stable."""
    return float(np.abs(np.linalg.eigvals(companion_matrix(B, n_vars, n_lags))).max())


def simulate(B: np.ndarray, n_vars: int, n_lags: int, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a path from a VAR with unit-variance shocks, starting from zero."""
    y = np.zeros((steps + n_lags, n_vars))
    for t in range(n_lags, steps + n_lags):
        x = np.concatenate([y[t - l] for l in range(1, n_lags + 1)])
        y[t] = B @ x + rng.standard_normal(n_vars)
    return y[n_lags:]


# %%
N_VARS, N_LAGS, N_DRAWS = 3, 4, 400
rng = np.random.default_rng(0)
lambdas = [0.05, 0.2, 1.0]

prior_draws = {}
for lam in lambdas:
    pp = MinnesotaPrior(tightness=lam).build_priors(n_vars=N_VARS, n_lags=N_LAGS)
    B_draws = rng.normal(pp["B_mu"], pp["B_sigma"], size=(N_DRAWS, *pp["B_mu"].shape))
    radii = np.array([spectral_radius(B, N_VARS, N_LAGS) for B in B_draws])
    prior_draws[lam] = (B_draws, radii)

pd.DataFrame(
    {
        "tightness": lambdas,
        "median radius": [np.median(prior_draws[lam][1]).round(2) for lam in lambdas],
        "90th pct radius": [np.quantile(prior_draws[lam][1], 0.9).round(2) for lam in lambdas],
        "share above 1.1": [(prior_draws[lam][1] > 1.1).mean().round(2) for lam in lambdas],
    }
).set_index("tightness")

# %% [markdown]
# Notice where the mass sits. At every tightness the *median* radius is at or just above one,
# because the prior mean {eq}`eq-minnesota-mean` is a random walk and a random walk has
# spectral radius exactly one. The Minnesota prior is deliberately parked on the boundary of
# stationarity — it is not a stationarity prior, and it never claims to be.
#
# Nor is a bare stability count the right diagnostic here: the largest of several near-unit
# eigenvalues is biased upward, so most draws come out technically explosive even at tiny
# $\lambda$. What matters is *by how much*. At $\lambda = 0.05$ only about an eighth of draws
# exceed a radius of 1.10 — these are near-unit-root systems, which is what macroeconomic
# levels look like. At $\lambda = 1.0$ the median draw has radius 1.9, meaning a shock roughly
# doubles every period. {numref}`minnesota-prior-predictive` shows what that difference means
# for simulated data.

# %% mystnb={"figure": {"caption": "Prior predictive paths for the first variable of a 3-variable VAR(4), 15 draws per panel, 120 periods each. Red paths come from draws with spectral radius below 1.1 (near-unit-root); grey paths from more explosive draws, most of which leave the frame within a few periods. Tight shrinkage implies plausible macroeconomic series; loose shrinkage implies almost nothing that resembles data.", "name": "minnesota-prior-predictive"}} tags=["remove-input"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
sim_rng = np.random.default_rng(7)
for ax, lam in zip(axes, lambdas, strict=True):
    B_draws, radii = prior_draws[lam]
    for k in range(15):
        path = simulate(B_draws[k], N_VARS, N_LAGS, steps=120, rng=sim_rng)
        calm = radii[k] < 1.1
        ax.plot(
            path[:, 0],
            linewidth=0.9,
            color="crimson" if calm else "0.55",
            alpha=0.85 if calm else 0.6,
        )
    ax.set_ylim(-40, 40)
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_title(rf"$\lambda = {lam}$ — {(radii > 1.1).mean():.0%} of draws above radius 1.1", fontsize=9)
    ax.set_xlabel("period")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("simulated $y_{1,t}$")
fig.tight_layout()

# %% [markdown]
# This is the argument for shrinkage stated in the units you care about. A loose prior is not
# "letting the data decide" — it is asserting, before seeing anything, that the economy
# probably detonates, and the likelihood then has to spend the sample arguing it back down.
# Tightening $\lambda$ concentrates prior mass on the wandering, highly persistent behaviour
# that real macroeconomic levels actually exhibit.
#
# ## What the prior does to the posterior
#
# Prior predictive plausibility is necessary, not sufficient. The tightest prior is always the
# most plausible-looking, and it is also useless — set $\lambda$ small enough and you get back
# your random walk regardless of what the data says. The real question is the bias–variance
# trade-off, so let us measure it.
#
# We simulate a 3-variable VAR(1) — persistent, as macroeconomic levels are, with a spectral
# radius of 0.89 — then deliberately fit a VAR(4), three times more lags than the truth. This
# is the realistic situation: you do not know $p$, so you pick generously and rely on the
# prior to switch off what is not there.

# %%
TRUE_A = np.array([
    [0.85, 0.05, -0.15],  # gdp: highly persistent, hurt by last period's rate
    [0.10, 0.80, 0.05],  # inflation: follows gdp, persistent
    [0.05, 0.20, 0.75],  # rate: leans against inflation, persistent
])
T_TRAIN, T_TEST = 100, 60

sim = np.random.default_rng(11)
y = np.zeros((T_TRAIN + T_TEST + 1, 3))
for t in range(1, len(y)):
    y[t] = TRUE_A @ y[t - 1] + sim.standard_normal(3) * 0.5
y_train, y_test_start = y[:T_TRAIN], T_TRAIN

index = pd.date_range("1990-01-01", periods=T_TRAIN, freq="QS")
train_data = VARData(endog=y_train, endog_names=["gdp", "infl", "rate"], index=index)

# True B padded out to VAR(4): lags 2-4 are genuinely zero.
FIT_LAGS = 4
B_true = np.zeros((3, 3 * FIT_LAGS))
B_true[:, :3] = TRUE_A

# %% [markdown]
# With only 100 training observations and 39 coefficients to place, this is exactly the
# regime the Minnesota prior was built for. We fit the same model at six tightness values
# spanning "almost a random walk" to "almost OLS".

# %%
TIGHTNESS_GRID = [0.02, 0.05, 0.1, 0.25, 0.5, 2.0]
sampler_kwargs = dict(draws=25, tune=25, chains=2, cores=1, random_seed=42) if ci else dict(
    draws=500, tune=500, chains=2, cores=1, random_seed=42
)

posterior_means = {}
for lam in TIGHTNESS_GRID:
    spec = VAR(lags=FIT_LAGS, prior=MinnesotaPrior(tightness=lam))
    fitted = spec.fit(train_data, sampler=NUTSSampler(**sampler_kwargs))
    posterior_means[lam] = (
        fitted.coefficients.mean(axis=(0, 1)),  # (n_vars, n_vars * n_lags)
        fitted.intercepts.mean(axis=(0, 1)),  # (n_vars,)
    )

# %% [markdown]
# Two things get measured. **Coefficient error** is the root mean squared distance between the
# posterior mean of $B$ and the truth — available only because we simulated the data.
# **One-step forecast error** is the honest out-of-sample version: using each fitted model's
# posterior mean, predict every one of the 60 held-out periods from its actual predecessors
# and compare against what happened.

# %%
def one_step_rmse(B: np.ndarray, c: np.ndarray) -> float:
    """RMSE of one-step-ahead predictions over the held-out block."""
    errors = []
    for t in range(y_test_start, len(y)):
        x = np.concatenate([y[t - l] for l in range(1, FIT_LAGS + 1)])
        errors.append(y[t] - (c + B @ x))
    return float(np.sqrt(np.mean(np.square(errors))))


scores = pd.DataFrame(
    {
        "tightness": TIGHTNESS_GRID,
        "coefficient RMSE": [
            np.sqrt(np.mean((posterior_means[lam][0] - B_true) ** 2)) for lam in TIGHTNESS_GRID
        ],
        "one-step forecast RMSE": [one_step_rmse(*posterior_means[lam]) for lam in TIGHTNESS_GRID],
    }
).set_index("tightness")
scores.round(4)

# %% mystnb={"figure": {"caption": "Left: both error measures against tightness, each normalised by its own minimum so the two curves share an axis. Right: posterior means of the four largest true coefficients as tightness varies, with the true values as dotted lines. Loose priors overshoot the truth; very tight priors pin every coefficient to the random walk.", "name": "minnesota-bias-variance"}} tags=["remove-input"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for col, colour, marker in [
    ("coefficient RMSE", "crimson", "o"),
    ("one-step forecast RMSE", "0.3", "s"),
]:
    axes[0].plot(scores.index, scores[col] / scores[col].min(), marker=marker, color=colour, label=col)
axes[0].axvline(0.1, color="0.6", linestyle=":", linewidth=1)
axes[0].text(0.105, axes[0].get_ylim()[1] * 0.95, "Impulso default", fontsize=7, color="0.4", va="top")
axes[0].set_xscale("log")
axes[0].set_xlabel(r"tightness $\lambda$ (log scale)")
axes[0].set_ylabel("error, relative to best on grid")
axes[0].set_title("Too tight and too loose both hurt", fontsize=10)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

tracked = [(0, 0, "gdp on gdp(-1)"), (1, 1, "infl on infl(-1)"), (2, 2, "rate on rate(-1)"), (2, 1, "rate on infl(-1)")]
for (i, j, label), colour in zip(tracked, ["crimson", "#1f4e79", "#2e7d32", "#b8860b"], strict=True):
    axes[1].plot(
        TIGHTNESS_GRID,
        [posterior_means[lam][0][i, j] for lam in TIGHTNESS_GRID],
        marker="o",
        color=colour,
        label=label,
    )
    axes[1].axhline(B_true[i, j], color=colour, linestyle=":", linewidth=1)
axes[1].set_xscale("log")
axes[1].set_xlabel(r"tightness $\lambda$ (log scale)")
axes[1].set_ylabel("posterior mean coefficient")
axes[1].set_title("Shrinkage pulls toward the random walk", fontsize=10)
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
# Both error curves are U-shaped, and that is the whole story in one picture. Move left and
# the prior overwhelms the data: at $\lambda = 0.02$ every own-lag coefficient is pinned near
# 0.98 and every cross-lag coefficient near zero, whatever the sample says, so the model is
# *biased*. Move right and the prior stops doing anything: the estimates chase noise across 39
# coefficients fitted on 100 observations, so the model is *high-variance*. Both ends are
# clearly worse than the middle.
#
# On this grid the minimum sits at $\lambda = 0.25$ rather than at the 0.1 default — but look
# at how flat the bottom of the curve is. Anything from 0.1 to 0.5 lands within two percent of
# the best forecast score available, whereas $\lambda = 0.02$ costs seven percent, and both
# extremes roughly double the coefficient error. The lesson is not that 0.25 is the right
# number. It is that the decision worth making is an *order of magnitude*, and that the flat
# region is wide enough that a sensible default will not embarrass you.
#
# Two honest caveats. The right panel shows shrinkage working *against* the truth for
# `rate on infl(-1)`: its prior mean is 0 but its true value is 0.2, so every step toward a
# tighter prior biases it down. That is the deal you are taking — you accept bias on the
# handful of coefficients that are real to buy variance reduction on the many that are not.
# And the forecast curve is much flatter than the coefficient curve — the loose end barely
# hurts it at all — because one-step-ahead forecasts are dominated by the own first lag, the
# one coefficient the prior is *least* wrong about. The 27 badly estimated lag-2-to-4
# coefficients hardly move a one-step forecast, which is why the coefficient panel is the
# sharper diagnostic. Shrinkage pays far more at longer horizons and in structural work, where
# those distant lags actually get used.
#
# ## Choosing the settings
#
# | Situation | Suggested starting point |
# |-----------|--------------------------|
# | Small system (2–4 variables), long sample | `tightness=0.2`, `cross_shrinkage=0.5` — you can afford to let the data speak |
# | Standard macro VAR (5–8 variables) | The defaults: `tightness=0.1`, `decay="harmonic"`, `cross_shrinkage=0.5` |
# | Large system (10+ variables) | `tightness=0.05` or lower — {cite:t}`giannoneLenzaPrimiceri2015` show optimal tightness falls as $n$ grows |
# | Many lags on monthly or weekly data | `decay="geometric"` to kill distant lags, or `decay="harmonic"` if you expect seasonality |
# | Forecasting benchmark | `cross_shrinkage=0.0` — reduces the VAR to independent AR($p$) models |
#
# Rather than trusting a table, fit two or three tightness values and compare. The prior
# predictive check above costs nothing and rules out the obviously bad end of the range; the
# held-out comparison settles the rest.
#
# :::{admonition} `MinnesotaPrior` does not rescale by variable
# :class: warning
# The classical Litterman formula multiplies the cross-variable standard deviation by
# $\sigma_i / \sigma_j$, the ratio of residual scales, so that a coefficient linking a
# variable measured in basis points to one measured in log points is shrunk sensibly.
# {eq}`eq-minnesota-sd` has no such term — `build_priors` only sees `n_vars` and `n_lags`, never
# your data. **Put your variables on comparable scales before fitting**, by standardising them
# or by expressing everything in percent. If you would rather the estimator handle scaling for
# you, `NIWPrior` computes per-variable AR(1) residual standard deviations internally; see
# [The Conjugate VAR](conjugate-var.py).
# :::
#
# :::{admonition} What the prior does not cover
# :class: note
# `MinnesotaPrior` governs the lag coefficients only. Intercepts get a fixed
# $\mathcal{N}(0, 1)$ prior, and the residual covariance $\Sigma$ is handled by the volatility
# process (`Constant` by default, `StochasticVolatility` optionally). Standardising your data
# also keeps that $\mathcal{N}(0, 1)$ intercept prior reasonable.
# :::
#
# ## Where to go next
#
# - **Fit one end to end** — the [Quickstart](quickstart.py) walks through a full model with
#   the default Minnesota prior.
# - **Let the data choose $\lambda$** — [The Conjugate VAR](conjugate-var.py) uses `NIWPrior`,
#   whose conjugate structure gives a closed-form marginal likelihood, so the tightness can be
#   selected rather than assumed ({cite:t}`giannoneLenzaPrimiceri2015`).
# - **Write your own** — [Writing a Custom Prior](../how-to/custom-priors.md) shows the
#   ten-line protocol any prior implements, which is how you would build the zero-mean or
#   scale-aware variants mentioned above.
#
# <section class="consulting-cta">
#     <p>We currently have some <strong>availability for consulting</strong> on how Bayesian modelling, vector autoregressions, and impulso can be integrated into your team's macroeconomic and financial forecasting work. If this sounds relevant, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation.</p>
# </section>
#
# ## References
#
# The works cited above are collected on the [project bibliography](../references.md) page.
