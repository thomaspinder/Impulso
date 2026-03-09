# Tier 1 Extensions Documentation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Document all 5 Tier 1 extensions across all Diataxis layers (reference, how-to, explanation, landing page).

**Architecture:** Create 6 new doc files (3 how-to, 1 explanation, 2 reference), extend 3 existing files (2 explanation, 1 landing page), and update mkdocs.yml nav. No code changes — documentation only. All docs are Markdown rendered by MkDocs Material with mkdocstrings, pymdownx.arithmatex (LaTeX), and admonitions.

**Tech Stack:** MkDocs Material, mkdocstrings, pymdownx.arithmatex (LaTeX via `$$`), pymdownx.superfences, admonitions (`!!! tip`, `!!! note`, `!!! warning`)

---

### Task 1: Reference pages and mkdocs.yml nav

**Files:**
- Create: `docs/reference/conjugate.md`
- Create: `docs/reference/conditions.md`
- Modify: `mkdocs.yml`

**Step 1: Create reference pages**

Create `docs/reference/conjugate.md`:

```markdown
# ConjugateVAR

::: impulso.conjugate
```

Create `docs/reference/conditions.md`:

```markdown
# Forecast Conditions

::: impulso.conditions
```

**Step 2: Update mkdocs.yml nav**

Add the new pages to the nav in `mkdocs.yml`. The full nav section should become:

```yaml
nav:
  - Home: index.md
  - Tutorials:
    - tutorials/index.md
    - tutorials/structural-analysis.ipynb
  - How-To Guides:
    - how-to/index.md
    - how-to/data-preparation.md
    - how-to/custom-priors.md
    - how-to/lag-selection.md
    - how-to/sign-restrictions.md
    - how-to/conjugate-estimation.md
    - how-to/long-run-restrictions.md
    - how-to/conditional-forecasting.md
  - Explanation:
    - explanation/index.md
    - explanation/bayesian-var.md
    - explanation/minnesota-prior.md
    - explanation/identification.md
    - explanation/conditional-forecasting.md
  - Reference:
    - reference/index.md
    - reference/data.md
    - reference/spec.md
    - reference/conjugate.md
    - reference/conditions.md
    - reference/priors.md
    - reference/samplers.md
    - reference/fitted.md
    - reference/results.md
    - reference/protocols.md
    - reference/identified.md
    - reference/identification.md
    - reference/plotting.md
```

**Step 3: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds with no errors for the new reference pages.

**Step 4: Commit**

```bash
git add docs/reference/conjugate.md docs/reference/conditions.md mkdocs.yml
git commit -m "docs: add reference pages for conjugate and conditions modules"
```

---

### Task 2: Update landing page

**Files:**
- Modify: `docs/index.md`

**Step 1: Update the landing page**

Replace the entire content of `docs/index.md` with the following. The key changes are:
1. Hero example shows both ConjugateVAR (fast) and VAR+NUTS (flexible) paths
2. Features list updated with new capabilities

```markdown
# Impulso

**Bayesian Vector Autoregression in Python.**

=== "Fast (Conjugate)"

    ```python
    import pandas as pd
    from impulso import ConjugateVAR, VARData

    # Load data
    df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
    data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

    # Estimate with data-driven prior selection (no MCMC needed)
    fitted = ConjugateVAR(lags=4, prior="minnesota_optimized").fit(data)

    # Forecast
    forecast = fitted.forecast(steps=8)
    forecast.median()  # point forecasts
    forecast.hdi()     # credible intervals
    ```

=== "Flexible (NUTS)"

    ```python
    import pandas as pd
    from impulso import VAR, VARData
    from impulso.identification import Cholesky

    # Load data
    df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
    data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

    # Estimate with NUTS (supports arbitrary priors)
    fitted = VAR(lags="bic", prior="minnesota").fit(data)

    # Structural analysis
    identified = fitted.set_identification_strategy(
        Cholesky(ordering=["gdp", "inflation", "rate"])
    )
    irf = identified.impulse_response(horizon=20)
    irf.plot()
    ```

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable pipeline** — `VARData` -> `VAR` -> `FittedVAR` -> `IdentifiedVAR`, each stage frozen after creation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains
- **Two estimation paths** — conjugate NIW sampling (instant, no MCMC) or full NUTS via PyMC (flexible, supports custom priors)
- **Data-driven prior selection** — automatic Minnesota hyperparameter optimisation via marginal likelihood (Giannone, Lenza & Primiceri, 2015)
- **Dummy observation priors** — encode persistence and unit root beliefs via sum-of-coefficients and single-unit-root priors
- **Minnesota prior** — smart defaults with tunable hyperparameters for shrinkage
- **Automatic lag selection** — AIC, BIC, and Hannan-Quinn criteria
- **Probabilistic forecasts** — posterior median, HDI credible intervals, tidy DataFrames
- **Conditional forecasting** — constrain future variable paths or structural shock paths for policy analysis
- **Structural identification** — Cholesky, sign restriction, and Blanchard-Quah long-run schemes
- **Built-in plotting** — IRF, FEVD, forecast, and historical decomposition plots

## Installation

```bash
pip install impulso
```

## Learn more

- [Fast estimation with ConjugateVAR](how-to/conjugate-estimation.md) — conjugate sampling, data-driven priors, dummy observations
- [Conditional forecasting](how-to/conditional-forecasting.md) — constrain future paths for policy analysis
- [Long-run restrictions](how-to/long-run-restrictions.md) — Blanchard-Quah structural identification
- [API Reference](reference/index.md) — complete module documentation
```

**Step 2: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds. Landing page renders with tabbed code examples.

**Step 3: Commit**

```bash
git add docs/index.md
git commit -m "docs: update landing page with new features and conjugate example"
```

---

### Task 3: How-to guide — Fast Estimation with ConjugateVAR

**Files:**
- Create: `docs/how-to/conjugate-estimation.md`

**Step 1: Write the how-to guide**

Create `docs/how-to/conjugate-estimation.md` with the following content:

````markdown
# Fast Estimation with ConjugateVAR

Standard Bayesian VAR estimation uses Markov chain Monte Carlo (MCMC) — typically the NUTS sampler — to explore the posterior distribution of model parameters. This is flexible but slow: a moderate-sized model might take minutes to sample, and you need to worry about convergence diagnostics, burn-in, and chain autocorrelation.

When your prior is Minnesota-type, none of this is necessary. The Minnesota prior combined with a Normal-Inverse-Wishart (NIW) likelihood gives a **conjugate** posterior — meaning the posterior has the same functional form as the prior. The posterior parameters can be computed in closed form, and draws are independent and identically distributed. No iteration, no burn-in, no convergence worries.

`ConjugateVAR` implements this direct sampling approach.

## Basic usage

```python
from impulso import ConjugateVAR, VARData

# Prepare your data as usual
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

# Estimate — this returns a FittedVAR, just like VAR(...).fit()
model = ConjugateVAR(lags=4, prior="minnesota")
fitted = model.fit(data)
```

The resulting `FittedVAR` is identical to what you'd get from `VAR(...).fit(data, sampler)` — you can call `.forecast()`, `.set_identification_strategy()`, and everything else the same way. The only difference is how the posterior draws were obtained.

!!! tip "When to use ConjugateVAR vs VAR"
    Use `ConjugateVAR` when you're happy with a Minnesota-type prior and want speed. Use `VAR` with a NUTS sampler when you need custom priors, non-standard likelihoods, or stochastic volatility — things that break conjugacy.

## Forecasting

Since `ConjugateVAR.fit()` returns a standard `FittedVAR`, forecasting works exactly as before:

```python
forecast = fitted.forecast(steps=8)

# Posterior median forecast
forecast.median()

# 89% highest density interval
forecast.hdi(prob=0.89)

# Plot fan chart
forecast.plot()
```

The forecasts are probabilistic — each of the 2000 posterior draws (by default) produces a different forecast path. The median and HDI summarise this distribution.

## Data-driven prior selection

The Minnesota prior has hyperparameters — `tightness`, `cross_shrinkage`, and `decay` — that control how aggressively the posterior is pulled toward the prior mean. Choosing these by hand is common but somewhat arbitrary.

**Giannone, Lenza & Primiceri (2015)** proposed a principled alternative: choose hyperparameters by maximising the **marginal likelihood** — the probability of the observed data given the hyperparameters, after integrating out all model parameters. This automatically balances fit and parsimony: too-tight priors underfit, too-loose priors overfit, and the marginal likelihood finds the sweet spot.

Because the model is conjugate, the marginal likelihood has a closed-form expression — no additional sampling is needed.

### One-step approach

The simplest way is the `"minnesota_optimized"` shorthand, which optimises the prior and fits the model in one call:

```python
fitted = ConjugateVAR(lags=4, prior="minnesota_optimized").fit(data)
```

This is equivalent to calling `optimize_prior()` followed by `fit()` with the optimised prior.

### Two-step approach

If you want to inspect or modify the optimised prior before fitting:

```python
model = ConjugateVAR(lags=4)

# Step 1: Find optimal hyperparameters
optimal_prior = model.optimize_prior(data)
print(optimal_prior)
# MinnesotaPrior(tightness=0.073, cross_shrinkage=0.42, decay='harmonic')

# Step 2: Fit with the optimised prior
fitted = ConjugateVAR(lags=4, prior=optimal_prior).fit(data)
```

The optimiser adjusts `tightness` and `cross_shrinkage` while preserving the `decay` setting from the starting prior.

!!! note "What the marginal likelihood measures"
    The marginal likelihood answers: "how well does this combination of hyperparameters predict the observed data, averaging over all possible parameter values?" A higher value means the prior is better calibrated to the data. It naturally penalises both underfitting (prior too tight, can't match the data) and overfitting (prior too loose, wastes probability mass on implausible parameter values).

### Comparing models

You can also use the marginal likelihood to compare models with different lag orders:

```python
for p in [1, 2, 3, 4]:
    ml = ConjugateVAR(lags=p).marginal_likelihood(data)
    print(f"Lags={p}: log ML = {ml:.1f}")
```

Higher log marginal likelihood indicates better fit after accounting for complexity.

## Dummy observation priors

Before fitting, you can augment your data with **dummy observations** that encode beliefs about persistence and unit roots. This is an elegant trick from the VAR literature (Doan, Litterman & Sims, 1984; Sims, 1993): rather than modifying the prior directly, you append synthetic data rows that push the posterior in the desired direction.

### Sum-of-coefficients prior (mu)

The sum-of-coefficients prior encodes the belief that **if all variables have been at their initial sample values forever, they should stay there**. This is a form of persistence belief — it discourages the model from predicting rapid mean-reversion that isn't supported by the data.

The hyperparameter `mu` controls the strength: larger values mean a weaker prior (less influence on the posterior).

```python
# Augment data with sum-of-coefficients dummy observations
data_augmented = data.with_dummy_observations(n_lags=4, mu=1.0)
```

### Single-unit-root prior (delta)

The single-unit-root prior encodes the belief that **each variable follows a random walk** — unit root behaviour. This is related to cointegration beliefs and helps prevent spurious cointegrating relationships.

The hyperparameter `delta` controls the strength: larger values mean a weaker prior.

```python
# Both priors together
data_augmented = data.with_dummy_observations(n_lags=4, mu=1.0, delta=1.0)
```

### Combining with GLP optimisation

Dummy observations work seamlessly with conjugate estimation and prior optimisation:

```python
# Augment data, then optimise and fit
data_augmented = data.with_dummy_observations(n_lags=4, mu=1.0, delta=1.0)
fitted = ConjugateVAR(lags=4, prior="minnesota_optimized").fit(data_augmented)
```

!!! warning "Choosing mu and delta"
    Start with `mu=1.0` and `delta=1.0` (moderate strength). Values below 0.5 impose strong beliefs; values above 5.0 have little effect. If you're unsure, the GLP marginal likelihood optimisation can help guide the choice — compare marginal likelihoods across different `mu`/`delta` combinations.

## Complete workflow

Putting it all together — dummy observations, optimised prior, estimation, and forecasting:

```python
from impulso import ConjugateVAR, VARData
from impulso.identification import Cholesky

# Prepare data
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

# Augment with dummy observation priors
data = data.with_dummy_observations(n_lags=4, mu=1.0, delta=1.0)

# Estimate with data-driven prior selection
fitted = ConjugateVAR(lags=4, prior="minnesota_optimized").fit(data)

# Forecast
forecast = fitted.forecast(steps=8)
forecast.plot()

# Structural analysis (works identically to VAR+NUTS path)
identified = fitted.set_identification_strategy(
    Cholesky(ordering=["gdp", "inflation", "rate"])
)
irf = identified.impulse_response(horizon=20)
irf.plot()
```
````

**Step 2: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docs/how-to/conjugate-estimation.md
git commit -m "docs: add how-to guide for conjugate estimation"
```

---

### Task 4: How-to guide — Long-Run Restrictions

**Files:**
- Create: `docs/how-to/long-run-restrictions.md`

**Step 1: Write the how-to guide**

Create `docs/how-to/long-run-restrictions.md` with the following content:

````markdown
# Long-Run Restrictions

Cholesky identification assumes a **contemporaneous** causal ordering: the first variable isn't affected by any other variable within the same period, the second is affected only by the first, and so on. This is a strong assumption that may not match your economic theory.

Sometimes theory says nothing about contemporaneous effects but makes clear predictions about **long-run** effects. For example, in the Blanchard-Quah (1989) framework:

- **Supply shocks** can have permanent effects on both output and prices
- **Demand shocks** have no permanent effect on output (only transitory)

Long-run restrictions implement this idea. Instead of applying Cholesky to the contemporaneous impact matrix, we apply it to the **long-run cumulative impact matrix** — forcing some shocks to have zero permanent effect on certain variables.

## Defining the scheme

The ordering determines which shocks can have permanent effects:

```python
from impulso.identification import LongRunRestriction

scheme = LongRunRestriction(ordering=["output", "prices"])
```

This means:
- The **first shock** (associated with "output") can have permanent effects on both output and prices
- The **second shock** (associated with "prices") has **no permanent effect on output** — only on prices

The ordering encodes your identifying assumption about long-run neutrality.

!!! tip "Reading the ordering"
    Think of it as: "shocks later in the ordering cannot permanently affect variables earlier in the ordering." The last shock has the most restrictions; the first shock is unrestricted.

## Applying to a fitted model

The workflow is identical to Cholesky — only the economics differ:

```python
from impulso import VAR, VARData
from impulso.identification import LongRunRestriction

# Fit reduced-form VAR
data = VARData.from_df(df, endog=["output", "prices"])
fitted = VAR(lags=4, prior="minnesota").fit(data)

# Apply long-run identification
scheme = LongRunRestriction(ordering=["output", "prices"])
identified = fitted.set_identification_strategy(scheme)
```

The resulting `IdentifiedVAR` supports all the same analysis methods — impulse responses, variance decomposition, and historical decomposition.

## Impulse response analysis

```python
irf = identified.impulse_response(horizon=40)
irf.plot()
```

When you examine the impulse responses, you should see the long-run restriction at work: the cumulative response of "output" to the second shock (the "prices" shock) converges to zero as the horizon increases. The first shock (the "output" shock) can have a permanent effect on both variables.

!!! note "Convergence to zero"
    The restriction forces the **cumulative** long-run effect to be zero, not the response at each individual horizon. The impulse response at horizon $h$ may be non-zero — it's only the sum across all horizons that vanishes.

## Variance decomposition and historical decomposition

These work identically to other identification schemes:

```python
# What fraction of forecast error variance is due to each shock?
fevd = identified.fevd(horizon=40)
fevd.plot()

# What drove the historical movements in each variable?
hd = identified.historical_decomposition()
hd.plot()
```

## When to use long-run restrictions

| Situation | Recommended scheme |
|-----------|-------------------|
| Theory specifies contemporaneous ordering | `Cholesky` |
| Theory specifies long-run neutrality | `LongRunRestriction` |
| Theory specifies signs but not ordering | `SignRestriction` |
| Multiple schemes plausible | Try several and compare IRFs |

!!! warning "Stationarity required"
    Long-run restrictions require the VAR to be stationary (all eigenvalues of the companion matrix inside the unit circle). If the model has a unit root, the long-run multiplier matrix is undefined. If some posterior draws are near-non-stationary, results may be numerically unstable.
````

**Step 2: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docs/how-to/long-run-restrictions.md
git commit -m "docs: add how-to guide for long-run restrictions"
```

---

### Task 5: How-to guide — Conditional Forecasting

**Files:**
- Create: `docs/how-to/conditional-forecasting.md`

**Step 1: Write the how-to guide**

Create `docs/how-to/conditional-forecasting.md` with the following content:

````markdown
# Conditional Forecasting

Standard forecasts let the model speak freely — given the historical data, where do the variables go next? But policy analysis often needs to answer a different question: **what happens if we assume a specific path for one variable?**

For example:
- "What happens to GDP and inflation if the central bank raises the policy rate by 25bp at each of the next 4 meetings?"
- "What is the inflation outlook if oil prices stay at \$80/barrel for the next year?"

Conditional forecasting answers these questions. It finds the **smallest set of shocks** consistent with the assumed path, then traces out the implications for all other variables. This implements the algorithm of Waggoner & Zha (1999).

## Defining conditions

A `ForecastCondition` specifies the variable, the periods (0-indexed forecast steps), and the target values:

```python
from impulso import ForecastCondition

# "The policy rate will be 5.25 at steps 0, 1, 2, and 3"
rate_path = ForecastCondition(
    variable="rate",
    periods=[0, 1, 2, 3],
    values=[5.25, 5.50, 5.75, 6.00],
)
```

You can specify multiple conditions on different variables:

```python
# Also condition on oil prices
oil_path = ForecastCondition(
    variable="oil",
    periods=[0, 1, 2, 3],
    values=[80.0, 80.0, 80.0, 80.0],
)
```

!!! note "Periods are 0-indexed"
    Period 0 is the first forecast step (one step ahead of the last observation). Period 3 is four steps ahead.

## Reduced-form conditional forecasts

On a `FittedVAR` (before identification), conditional forecasts use the Cholesky factor of the residual covariance to define the shock space:

```python
from impulso import VAR, VARData, ForecastCondition

data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
fitted = VAR(lags=4, prior="minnesota").fit(data)

# Condition on a rising rate path
rate_path = ForecastCondition(
    variable="rate",
    periods=[0, 1, 2, 3],
    values=[5.25, 5.50, 5.75, 6.00],
)

result = fitted.conditional_forecast(steps=8, conditions=[rate_path])
```

The result is a `ConditionalForecastResult` — a subclass of `ForecastResult` that also stores the conditions. You can inspect it the same way:

```python
# Posterior median conditional forecast
result.median()

# HDI credible intervals
result.hdi(prob=0.89)

# Plot
result.plot()
```

The constrained variable ("rate") will hit its target values exactly at the specified periods. The unconstrained variables ("gdp", "inflation") show the model's best prediction given those constraints — how the economy would evolve under the assumed rate path.

!!! tip "Interpreting the results"
    The conditional forecast answers: "What is the most likely path for all variables, given that `rate` follows the specified path?" The uncertainty bands for unconstrained variables reflect both parameter uncertainty (different posterior draws give different answers) and the uncertainty about which combination of shocks would produce the assumed path.

## Structural conditional forecasts

If you have an identified model, you can also condition on **structural shock paths**. This is useful for scenario analysis: "What happens if there are no supply shocks for the next 4 quarters?"

```python
from impulso.identification import Cholesky

identified = fitted.set_identification_strategy(
    Cholesky(ordering=["gdp", "inflation", "rate"])
)

# Condition on zero supply shocks
no_supply_shocks = ForecastCondition(
    variable="gdp",  # shock named after the first variable in ordering
    periods=[0, 1, 2, 3],
    values=[0.0, 0.0, 0.0, 0.0],
)

result = identified.conditional_forecast(
    steps=8,
    conditions=[rate_path],
    shock_conditions=[no_supply_shocks],
)
```

Here, `conditions` constrain observable variable paths (as before), and `shock_conditions` constrain structural shock paths. You can use either or both.

!!! warning "Shock naming"
    Shock names correspond to the variable names in the identification scheme's ordering. With `Cholesky(ordering=["gdp", "inflation", "rate"])`, the shocks are named "gdp", "inflation", and "rate".

## Degrees of freedom

Each condition uses up one degree of freedom per constrained period. The total number of constraints cannot exceed `steps * n_variables` (the total number of shock values to be determined). In practice, keeping constraints well below this limit gives more stable results — the system becomes increasingly sensitive as you approach the maximum.

!!! tip "Start simple"
    Begin with conditions on one variable at a few periods. Add more constraints incrementally and check that results remain sensible. Over-constraining can produce large, implausible shocks.
````

**Step 2: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docs/how-to/conditional-forecasting.md
git commit -m "docs: add how-to guide for conditional forecasting"
```

---

### Task 6: Extend explanation — Minnesota prior page

**Files:**
- Modify: `docs/explanation/minnesota-prior.md`

**Step 1: Extend the page**

Replace the entire content of `docs/explanation/minnesota-prior.md` with the following. The existing content is preserved at the top; three new sections are appended after "Usage in Impulso".

````markdown
# The Minnesota Prior

The **Minnesota prior** (Litterman, 1986) is the most widely used prior for Bayesian VARs. It encodes the belief that each variable follows a random walk, with coefficients on other variables' lags shrunk toward zero.

## Key hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tightness` | 0.1 | Overall shrinkage. Smaller = more shrinkage toward prior. |
| `decay` | `"harmonic"` | How fast coefficients shrink on longer lags. `"harmonic"`: $1/l$. `"geometric"`: $1/l^2$. |
| `cross_shrinkage` | 0.5 | Relative shrinkage on other variables' lags vs own lags. 0 = only own lags matter, 1 = equal treatment. |

## Intuition

The prior mean for the coefficient on a variable's own first lag is 1.0 (random walk). All other coefficients have prior mean 0.0. The prior standard deviation controls how far the posterior can move from these defaults.

## Usage in Impulso

```python
from impulso import VAR
from impulso.priors import MinnesotaPrior

# Use defaults
spec = VAR(lags=4, prior="minnesota")

# Customize hyperparameters
prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.3)
spec = VAR(lags=4, prior=prior)
```

## Conjugate estimation

When the Minnesota prior is paired with a Normal-Inverse-Wishart (NIW) likelihood, the posterior belongs to the same family — this is called **conjugacy**. The practical consequence is dramatic: instead of running an iterative MCMC sampler, we can compute the posterior parameters in closed form and draw from it directly.

The prior specifies:

$$B \mid \Sigma \sim \mathcal{MN}(B_0, \Sigma, V_0), \qquad \Sigma \sim \mathcal{IW}(S_0, \nu_0)$$

where $\mathcal{MN}$ is the matrix normal distribution and $\mathcal{IW}$ is the inverse Wishart. Given data matrices $Y$ (observations) and $X$ (lagged regressors), the posterior parameters are:

$$V_{\text{post}} = (V_0^{-1} + X^\top X)^{-1}$$

$$B_{\text{post}} = V_{\text{post}}(V_0^{-1} B_0 + X^\top Y)$$

$$\nu_{\text{post}} = \nu_0 + T$$

$$S_{\text{post}} = S_0 + Y^\top Y + B_0^\top V_0^{-1} B_0 - B_{\text{post}}^\top V_{\text{post}}^{-1} B_{\text{post}}$$

The posterior mean for $B$ is a precision-weighted average of the prior mean $B_0$ and the OLS estimate $(X^\top X)^{-1} X^\top Y$. When the prior is tight (small $V_0$), the posterior stays close to the random walk prior. When data is abundant (large $X^\top X$), the posterior approaches OLS. The posterior for $\Sigma$ is an inverse Wishart with updated scale and degrees of freedom.

Sampling is straightforward: draw $\Sigma \sim \mathcal{IW}(S_{\text{post}}, \nu_{\text{post}})$, then $B \mid \Sigma \sim \mathcal{MN}(B_{\text{post}}, \Sigma, V_{\text{post}})$. Each draw is independent — no burn-in, no autocorrelation, no convergence diagnostics. `ConjugateVAR` implements this approach.

## Data-driven prior selection

The Minnesota prior's hyperparameters — `tightness`, `cross_shrinkage`, and `decay` — are often chosen by convention or trial-and-error. Giannone, Lenza & Primiceri (2015) proposed an empirical Bayes approach: choose hyperparameters by maximising the **marginal likelihood**.

The marginal likelihood is the probability of the observed data $Y$ given hyperparameters $\lambda$, after integrating out all model parameters:

$$p(Y \mid \lambda) = \int p(Y \mid B, \Sigma) \, p(B, \Sigma \mid \lambda) \, dB \, d\Sigma$$

Thanks to conjugacy, this integral has a closed-form solution involving determinants and multivariate gamma functions. Optimising $\log p(Y \mid \lambda)$ over $\lambda = (\text{tightness}, \text{cross\_shrinkage})$ using standard numerical optimisation (L-BFGS-B) is fast and reliable.

The marginal likelihood naturally balances two forces:

- **Fit**: a loose prior (high tightness) lets the model fit the data closely
- **Parsimony**: a loose prior also spreads probability mass over implausible parameter values, reducing the marginal likelihood

The optimal hyperparameters sit at the sweet spot where the prior is just flexible enough to capture the data's patterns without wasting probability on noise.

This approach is sometimes called **GLP** after the authors' initials. In Impulso, use `ConjugateVAR.optimize_prior()` or the shorthand `prior="minnesota_optimized"`.

## Dummy observation priors

An elegant trick from the classical VAR literature encodes prior beliefs not by modifying the prior distribution directly, but by **appending synthetic observations** to the dataset. These "dummy observations" push the posterior in the desired direction while preserving conjugacy.

### Sum-of-coefficients prior

The sum-of-coefficients prior (Doan, Litterman & Sims, 1984) encodes the belief that if all variables have been at their initial sample values forever, they should persist at those values. Formally, it adds observations implying:

$$\sum_{j=1}^{p} A_j \approx I$$

where $A_j$ are the lag coefficient matrices. This discourages the model from predicting rapid mean-reversion when it isn't supported by the data. The hyperparameter $\mu$ controls the prior's strength: smaller $\mu$ imposes the belief more tightly.

### Single-unit-root prior

The single-unit-root prior (Sims, 1993) encodes the belief that each variable follows an independent random walk. It adds a single observation per variable that makes the model reluctant to introduce cointegrating relationships not strongly supported by the data. The hyperparameter $\delta$ controls its strength.

### Practical usage

In Impulso, dummy observations are added via `VARData.with_dummy_observations()`:

```python
data_augmented = data.with_dummy_observations(n_lags=4, mu=1.0, delta=1.0)
```

The augmented data can then be passed to any estimation method — `ConjugateVAR`, `VAR`, or used with `optimize_prior()`. Because the dummy observations are simply extra rows in the data matrices, they preserve conjugacy and integrate seamlessly with marginal likelihood optimisation.
````

**Step 2: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds. LaTeX equations render correctly.

**Step 3: Commit**

```bash
git add docs/explanation/minnesota-prior.md
git commit -m "docs: extend Minnesota prior explanation with conjugacy, GLP, and dummy obs theory"
```

---

### Task 7: Extend explanation — Identification page

**Files:**
- Modify: `docs/explanation/identification.md`

**Step 1: Extend the page**

Append a new section after the existing "Sign restrictions" section in `docs/explanation/identification.md`. The existing content is preserved; add the following at the end of the file:

```markdown

## Long-run restrictions (Blanchard-Quah)

Cholesky and sign restrictions both constrain the **contemporaneous** (impact) response to structural shocks. Long-run restrictions take a different approach: they constrain the **cumulative** response as the horizon goes to infinity.

The key object is the **long-run multiplier matrix**:

$$C(1) = (I - A_1 - A_2 - \cdots - A_p)^{-1}$$

This matrix captures the total cumulative effect of a one-time shock. If the VAR is stationary, all shocks are transitory and $C(1)$ is finite. The long-run impact of structural shocks is $C(1) P$, where $P$ is the structural impact matrix.

Blanchard & Quah (1989) proposed forcing $C(1) P$ to be **lower triangular**. This is achieved by applying the Cholesky decomposition not to the residual covariance $\Sigma$ (as in short-run Cholesky) but to the long-run covariance:

$$C(1) \, \Sigma \, C(1)^\top = L \, L^\top$$

The structural impact matrix is then $P = C(1)^{-1} L$.

The interpretation depends on the variable ordering:
- Shocks later in the ordering have **zero long-run cumulative effect** on variables earlier in the ordering
- The first shock is unrestricted in its long-run effects

This is exactly the same Cholesky math, applied to a different matrix. The economics, however, are very different: you're restricting permanent effects rather than contemporaneous effects. In the classic Blanchard-Quah example, ordering output before prices means the second shock (interpreted as a demand shock) has no permanent effect on output — only supply shocks do.
```

**Step 2: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docs/explanation/identification.md
git commit -m "docs: add long-run restrictions theory to identification explanation"
```

---

### Task 8: New explanation page — Conditional Forecasting

**Files:**
- Create: `docs/explanation/conditional-forecasting.md`

**Step 1: Write the explanation page**

Create `docs/explanation/conditional-forecasting.md`:

````markdown
# Conditional Forecasting

## The problem

A standard VAR forecast projects all variables forward simultaneously, with no external constraints. But many questions in macroeconomics and finance require **conditional** projections:

- Central banks publish forecasts conditioned on assumed interest rate paths
- Financial institutions stress-test portfolios under assumed macroeconomic scenarios
- Policy analysts ask "what if" questions about specific variable trajectories

Conditional forecasting provides the mathematical framework for these exercises.

## The idea

A VAR forecast can be decomposed into two parts:

$$y_{t+h} = \underbrace{y_{t+h}^{u}}_{\text{unconditional}} + \underbrace{\sum_{s=0}^{h} \Phi_{h-s} \, P \, \varepsilon_{s}}_{\text{shock-driven deviation}}$$

where $y^{u}_{t+h}$ is the unconditional forecast (no future shocks), $\Phi_j$ are the moving-average (MA) coefficient matrices, $P$ is the impact matrix (Cholesky factor of $\Sigma$ or a structural matrix), and $\varepsilon_s$ are the future structural shocks.

The unconditional forecast is deterministic given the posterior draw. The future shocks are unknown. Conditional forecasting amounts to **choosing the shock paths** that make the forecast satisfy the desired constraints.

## The Waggoner-Zha algorithm

Waggoner & Zha (1999) showed that this can be formulated as a linear system. Stack all future shocks into a vector $\varepsilon = [\varepsilon_0, \varepsilon_1, \ldots, \varepsilon_{H-1}]$ of length $H \times n$, where $H$ is the number of forecast steps and $n$ is the number of variables.

Each constraint (e.g., "variable $i$ equals value $v$ at period $h$") translates into a linear equation:

$$R \, \varepsilon = c$$

where $R$ is constructed from the MA coefficients and the impact matrix, and $c$ contains the differences between target values and unconditional forecasts.

The minimum-norm solution $\varepsilon^* = R^\top (R R^\top)^{-1} c$ gives the **smallest set of shocks** (in a least-squares sense) that satisfies all constraints. This is computed via `numpy.linalg.lstsq`.

The conditional forecast is then:

$$y_{t+h}^{c} = y_{t+h}^{u} + \sum_{s=0}^{h} \Phi_{h-s} \, P \, \varepsilon^*_s$$

## Structural extensions

When the model is identified (you have a structural impact matrix $P$ rather than just the Cholesky factor of $\Sigma$), you can also condition on **structural shock paths**. For example, "assume the supply shock is zero for the next 4 periods" translates into direct constraints on elements of $\varepsilon$.

Observable constraints and shock constraints can be combined in a single system. Observable constraints use the full MA representation (involving $\Phi$ and $P$), while shock constraints are simpler — they directly pin individual elements of $\varepsilon$.

## Bayesian uncertainty

The algorithm is applied independently to each posterior draw of $(B, \Sigma)$ or $(B, P)$. This means:

- The unconditional forecast differs across draws (parameter uncertainty)
- The MA coefficients differ across draws
- The shock paths satisfying the constraints differ across draws

The result is a full posterior distribution of conditional forecasts, from which you can compute medians, HDIs, and other summaries. The constrained variables will hit their targets exactly in every draw, but the unconstrained variables will show genuine posterior uncertainty about the conditional projection.
````

**Step 2: Verify docs build**

Run: `make docs-test`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docs/explanation/conditional-forecasting.md
git commit -m "docs: add conditional forecasting theory explanation"
```

---

### Task 9: Final verification

**Files:**
- None (verification only)

**Step 1: Full docs build**

Run: `make docs-test`
Expected: Build succeeds with no warnings about missing pages or broken links.

**Step 2: Verify all nav entries resolve**

Run: `uv run mkdocs build --strict 2>&1 | grep -i "warning\|error"` (if `--strict` is available; otherwise just `make docs-test` is sufficient).

**Step 3: Run tests to confirm nothing is broken**

Run: `uv run python -m pytest -m "not slow" -q`
Expected: 167 passed (docs changes should not affect tests).
