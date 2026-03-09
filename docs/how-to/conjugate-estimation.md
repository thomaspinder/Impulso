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
