# What Is a Bayesian VAR?

A **Vector Autoregression (VAR)** models multiple time series as a system of equations where each variable depends on its own lags and the lags of all other variables in the system.

A **Bayesian VAR** adds prior distributions over the model parameters. This serves two purposes:

1. **Regularization** — VARs have many parameters (grows as $n^2 \times p$ where $n$ is the number of variables and $p$ is the lag order). Priors shrink estimates toward sensible values, reducing overfitting.
2. **Uncertainty quantification** — instead of point estimates, you get a full posterior distribution over coefficients, forecasts, and structural quantities.

## When to use a Bayesian VAR

- You have a moderate number of macroeconomic or financial time series (2--20 variables)
- You want probabilistic forecasts with credible intervals
- You want to study how shocks propagate through a system (impulse responses)
- You want to decompose forecast error variance or historical variation by shock source

## The Litterman pipeline

Litterman models this as a sequence of immutable types:

```
VARData -> VAR -> FittedVAR -> IdentifiedVAR
```

Each step adds information. You cannot skip steps or go backward.
