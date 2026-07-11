# Tutorials

These tutorials walk you through Impulso's core workflow: fitting a Bayesian VAR, producing probabilistic forecasts, and running structural analysis. They assume familiarity with regression and autoregressive models but explain VAR-specific concepts as they arise.


| Tutorial | What you'll learn |
|----------|-------------------|
| [Fitting Your First Bayesian VAR](quickstart.py) | Data loading, lag selection, model fitting, posterior inspection |
| [Probabilistic Forecasts](forecasting.py) | Multi-step forecasts, credible intervals, fan charts |
| [Structural Shocks and Their Effects](structural-analysis.py) | Cholesky identification, impulse responses, FEVD, historical decomposition |
| [Monetary Policy Analysis](monetary-policy.py) | Policy reaction functions, scenario analysis |
| [Stochastic Volatility](stochastic-volatility.py) | Time-varying residual volatility via univariate SV: fit, interpret, and forecast |

Start with the **Quickstart** if you're new to Impulso. The Forecasting and Structural Analysis tutorials build on concepts introduced there.

```{toctree}
:hidden:
:maxdepth: 1

quickstart
forecasting
structural-analysis
monetary-policy
stochastic-volatility
```
