# Tutorials

These tutorials walk you through Impulso's core workflow: fitting a Bayesian VAR, producing probabilistic forecasts, and running structural analysis. They assume familiarity with regression and autoregressive models but explain VAR-specific concepts as they arise.


| Tutorial | What you'll learn |
|----------|-------------------|
| [Fitting Your First Bayesian VAR](quickstart.py) | Data loading, lag selection, model fitting, posterior inspection |
| [Probabilistic Forecasts](forecasting.py) | Multi-step forecasts, credible intervals, fan charts |
| [Structural Shocks and Their Effects](structural-analysis.py) | Cholesky identification, impulse responses, FEVD, historical decomposition |
| [Monetary Policy Analysis](monetary-policy.py) | Policy reaction functions, scenario analysis |
| [Stochastic Volatility](stochastic-volatility.py) | Time-varying residual volatility via univariate SV: fit, interpret, and forecast |
| [Oil Supply News with an External Instrument](proxy-svar.py) | Proxy-SVAR identification, external instruments, Känzig (2021) replication |
| [The Conjugate VAR](conjugate-var.py) | Closed-form NIW estimation, data-selected shrinkage, conjugate-vs-NUTS comparison |
| [Estimating a VAR after March 2020](post-march-2020.py) | Conjugate NIW VAR, COVID volatility break, conditional forecasts, Lenza & Primiceri (2022) replication |

Start with the **Quickstart** if you're new to Impulso. The Forecasting and Structural Analysis tutorials build on concepts introduced there.

```{toctree}
:hidden:
:maxdepth: 1

quickstart
forecasting
structural-analysis
monetary-policy
stochastic-volatility
proxy-svar
conjugate-var
post-march-2020
```
