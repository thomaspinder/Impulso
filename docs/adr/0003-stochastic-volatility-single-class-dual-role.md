# `StochasticVolatility` is one class with two roles

`StochasticVolatility` serves both as a standalone univariate Bayesian SV model (via `.fit(SVData)`) and as a `VolatilityProcess` adapter plugged into `VAR(volatility=StochasticVolatility(...))`. There is one class, not two.

**Why**: Conceptually, "SV" is one model — same dynamics, same priors, same parameters whether the surrounding model has lags or n>1. Splitting into two classes (a `StochasticVolatility` standalone model and a `StochasticVolatilityProcess` for VAR consumption) creates a discoverability problem (which to import for which use case?) and duplicates substrate.

**Implementation note**: standalone `.fit(SVData)` keeps a separate code path from the `VAR(volatility=...).fit(VARData)` path. They share `SVDynamics` (priors, log-vol AR(1)) but the standalone observation equation `y = μ + exp(h/2)·ε` is simpler than the VAR-with-SV one and shouldn't be made load-bearing for the multivariate machinery. The shared substrate is dynamics + priors, not the surrounding pipeline.

**Constraint preserved**: the existing public API `from impulso.sv import StochasticVolatility` and `.fit(SVData)` keeps working unchanged through the deepening.
