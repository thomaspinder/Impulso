# Conjugate NIW VAR is a sibling estimator, not a mode of `VAR`

We add a second estimation paradigm — a conjugate Normal-Inverse-Wishart BVAR with analytical marginal-likelihood hyperparameter selection (Giannone, Lenza & Primiceri 2015) — as a distinct `ConjugateVAR` class rather than a mode of the existing `VAR` or a new `Sampler`. It shares nothing with the PyMC/NUTS seams: there is no `pm.Model`, the conjugate prior is not expressible through `Prior.build_priors` (which returns independent-Normal `B_mu`/`B_sigma`), and volatility enters by rescaling the data (`ỹ_t = y_t / s_t`) with a Jacobian in the marginal likelihood rather than as a PyMC latent. `ConjugateVAR` returns the shared `FittedVAR`, so the entire downstream — identification, IRF/FEVD, forecast, `at=` queries — is reused unchanged.

## Considered options

- **Prior-type dispatch on `VAR`** (`VAR(prior=NIWPrior(...)).fit(...)`) — rejected: forks `fit` internally and makes `sampler`/`volatility` conditionally-meaningful, the "means different things depending on siblings" ambiguity the glossary warns against.
- **An analytical `Sampler`** — rejected: `Sampler.sample(model)` presupposes a `pm.Model` the conjugate engine never builds.
- **A unifying builder/dispatcher function** — rejected: a third config surface duplicating `VAR` and `ConjugateVAR`, and a god-function whose argument list hides which combinations are legal. With only two estimators (YAGNI), composition validity is better expressed as types.

## Consequences

- The `VolatilityProcess` protocol splits into a query surface (`is_time_varying`, `cholesky_at`, `cholesky_path`, `forecast_cholesky_path`) used by `FittedVAR`/`IdentifiedVAR`, and a `PyMCVolatilityProcess(VolatilityProcess)` sub-protocol adding `build_pymc_latent`, required only by `VAR.fit`. `Constant`/`StochasticVolatility` satisfy the PyMC sub-protocol; conjugate volatility adapters implement only the query surface.
- Cross-paradigm composition (e.g. an independent-Normal prior in `ConjugateVAR`, a conjugate volatility break in `VAR`) is prevented by typed fields plus Pydantic validators that raise domain-level errors — deliberately not a runtime dispatcher.
