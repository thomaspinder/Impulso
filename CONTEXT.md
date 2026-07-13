# Impulso

Bayesian Vector Autoregression (VAR) with structural identification and stochastic volatility extensions. This file defines the load-bearing terms used in the codebase, docs, and public API. Add new terms here as they're sharpened in design discussions; don't drift into synonyms.

## Language

**VAR (Vector Autoregression)**:
A multivariate linear time-series model where each variable is regressed on its own lags and the lags of every other variable. The class `VAR` represents the *reduced-form* specification (lags + prior + volatility process); structural meaning is layered on top via identification.
_Avoid_: "VAR model" (redundant), "autoregression" (the singular form is misleading for multivariate models).

**Reduced-form / Structural**:
The reduced-form VAR fits dynamics without economic interpretation; the structural VAR adds an identification scheme that maps reduced-form residuals to economically-meaningful shocks. In code, `FittedVAR` is reduced-form; `IdentifiedVAR` is structural.
_Avoid_: "raw" / "interpretable" — they obscure the technical meaning.

**Identification scheme**:
A rule for recovering structural shocks from reduced-form covariance, implemented as adapters of the `IdentificationScheme` Protocol (`Cholesky`, `SignRestriction`). The scheme is a pure function: it consumes a Cholesky factor `L` and produces a structural shock matrix `B = identify(L)`. It does not own time iteration.
_Avoid_: "identification strategy" (used colloquially; the Protocol is named "scheme").

**Volatility process**:
The seam that owns how the structural-shock covariance Σ_t is constructed (in PyMC), evolved over time, and queried. Concrete adapters of the `VolatilityProcess` Protocol: `Constant` (homoscedastic Σ; the default) and `StochasticVolatility` (time-varying). The volatility process owns its downstream computation — forecast covariance paths, time-`t` Cholesky query, per-variable volatility paths — so the pipeline never branches on adapter type.
_Avoid_: "variance model", "covariance specification" — these describe the *output*, not the process.

**Constant volatility**:
The default volatility process: a single Σ shared across all time points. Today's manual Cholesky parameterisation in `spec.py:_build_pymc_model` lifted into the `Constant` adapter.

**Stochastic volatility**:
Time-varying volatility where Σ_t evolves stochastically. Two flavours:
- **Clark-style** — per-variable log-volatility process (AR(1) or random walk on `h_i,t`) plus a constant correlation Cholesky `R`. The first concrete `StochasticVolatility` adapter.
- **Primiceri-style** — TVP correlations on top of per-variable log-vol. Planned future adapter; the seam is shaped to admit it without redesign.
The class `StochasticVolatility` serves a dual role: a standalone univariate model (via `.fit(SVData)`) and a `VolatilityProcess` plugged into VAR.

**L_t**:
Lower-triangular Cholesky factor of the time-`t` structural-shock covariance: `Σ_t = L_t @ L_t.T`. The primary output of the volatility process. For constant volatility, `L_t` is constant in `t`. For stochastic volatility, `L_t` varies. Identification operates on `L_t` directly — no redundant Cholesky decomposition.

**Impulse response function (IRF)**:
The dynamic response of each variable to a unit structural shock at horizons `0..h`. Computed from the reduced-form lag matrices `A_1, ..., A_p` and the structural shock matrix `B`. With a stochastic volatility process, IRFs depend on the shock period; the `at` parameter on `IdentifiedVAR.impulse_response` selects which time slice.

**at**:
The time-index parameter on time-varying queries (`impulse_response(at=...)`, `fevd(at=...)`). Accepts an integer `t`, the literal `"last"` (most recent), `"all"` (full T-axis returned in the result), or `None` (default; resolves to `"last"` for stochastic volatility, ignored for constant volatility).

**Estimation paradigm (`VAR` vs `ConjugateVAR`)**:
Two ways to estimate the reduced-form VAR. `VAR` uses independent-Normal coefficient priors sampled by NUTS (the `Sampler` seam). `ConjugateVAR` uses a conjugate Normal-Inverse-Wishart prior with closed-form posteriors and marginal-likelihood hyperparameter selection (Giannone et al. 2015): it draws (β, Σ) analytically and samples only the hyperparameters by Metropolis. Both return a `FittedVAR`, so identification and forecasting are identical downstream.
_Avoid_: "the Bayesian VAR" as if there were one estimator; name the paradigm.

**NIW prior (`NIWPrior`)**:
The conjugate Normal-Inverse-Wishart Minnesota prior consumed by `ConjugateVAR`, encoded via dummy observations. Distinct from `MinnesotaPrior`, which parameterises the independent-Normal prior for the NUTS path.
_Avoid_: conflating with `MinnesotaPrior` — different priors for different estimators.

**Minnesota tightness (λ)**:
The overall standard deviation of the Minnesota prior — the scalar controlling how hard all coefficients shrink toward the random-walk prior mean. `MinnesotaPrior.tightness` is this λ, held fixed; `ConjugateVAR` instead selects λ by maximising / sampling the marginal likelihood (hierarchical, à la Giannone et al. 2015).
_Avoid_: bare "shrinkage" (ambiguous with cross-variable shrinkage).

**Deterministic volatility break (`ConjugateVolatility`)**:
A volatility process whose per-period scale `s_t` follows a deterministic, hyperparameter-driven path with a known break date — not a stochastic process. Used only by `ConjugateVAR`: the scale enters as data rescaling `ỹ_t = y_t / s_t` with a Jacobian in the marginal likelihood, and its hyperparameters are estimated jointly with λ. `PandemicBreak` (three outbreak scales + geometric decay from March 2020) is the concrete case reproducing Lenza & Primiceri (2020).
_Avoid_: "stochastic volatility" — the break is deterministic given its hyperparameters.

## Relationships

- A **VAR** carries one **prior** and one **volatility process**.
- A **FittedVAR** plus an **identification scheme** produces an **IdentifiedVAR**.
- An **identification scheme** consumes an `L_t` (queried from the volatility process) and produces a structural shock matrix `B`.
- An **IdentifiedVAR** computes **IRFs**, FEVDs, and historical decompositions by asking the volatility process for `L_t` at the requested `at`, then applying the identification scheme.
- A **stochastic volatility** can plug into a **VAR** as its volatility process *or* be fitted standalone on a univariate series.
- A **VAR** is estimated by NUTS; a **ConjugateVAR** is estimated analytically with a Metropolis step on hyperparameters. Both produce a **FittedVAR**.
- A **ConjugateVAR** carries an **NIW prior** and optionally a **deterministic volatility break**; a **VAR** carries a **MinnesotaPrior** and a **PyMC volatility process** (`PyMCVolatilityProcess`, the `build_pymc_latent` extension of the `VolatilityProcess` query surface). Each estimator's fields accept only its compatible components, enforced by types + validators rather than a builder.

## Example dialogue

> **User:** "Fit a 4-variable VAR with stochastic volatility, AR(1) log-vol."
> **Library:** `VAR(lags=4, volatility=StochasticVolatility(dynamics="ar1")).fit(VARData(...))`. The `volatility` parameter accepts a string shorthand (`"constant"`, `"sv"`) or any `VolatilityProcess` instance.
>
> **User:** "Show me the IRF for shocks hitting in 2008Q3."
> **Library:** `identified.impulse_response(horizon=20, at=t_2008Q3)`. The pipeline queries `volatility.cholesky_at(t_2008Q3)` for `L`, the identification scheme rotates it into `B`, and the IRF is computed from `A_1..A_p` and `B`.
>
> **User:** "Just a univariate SV fit."
> **Library:** `StochasticVolatility(dynamics="ar1").fit(SVData(y))`. Same class, standalone code path.

## Conventions

**Discriminator field on adapters**: every concrete adapter class (`Constant`, `RandomWalk`, `AR1`, …) declares its registry key with `name: Literal["x"] = "x"`, *not* `name: ClassVar[str] = "x"`. The Literal form is the modern Pydantic v2 idiom: it makes `name` a real instance attribute, fires `ValidationError` on construction-time mismatch (`Constant(name="other")`) and on post-construction mutation (under `frozen=True`), and participates in `model_dump`/`model_validate` round-trips. Class-level access (`Constant.name`) does *not* work with this pattern — registries that need the key value should hardcode the literal string.

## Flagged ambiguities

- "SV" is both a noun (the model family — *stochastic volatility*) and an adjective ("an SV adapter"). The class `StochasticVolatility` is the canonical noun reference; the adjective form is fine in prose after the term has been spelled out.
- "Volatility" alone is ambiguous between *volatility process* (the seam) and *volatility paths* (the per-variable σ_i,t time series, useful for plotting). Be explicit when the distinction matters.
- "Minnesota prior" now denotes two distinct encodings: the independent-Normal `MinnesotaPrior` (NUTS path) and the conjugate `NIWPrior` (`ConjugateVAR`). Name the estimator when it matters.
