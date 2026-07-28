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

**Dynamic multiplier**:
The response of each endogenous variable to a unit impulse in an *exogenous* regressor at horizons `0..h`: `Psi_h = Phi_h @ B_exog`. Shares the moving-average coefficients `Phi_h` with the IRF, but needs no identification scheme — exogenous regressors are exogenous by assumption — so it lives on `FittedVAR`, not `IdentifiedVAR`, and takes no `at`. The cumulative variant is the response to a permanent unit *step* rather than a one-off impulse. All dynamics come from the endogenous lag structure; `B_exog` enters contemporaneously and carries no lags of its own.
_Avoid_: "exogenous IRF" — IRF is reserved for responses to identified structural shocks.

**Historical decomposition (HD)**:
The attribution of each in-sample observation to a deterministic baseline (initial conditions, intercept, and any exogenous path) plus the *propagated* contribution of each structural shock: `c_{j,t} = P_t[:,j] ε_{j,t} + Σ_i A_i c_{j,t-i}`, with `y_t = baseline_t + Σ_j c_{j,t}` holding exactly. "Propagated" is load-bearing: a shock's contribution carries forward through the lag dynamics beyond its impact period.
_Avoid_: presenting the contemporaneous split of the one-step forecast error (`u_t = Σ_j P[:,j] ε_{j,t}`) as HD — that is a residual decomposition, and was the (incorrect) behaviour of the implementation prior to the scenario-analysis stack (2026-07).

**Historical counterfactual**:
The in-sample "what if": back out the realised structural shocks per posterior draw (`ε_t = P_t⁻¹ u_t`), overwrite a chosen shock's path over a window, and re-propagate the system to get the path history would have followed. Computed by `IdentifiedVAR.counterfactual(shocks=[ShockPath(...)])`. Realised shocks are edited, never re-drawn, so the posterior spread of a counterfactual reflects parameter and identification uncertainty only. `actual − counterfactual` for a shock zeroed over the full sample equals that shock's HD contribution.
_Avoid_: bare "counterfactual" for forecast-side operations (those are conditional forecasts or structural scenarios); "policy counterfactual" (rule replacement is a different, out-of-scope object — and the Lucas-critique caveat documented on the method applies even to fixed-path edits).

**Conditional forecast**:
An h-step forecast constrained so chosen endogenous variables follow pinned future paths, with *all* structural shocks free to adjust (Waggoner–Zha 1999). Identification-free: the observable-space answer is invariant to the identification scheme, so it lives on `FittedVAR.conditional_forecast()` — the same placement logic as the dynamic multiplier. Hard (point) conditions are the v1 default — pins hold pathwise on every draw; a second v1 mode, `path_uncertainty="unconditional"` (ADPRR's Ω_f = DD′), restricts the forecast *mean* only while bands keep their unconditional width. `NaN` entries in a pinned path mean "unconstrained at that step".
_Avoid_: "scenario" for an all-shocks-adjust conditional forecast — scenarios restrict *who* adjusts.

**Structural scenario**:
A conditional forecast with structural attribution (Antolín-Díaz, Petrella & Rubio-Ramírez 2021): pinned variable paths must be absorbed by a named `adjusting` set of shocks (non-adjusting shocks keep their unconditional draws), and/or future shock paths are prescribed directly. Computed by `IdentifiedVAR.structural_scenario()`.
_Avoid_: "conditional forecast" when an adjusting set is named — the restriction is the point.

**Condition vocabulary (`ShockPath`, `VariablePath`)**:
Frozen spec objects expressing scenario content. `ShockPath(shock, values, start, end)` sets a structural shock's path — values in one-standard-deviation units, scalar broadcast (`0.0` = "switch the shock off") or explicit array; `start`/`end` timestamps window it in-sample (counterfactual), while on the forecast axis values run from step 1 with `NaN` marking free entries. `VariablePath(variable, values)` pins a future endogenous path the same NaN-masked way. A scalar stays scalar until *application* time, where it broadcasts to the full resolved window — deliberately not the same as a length-1 array, which pins exactly one period. Each method accepts only the condition types legal for it — illegal combinations are unrepresentable rather than validated away.
_Avoid_: "Conditions object" / "Scenario object" for these primitives — a bundling `Scenario` container may arrive later for connector round-trips; the primitives are paths.

**Adjusting shocks**:
The subset of structural shocks permitted to absorb a structural scenario's conditions (`adjusting=[...]`). Existence is a per-draw rank condition (enough adjusting-shock dimensions to span the conditions); under-determination resolves through the conditional Gaussian, over-determination errors.
_Avoid_: "driving" / "offsetting" shocks in API surface (fine in prose, where ADPRR and Leeper–Zha use them).

**Plausibility statistic (q)**:
Per-draw squared Mahalanobis distance of a scenario's binding restriction values from their unconditional law, `q = c̄′(C C′)⁻¹ c̄ = ‖μ*‖²`, plus `‖v_S‖²` for prescribed shock paths — distributed `χ²_r` under the model when all shocks adjust (`r` = number of binding restrictions), reported with `r` and the tail probability `P(χ²_r ≥ q)`. The Leeper–Zha "modest interventions" check in the ADPRR lineage: large `q` (tiny tail probability) means the scenario demands incredible shocks and the model's answer should not be trusted. Stored per draw as a `plausibility` variable on `ConditionalForecastResult` and `ScenarioResult`, alongside the ADPRR-calibrated companion `q_cal ∈ [0.5, 1]` (`plausibility_calibrated`; McCulloch binomial matching with `z = q/2`) — finite only under the unconditional-variance mode, pegged at its ceiling of 1 under hard pins, floored at 0.5 with no conditions.
_Avoid_: calling it a Kullback–Leibler divergence under hard conditions — the conditional law is singular there and that KL is infinite; the KL form applies only to future *soft* conditioning. "Modesty statistic" stays prose-only.

**at**:
The time-index parameter on time-varying queries (`impulse_response(at=...)`, `fevd(at=...)`). Accepts an integer `t`, the literal `"last"` (most recent), `"all"` (full T-axis returned in the result), or `None` (default; resolves to `"last"` for stochastic volatility, ignored for constant volatility).

**Estimation paradigm (`VAR` vs `ConjugateVAR`)**:
Two ways to estimate the reduced-form VAR. `VAR` uses independent-Normal coefficient priors sampled by NUTS (the `Sampler` seam). `ConjugateVAR` uses a conjugate Normal-Inverse-Wishart prior with closed-form posteriors and marginal-likelihood hyperparameter selection (Giannone et al. 2015): it draws (β, Σ) analytically and samples only the hyperparameters by Metropolis. Both return a `FittedVAR`, so identification and forecasting are identical downstream.
_Reader-facing shorthand_: "the conjugate VAR" (`ConjugateVAR`) vs "the NUTS VAR" (`VAR`) — the contrast axis is the mode of inference (closed-form vs MCMC).
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
- A **FittedVAR** fitted with exogenous regressors computes **dynamic multipliers** on its own; no identification scheme is involved, because the driver is already exogenous.
- A **stochastic volatility** can plug into a **VAR** as its volatility process *or* be fitted standalone on a univariate series.
- A **FittedVAR** computes **conditional forecasts** on its own — all shocks adjust, so no identification scheme is involved (the dynamic-multiplier placement logic).
- An **IdentifiedVAR** computes **historical counterfactuals** and **structural scenarios** through the four-layer scenario engine (back out → constrain → solve → propagate); the propagate layer is shared with `forecast()` and the **historical decomposition**.
- The **condition vocabulary** is consumed by all three scenario methods; each method accepts only the condition types legal for it.
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
- "Counterfactual" in the wider literature spans shock-path edits (Impulso's meaning), policy-rule replacement (Sims–Zha style; out of scope), and Lucas-robust constructions (McKay–Wolf; out of scope). When comparing with external work, say which one is meant.
