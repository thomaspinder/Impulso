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
A rule for recovering structural shocks from reduced-form covariance, implemented as adapters of the `IdentificationScheme` Protocol (`Cholesky`, `SignRestriction`, `LongRunRestriction`, `ProxySVAR`). The scheme is a pure function: it consumes a Cholesky factor `L` and produces a structural shock matrix `B = identify(L)`. It does not own time iteration.
_Avoid_: "identification strategy" (used colloquially; the Protocol is named "scheme").

**Long-run multiplier (C(1))**:
The sum of the moving-average coefficients of a stable reduced-form VAR, `C(1) = Σ_h Φ_h = (I − Σ_j A_j)^{-1}` — the total effect of a one-off reduced-form innovation on the *level* of each variable. Exists only when the companion spectral radius is below one; `I − Σ_j A_j` near-singular means it is numerically undefined, which is a distinct failure from divergence.

**Cumulative MA impact matrix (Θ(1))**:
The structural counterpart, `Θ(1) = C(1) P`: the total effect of each structural shock on each variable's level. Where `Cholesky` and `SignRestriction` constrain the impact matrix `Θ(0) = P`, `LongRunRestriction` constrains `Θ(1)` to be lower-triangular in a stated variable ordering, with positive diagonal (shock `j` raises variable `j`'s long-run level).
_Avoid_: "Blanchard-Quah identification" as an API name — the scheme is named for what it restricts, not for its first users (the prose citation is fine). "Permanent shock" for anything but the first column: only the shock restricted nowhere is unambiguously permanent.

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

**Observation error distribution**:
The seam that owns *which law the observation error follows* — the likelihood registered inside PyMC and the innovation law used on the forecast side. Concrete adapters of the `ErrorDistribution` Protocol: `Gaussian` (the default) and `StudentT`. It is the sibling of the volatility process, which owns how *big* the error is: the error distribution never touches Σ, and the volatility process never touches the shape of the tails. A `VAR` selects it with `error_dist=`.
_Avoid_: "likelihood" as the name of the seam (the likelihood is what the adapter *builds*, and the word also gets used for the whole model's `logp`); "error model", "noise distribution".

**Degrees of freedom (ν)**:
The tail-weight parameter of `StudentT`. Either a fixed float strictly greater than 2, or `"infer"` (the default) to estimate it. The bound at 2 is hardcoded, not configurable: below it the t has infinite variance and every variance-shaped consumer stops being defined. Under both parameterisations the posterior carries `nu` as a deterministic; under inference the free random variable is `nu_excess`, with `nu = 2 + nu_excess` (a *shift*, not a truncation — `Gamma(α, ·)` has zero density at the origin for `α > 1`, so the prior vanishes exactly where `ν/(ν−2)` diverges; `prior_alpha` is constrained to `> 1` to keep that property).
_Avoid_: "df" as a field name (collides with the DataFrame idiom); "tail parameter".

**Scale matrix (Ω)**:
What `L_t L_tᵀ` *is* under a heavy-tailed error distribution. It equals the innovation covariance under `Gaussian` errors, but under `StudentT` the covariance is `ν/(ν−2)·Ω`. `FittedVAR.sigma()` returns Ω in both cases — identification factorises the scale matrix — and `FittedVAR.innovation_covariance()` returns the covariance. FEVD, historical decompositions and zero-edit counterfactuals are *exactly* invariant to the distinction; impulse responses are in scale units. See ADR-0007.
_Avoid_: calling `sigma()` "the covariance" without qualification — true only under Gaussian errors.

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

**Predictive check (prior / posterior)**:
Data simulated from the model on the *estimation window*, to be compared against the data itself. `VAR.prior_predictive(data)` draws from the PyMC graph before conditioning on the likelihood; `FittedVAR.posterior_predictive()` replicates the sample from the posterior. Both are **one step ahead given the observed lags** — `y_t = c + B x_t^obs + L_t ε_t` — so they sit on the data's own time axis and feed `az.plot_ppc` directly. The posterior-predictive innovations come from the volatility seam's `L_t`, so their spread is time-varying under SV (ADR-0011).
_Avoid_: "forecast" for either — a forecast leaves the sample and iterates its own predictions; a predictive check never does.

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

**Model evidence (`ModelEvidence`)**:
The conjugate VAR's closed-form log marginal likelihood of the observed response block, `log p(y_{p+1:T} | y_{1:p}, hyperparameters, model)`, attached to `FittedVAR.evidence` by `ConjugateVAR.fit` (`None` on the NUTS path, which has no closed form). Because the value includes the volatility-rescaling Jacobian it is a density over the *observed* data, so a break model and a homoscedastic model on the same observations are directly comparable. It is conditional on the presample and on the hyperparameters it was evaluated at, so with `NIWPrior(select=True)` a ratio of two evidences is an empirical-Bayes Bayes factor. `compare_evidence(**fits)` checks comparability (same variable set, effective sample, window and response digest) and returns an `EvidenceComparison` of Bayes factors and posterior model probabilities.
_Avoid_: "log ML" in the API surface (spell out marginal likelihood); "model probability" for a raw Bayes factor — the probability requires prior model weights.

**Deterministic volatility break (`ConjugateVolatility`)**:
A volatility process whose per-period scale `s_t` follows a deterministic, hyperparameter-driven path with a known break date — not a stochastic process. Used only by `ConjugateVAR`: the scale enters as data rescaling `ỹ_t = y_t / s_t` with a Jacobian in the marginal likelihood, and its hyperparameters are estimated jointly with λ. `PandemicBreak` (three outbreak scales + geometric decay from March 2020) is the concrete case reproducing Lenza & Primiceri (2020).
_Avoid_: "stochastic volatility" — the break is deterministic given its hyperparameters.

**Stationarity pretest**:
A classical frequentist test run *before* a VAR is specified, to decide levels versus differences. `adf_test` (Augmented Dickey-Fuller; null = unit root) and `kpss_test` (Kwiatkowski-Phillips-Schmidt-Shin; null = stationarity) have opposite nulls, so both are reported and their `conclusion` columns are oriented per test. ADF decides on its p-value; **KPSS decides on statistic vs critical value**, because statsmodels clips the KPSS p-value to `[0.01, 0.10]` — which is also why `kpss_test` (and `integration_order`, which runs it) restricts `alpha` to the tabulated 0.10 / 0.05 / 0.025 / 0.01. Lives behind the optional `impulso[diagnostics]` extra (statsmodels). The pretests report and never decide: no mechanical "difference it" rule is applied, because unit-root tests have low power against persistent alternatives and are sensitive to deterministic terms and breaks.
_Avoid_: "stationarity check" / "unit-root check" — "pretest" carries the sequencing (it precedes specification) and the pretesting-bias caveat.

**Integration order (`d`, `d_max`)**:
The number of differences a series needs before ADF rejects a unit root, per variable in `IntegrationOrderResult.order`. `d_max = max(order.values())` is the system-wide maximum and is the augmentation term a Toda-Yamamoto procedure consumes — **the names `order` and `d_max` are the frozen contract for that consumer**. ADF drives the stopping rule; KPSS is recorded at every level as a `joint_status` 2×2 (`stationary` / `unit_root` / `conflicting` / `inconclusive`). Variables that are still integrated at `max_order`, or whose tests conflict where the search stopped, are listed in `inconclusive` — human judgement, not a silent verdict. **`d_max` understates whenever `inconclusive` is non-empty**: a variable still non-stationary at `max_order` is recorded at `max_order`, which is a floor rather than a finding, so a Toda-Yamamoto consumer must read `inconclusive` before trusting the augmentation.
_Avoid_: "order of differencing" for `d_max` — `d_max` is the maximum across the system, not any one series' `d`.

**Cointegration rank**:
The number of independent long-run relationships among integrated series, from the Johansen procedure (`johansen_test`). Both sequential tests are reported — `rank_trace` and `rank_max_eigen` — and `rank` is `rank_trace` by documented convention. Decisions rest on **critical values, not p-values** (MacKinnon-Haug-Michelis 1996 tables, as vendored by statsmodels), which is why `alpha` is restricted to 0.10 / 0.05 / 0.01. Rank ≥ 1 means differencing every series discards the long-run relationship. A vector error-correction model (VECM) is **out of scope**; the recommended response is a VAR in levels (the Sims–Stock–Watson stance; the Minnesota prior already shrinks toward random walks).
_Avoid_: "number of cointegrating vectors" in API surface (fine in prose); "cointegration test" without saying which statistic, since trace and max-eigen can disagree.

## Relationships

- A **VAR** carries one **prior**, one **volatility process**, and one **observation error distribution**.
- A **FittedVAR** plus an **identification scheme** produces an **IdentifiedVAR**.
- An **identification scheme** consumes an `L_t` (queried from the volatility process) and produces a structural shock matrix `B`.
- An **identification scheme** may additionally consume the posterior lag coefficients: `LongRunRestriction` needs them for the **long-run multiplier**, as `SignRestriction(restriction_horizon > 0)` and `ProxySVAR` already do. `L_t` alone is the minimum, not the maximum, of what a scheme may ask for.
- An **IdentifiedVAR** computes **IRFs**, FEVDs, and historical decompositions by asking the volatility process for `L_t` at the requested `at`, then applying the identification scheme.
- A **FittedVAR** fitted with exogenous regressors computes **dynamic multipliers** on its own; no identification scheme is involved, because the driver is already exogenous.
- A **stochastic volatility** can plug into a **VAR** as its volatility process *or* be fitted standalone on a univariate series.
- A **FittedVAR** computes **conditional forecasts** on its own — all shocks adjust, so no identification scheme is involved (the dynamic-multiplier placement logic).
- An **IdentifiedVAR** computes **historical counterfactuals** and **structural scenarios** through the four-layer scenario engine (back out → constrain → solve → propagate); the propagate layer is shared with `forecast()` and the **historical decomposition**.
- The **condition vocabulary** is consumed by all three scenario methods; each method accepts only the condition types legal for it.
- A **VAR** simulates its own **prior predictive** from the graph it would fit; a **FittedVAR** replicates the estimation sample as a **posterior predictive**, computed in NumPy from the posterior and the volatility seam so the conjugate estimator gets it for free (ADR-0011).
- A **VAR** is estimated by NUTS; a **ConjugateVAR** is estimated analytically with a Metropolis step on hyperparameters. Both produce a **FittedVAR**.
- A **stationarity pretest** consumes `VARData` (endogenous block only), a DataFrame, or a Series, and produces a result object — never a modified dataset and never a specification. It sits *beside* the pipeline, not in it: nothing downstream of `VAR.fit()` reads its output.
- **Integration order** feeds **cointegration rank**: the Johansen test is only meaningful for series that are individually integrated, and it is conditioned on a lag order (`k_ar_diff = p - 1`) that `select_lag_order` supplies.
- A **ConjugateVAR** carries an **NIW prior** and optionally a **deterministic volatility break**; a **VAR** carries a **MinnesotaPrior** and a **PyMC volatility process** (`PyMCVolatilityProcess`, the `build_pymc_latent` extension of the `VolatilityProcess` query surface). Each estimator's fields accept only its compatible components, enforced by types + validators rather than a builder.

## Example dialogue

> **User:** "Fit a 4-variable VAR with stochastic volatility, AR(1) log-vol."
> **Library:** `VAR(lags=4, volatility=StochasticVolatility(dynamics="ar1")).fit(VARData(...))`. The `volatility` parameter accepts a string shorthand (`"constant"`, `"sv"`) or any `VolatilityProcess` instance.
>
> **User:** "Is my prior sane before I spend an hour on NUTS?"
> **Library:** `VAR(lags=4).prior_predictive(data, draws=500)` then `az.plot_ppc(idata, group="prior")`. After fitting, `fitted.posterior_predictive()` returns the same-shaped in-sample replicates for the other end of the check.
>
> **User:** "Show me the IRF for shocks hitting in 2008Q3."
> **Library:** `identified.impulse_response(horizon=20, at=t_2008Q3)`. The pipeline queries `volatility.cholesky_at(t_2008Q3)` for `L`, the identification scheme rotates it into `B`, and the IRF is computed from `A_1..A_p` and `B`.
>
> **User:** "Just a univariate SV fit."
> **Library:** `StochasticVolatility(dynamics="ar1").fit(SVData(y))`. Same class, standalone code path.
>
> **User:** "What would food prices have done if the 2022 energy shock had never happened?"
> **Library:** `identified.counterfactual(shocks=[ShockPath(shock="energy", values=0.0, start="2022-01", end="2022-12")])`. Realised shocks are edited and re-propagated; `result.difference()` is the shock's effect.
>
> **User:** "Forecast inflation if the policy rate follows this path for two years."
> **Library:** `fitted.conditional_forecast(steps=8, conditions=[VariablePath(variable="rate", values=path)])`. All shocks adjust (Waggoner–Zha); no identification scheme needed. Add `path_uncertainty="unconditional"` to restrict the mean only and keep honest bands.
>
> **User:** "Same rate path, but make the *policy* shock do the work — and how believable is that?"
> **Library:** `identified.structural_scenario(steps=8, conditions=[VariablePath(variable="rate", values=path)], adjusting=["policy"])`. Non-adjusting shocks keep their unconditional draws; `result.plausibility()` reports `q` and the calibrated `q_cal`.
>
> **User:** "2020Q2 is wrecking my estimates. Can I stop dummying it out?"
> **Library:** `VAR(lags=4, error_dist="student_t").fit(VARData(...))`. The t likelihood downweights the observation automatically; the degrees of freedom come back in the posterior as `nu`. Pass `StudentT(nu=5.0)` to fix them instead — the robust choice on short samples.

## Conventions

**Discriminator field on adapters**: every concrete adapter class (`Constant`, `RandomWalk`, `AR1`, …) declares its registry key with `name: Literal["x"] = "x"`, *not* `name: ClassVar[str] = "x"`. The Literal form is the modern Pydantic v2 idiom: it makes `name` a real instance attribute, fires `ValidationError` on construction-time mismatch (`Constant(name="other")`) and on post-construction mutation (under `frozen=True`), and participates in `model_dump`/`model_validate` round-trips. Class-level access (`Constant.name`) does *not* work with this pattern — registries that need the key value should hardcode the literal string.

## Flagged ambiguities

- "SV" is both a noun (the model family — *stochastic volatility*) and an adjective ("an SV adapter"). The class `StochasticVolatility` is the canonical noun reference; the adjective form is fine in prose after the term has been spelled out.
- "Volatility" alone is ambiguous between *volatility process* (the seam) and *volatility paths* (the per-variable σ_i,t time series, useful for plotting). Be explicit when the distinction matters.
- "Minnesota prior" now denotes two distinct encodings: the independent-Normal `MinnesotaPrior` (NUTS path) and the conjugate `NIWPrior` (`ConjugateVAR`). Name the estimator when it matters.
- "Σ" now means the *scale* matrix under `StudentT` errors and the covariance under `Gaussian` errors. `sigma()` returns the same object either way; when the number has to be a variance, say so and use `innovation_covariance()`.
- "Counterfactual" in the wider literature spans shock-path edits (Impulso's meaning), policy-rule replacement (Sims–Zha style; out of scope), and Lucas-robust constructions (McKay–Wolf; out of scope). When comparing with external work, say which one is meant.
