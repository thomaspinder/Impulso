# The convergence report is block-structured, and explosive draws never fail it

`convergence_report` is a VAR-specific diagnostic object rather than a wrapper over `arviz.summary`. It makes three commitments: every sampling metric is reported per *parameter block* with the offending coordinate named; dynamic stability is reported alongside convergence, computed from the companion matrix of every draw; and the two VAR-specific failure modes get named, machine-readable messages carrying remedies. A `"failed"` status is reserved for sampler pathology — R-hat above 1.05, effective sample size below 100, or a divergence rate at or above 1% — and is never triggered by explosive draws.

## Block taxonomy

Blocks are `coefficient`, `intercept`, `exog`, `covariance`, `volatility`, `identification`, `other`, reported in that order and omitted when empty. A single worst-R-hat over a whole VAR posterior hides which part of the model is failing: the lag coefficients, the covariance parameterisation, and the stochastic-volatility latents mix at very different rates, and a user staring at `max_rhat = 1.4` learns nothing about what to change.

Resolution is three-tiered, first match wins:

1. A static map of the posterior variable names Impulso's own estimators register (`B`, `intercept`, `B_exog`, `sigma_sd`, `tril_offdiag`, `L`, `Sigma`, `h`, `R_chol`, `R_chol_offdiag`, `structural_shock_matrix`, `P`).
2. The `v{i}_` prefix carried by every per-variable stochastic-volatility latent, which covers the whole family — present adapters and future ones — without enumerating parameter names that change whenever a dynamics adapter gains a field.
3. The optional `posterior_var_names()` capability on `VolatilityProcess`, letting an adapter claim the variables it registered. It is documented as an optional capability in the mould of `IdentificationScheme._samples_rotations`, read through `getattr`, and is deliberately *not* a protocol requirement — a third-party adapter that omits it still works.

Anything unresolved lands in `other`, never an error. A hand-built or third-party posterior still gets a full report, and the block's variable list makes plain what was not recognised. Refusing to diagnose a posterior because one variable is unfamiliar would be the wrong trade.

The `identification` block exists but is normally empty: the structural shock matrix is memoised lazily on `IdentifiedVAR` and never written back to the posterior. Excluding it is deliberate rather than incidental. Under `Cholesky` it is a deterministic function of draws already diagnosed in the covariance block, so its R-hat adds nothing; under `SignRestriction` a fresh rotation is drawn per call, so its R-hat would describe the rotation sampler rather than the posterior — actively misleading. The block is kept for legacy and hand-built posteriors that do carry the variable.

## Thresholds

| Metric | Warn | Fail | Source |
| --- | --- | --- | --- |
| R-hat | 1.01 | 1.05 | Vehtari et al. (2021); classic Gelman–Rubin |
| Effective sample size | 400 | 100 | 100 per chain at four chains |
| Divergence rate | any divergence | 1% | Betancourt (2017) |
| Explosive draw fraction | 5% | *never* | — |

R-hat and ESS comparisons are strict, so a metric sitting exactly on a threshold passes; the two rate thresholds (divergence rate, explosive fraction) trigger at the boundary. Thresholds live in a frozen `ConvergenceThresholds` model rather than as module constants so a caller can tighten them for a specific study and the report echoes back what it used.

## Why explosive draws never fail

Posterior mass on parameter draws whose companion matrix has spectral radius at or above 1 is reported prominently, with its consequences (impulse responses that diverge with the horizon, unbounded forecast fans, uninterpretable long-horizon FEVD shares, drifting historical-decomposition baselines) and its remedies. It is still only a warning, and at fractions below `explosive_warn` only informational.

The reason is that explosiveness is a property of the *model*, not of the sampler. Macroeconomic data in levels under a Minnesota prior centred on a random walk puts substantial mass near the unit circle by construction; that is the prior doing its job, and a fraction of draws crossing it is expected rather than pathological. Failing the report there would train users to ignore `"failed"`, which must keep meaning "these draws do not describe the posterior". Convergence and stability are different questions and are reported as such.

## Rejected alternatives

- **A thin wrapper over `az.summary`.** Rejected: it produces one row per coordinate with no block structure, no stability, and no VAR-specific interpretation — exactly the output users already have and cannot act on.
- **Living in `results.py` alongside the other result objects.** Rejected: `VARResultBase` contracts for `median`/`hdi`/`to_dataframe`/`plot` over a posterior-predictive DataArray, and a convergence report has no such array. Following `LagOrderResult`'s precedent would have forced a fake `plot` and a fake `median`. A dedicated `diagnostics.py` also gives the diagnostics family (issue #57's umbrella) somewhere to grow.
- **Per-block divergence attribution.** Rejected: a divergence is a property of a trajectory through the whole parameter space. Splitting the count by block would invent an attribution the sampler never made.
- **Warning through `warnings.warn`.** Rejected: the report object carries `status`, `messages`, and the per-block table, so the caller decides whether to print, raise, or ignore. A diagnostic that emits warnings cannot be used inside a loop over model specifications.
- **A `.plot()` method in v1.** Deferred: issue #57 owns diagnostic visuals, and the raw `(chain, draw)` radius array is exposed so a histogram or unit-circle scatter is a few lines away.
