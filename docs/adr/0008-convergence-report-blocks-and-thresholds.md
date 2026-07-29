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
| E-BFMI | 0.3 | *never* | Betancourt (2016), arXiv:1604.00695 |
| Max-treedepth saturation rate | 1% | *never* | — |
| Explosive draw fraction | 5% | *never* | — |

R-hat, ESS and E-BFMI comparisons are strict, so a metric sitting exactly on a threshold passes; the three rate thresholds (divergence rate, treedepth saturation rate, explosive fraction) trigger at the boundary. Thresholds live in a frozen `ConvergenceThresholds` model rather than as module constants so a caller can tighten them for a specific study and the report echoes back what it used.

The treedepth threshold has no literature source, unlike the others. Stan and PyMC surface any saturation at all, but a handful of hits in a long run costs wall-clock time and nothing else, so reporting them would train users to skim past the message. One transition in a hundred is the point at which the sampler is spending real effort on trajectories it never gets to finish. Both backends record the flag under different names — `reached_max_treedepth` in PyMC, `maxdepth_reached` in nutpie — so this is the one statistic the report resolves through a name map rather than reading directly.

## Why explosive draws never fail

Posterior mass on parameter draws whose companion matrix has spectral radius at or above 1 is reported prominently, with its consequences (impulse responses that diverge with the horizon, unbounded forecast fans, uninterpretable long-horizon FEVD shares, drifting historical-decomposition baselines) and its remedies. It is still only a warning, and at fractions below `explosive_warn` only informational.

The reason is that explosiveness is a property of the *model*, not of the sampler. Macroeconomic data in levels under a Minnesota prior centred on a random walk puts substantial mass near the unit circle by construction; that is the prior doing its job, and a fraction of draws crossing it is expected rather than pathological. Failing the report there would train users to ignore `"failed"`, which must keep meaning "these draws do not describe the posterior". Convergence and stability are different questions and are reported as such.

## Why E-BFMI and max-treedepth never fail either

Both are statements about *efficiency*, not about wrongness. A saturated tree depth means NUTS stopped a trajectory before it turned back on itself, so the draws are more autocorrelated than they need to be; a low E-BFMI means momentum resampling is exploring the energy distribution slowly, so the tails are undersampled relative to the bulk. Neither says the retained draws come from the wrong distribution — unlike a divergence, which says the sampler could not follow the geometry at all, or an unmixed R-hat, which says the chains are not describing one distribution.

Both are also remediable by changing the sampler or the parameterisation without touching the model, so failing on them would block work that is merely slower than it should be. Both metrics are on the report regardless, so a caller who wants them fatal reads `ebfmi` and `treedepth_saturation_rate` and decides.

## Rejected alternatives

- **A thin wrapper over `az.summary`.** Rejected: it produces one row per coordinate with no block structure, no stability, and no VAR-specific interpretation — exactly the output users already have and cannot act on.
- **Living in `results.py` alongside the other result objects.** Rejected: `VARResultBase` contracts for `median`/`hdi`/`to_dataframe`/`plot` over a posterior-predictive DataArray, and a convergence report has no such array. Following `LagOrderResult`'s precedent would have forced a fake `plot` and a fake `median`. A dedicated `diagnostics.py` also gives the diagnostics family (issue #57's umbrella) somewhere to grow.
- **Per-block divergence attribution.** Rejected: a divergence is a property of a trajectory through the whole parameter space. Splitting the count by block would invent an attribution the sampler never made.
- **Warning through `warnings.warn`.** Rejected: the report object carries `status`, `messages`, and the per-block table, so the caller decides whether to print, raise, or ignore. A diagnostic that emits warnings cannot be used inside a loop over model specifications.
- **A `.plot()` method in v1.** Deferred, then added on `StabilitySummary` alone (issue #179): `report.stability.plot()` gives the spectral-radius histogram beside the companion-root scatter, matching how every other result object in the library is plotted. `ConvergenceReport` itself still has no `plot` — a block table of R-hat and ESS is a table, and issue #57 owns whatever diagnostic visuals go beyond stability.
- **Retaining every companion eigenvalue on `StabilitySummary`.** Rejected: the array grows as `draws × n_vars × n_lags` complex numbers — roughly 15 MB for 4000 draws of a 240×240 companion matrix — on an object whose every statistic is derived from the radii. The summary keeps a chain-pooled, deterministically strided subset of at most 200 draws, computed from the single eigendecomposition the radii already require, which is more points than the scatter panel can distinguish and a fixed ceiling regardless of posterior size.
