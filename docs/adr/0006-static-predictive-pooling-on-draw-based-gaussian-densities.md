# Static predictive pooling scores draw-based Gaussian densities on a held-out window

Combining several fitted VARs into one predictive distribution needs a score. Impulso scores each
candidate model's predictive density on an explicit **held-out window** and turns those scores into
**static weights** — one weight per model, fixed across horizons and across time. `pool_forecasts`
owns estimation (it calls `.forecast()` itself, so the forecasts it scores are provably
density-mode, from a single origin, on the models' own training samples); `PredictivePool.combine()`
owns application (frozen weights applied to *new* forecasts from full-sample refits). Weights come
from either **stacking** (default; the log-score of the pooled predictive, maximised over the
simplex) or **log-score/pseudo-BMA** weights (softmax of per-model total log scores).

## Considered options

- **Exact Rao-Blackwellised mixture score** — for each held-out point, average the per-draw Gaussian
  predictive densities implied by each posterior draw rather than moment-matching the draws to one
  Gaussian. Strictly better (it keeps the mixture's heavy tails) and deferred, not rejected: the
  `density=` keyword is the seam, and `density="mixture"` is where it lands. The moment-matched
  Gaussian was chosen first because it needs only the forecast draws — no re-entry into the
  volatility process — and therefore works identically for every estimator (`VAR`, `ConjugateVAR`)
  and every volatility process.
- **Per-variable (diagonal) densities only** — rejected as the default: it throws away the
  cross-variable correlation that is the entire point of a VAR. Retained as the `density="diagonal"`
  escape hatch for near-singular covariances and small draw counts.
- **Joint-path scoring** — score the whole `H`-step path as one `H·n`-dimensional Gaussian instead
  of summing per-horizon scores. Rejected: it needs `S > H·n` draws to be non-singular at realistic
  horizons, and the per-horizon sum is the standard density-forecast-evaluation object.
- **Rolling-origin scoring** — re-forecast from each held-out date. Rejected for v1: it multiplies
  cost by `H` and requires refitting to stay honest about information sets. One fixed origin keeps
  the contract provable from `FittedVAR.data.index[-1]`.
- **Dynamic (time-varying) weights** — Del Negro, Hasegawa & Schorfheide (2016). Out of scope; the
  static pool is the object this ADR fixes, and dynamic weights slot in behind the existing
  `method=` keyword when they arrive.
- **Pooling over `ForecastResult` objects supplied by the user** — rejected as the estimation entry
  point: a `ForecastResult` carries no origin or estimation metadata, so "these were produced from
  the same information set" is unenforceable. `pool_forecasts` takes `FittedVAR`s and forecasts them
  itself. `combine()` *does* take forecasts, because by then the weights are already fixed.

## Consequences

- The **score matrix** — `(H, M)` log predictive densities, one row per held-out date, one column
  per model — is the contract between scoring and weighting. Both solvers consume it and nothing
  else, which is why they are unit-testable against closed-form optima and why a future
  `density="mixture"` is a drop-in.
- Scores are *comparable across models*, not exact predictive densities: the Gaussian
  moment-matching approximates a genuinely heavier-tailed posterior-predictive mixture. The
  approximation degrades with few draws, stochastic volatility, and heavy tails.
- The pool **never refits**. It cannot, in general — the held-out window postdates every model's
  estimation sample by construction, and refitting on it would destroy the held-out property. Users
  who want full-sample models for genuine forecasting refit themselves and pass those forecasts to
  `combine()`.
- Exponentiation is done after a **per-row maximum shift** (stacking) or a global maximum shift
  (log-score weights). The row shift changes the stacking objective by a `w`-independent constant,
  so the optimum is unchanged, and it keeps the pool finite at log scores where an unshifted
  implementation underflows to zero and reports a degenerate weight vector.
- ArviZ parity is a *property Impulso tests*, not a dependency it takes: with degenerate
  (draw-constant) log-likelihoods, `pool_forecasts` reproduces `az.compare(method="stacking")` and
  `method="pseudo-BMA"`. The primary regression check is a direct SciPy re-solve of the score
  matrix, so ArviZ API drift cannot silently weaken the suite.
