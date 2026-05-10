# Volatility-process seam exposes L_t (lower-triangular Cholesky factor)

The `VolatilityProcess` Protocol is the seam unifying constant and stochastic volatility under a single deep `VAR` pipeline. Adapters expose the *time-`t` lower-triangular Cholesky factor* `L_t` (such that `Σ_t = L_t @ L_t.T`) as the seam's primary output — not `Σ_t` directly, and not `(h_t, R)` separately.

**Why**: Cholesky identification needs `L`, not `Σ` — returning `L_t` avoids redundant `cholesky(Σ_t)` per draw per `t`. Sign-restriction identification rotates `L`, same starting point. Today's constant-vol parameterisation, Clark-style SV (per-variable log-vol + constant correlation), and Primiceri-style SV (TVP correlations) all compose `L_t` natively. Per-variable volatility paths derive cheaply from `L_t` for plotting.

**Trade-off accepted**: the seam is opinionated about Cholesky factorisation. Adapters that internally parameterise covariance differently (factor models, low-rank approximations) must convert to `L_t` at the seam — a one-shot `cholesky()` call per draw, acceptable for the simplification it provides everywhere downstream.

**Considered and rejected**:
- `Σ_t` as the primary output — forces every identification call to recompute `chol(Σ_t)` per draw per `t`; wasteful for SV where `L_t` is already the natural parameterisation.
- `(h_t, R)` returned separately — the constant-vol adapter has no `h`; Primiceri's correlations are time-varying so there's no constant `R`. The seam shape forecloses adapters we want to support.
