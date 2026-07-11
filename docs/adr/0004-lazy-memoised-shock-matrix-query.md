# Structural shock matrix is a lazy, memoised query — not stored in the posterior

`IdentifiedVAR.shock_matrix(at=...)` is the single pathway producing the structural shock matrix: resolve `at`, query the volatility process for `L_t` (slice or path), apply the identification scheme, return a labelled DataArray. The result is memoised per `at` value on the instance (internal cache via `object.__setattr__`, the established frozen-model pattern). `set_identification_strategy` no longer computes or stores `structural_shock_matrix` in the posterior; IRF, FEVD, and historical decomposition all go through the query.

**Why**: Under `SignRestriction`, `identify()` samples rotations — previously each of IRF/FEVD/HD called it independently, so the FEVD was computed from *different* accepted rotations than the IRF it supposedly decomposes, and repeat calls gave different answers. Memoising per `at` makes every quantity from one `IdentifiedVAR` share the same structural draws (statistically coherent, deterministic per object) and runs the expensive rejection loop once instead of per quantity. Dropping the eager stored copy removes a second source of truth that IRF/FEVD/HD never read, and lets `IdentifiedVAR` be constructed through normal Pydantic validation instead of `model_construct`.

**Considered and rejected**:
- *Re-draw on every call* (status quo, blessed) — stateless, but quantities from one object stay mutually inconsistent under set identification and are non-reproducible without threading an rng everywhere.
- *Eager computation at identification time* — restores the stored copy and makes construction expensive; queries at other `at` values would still need the lazy path anyway.
