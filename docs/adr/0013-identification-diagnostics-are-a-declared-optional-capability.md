# Identification diagnostics are a declared optional capability, not a Protocol member

Every identification scheme that has something to report about its most recent `identify()` call — acceptance rates, first-stage instrument strength, long-run screen condition numbers, memoisation cache hits — exposes it through one public surface: `last_diagnostics`, a flat `dict[str, float]` under scheme-prefixed keys, backed by a single-call scratchpad that each `identify()` overwrites. The pipeline (`IdentifiedVAR.shock_matrix`) reads it with a single `getattr(scheme, "last_diagnostics", None)` and surfaces the entries onto the shock-matrix `attrs`. The capability is documented on the `IdentificationScheme` Protocol docstring but deliberately **not** declared as a structural Protocol member.

This replaces a shadow protocol: two undeclared private spellings (`SignRestriction._last_acceptance_rate`, a bare float; `_last_diagnostics` dicts on `LongRunRestriction`/`ProxySVAR`/`ZeroSignRestriction`) that `IdentifiedVAR` reached for through separate `getattr` probes — one concept, two attribute names, two value shapes, none of it on the declared seam.

## Considered options

- **A structural Protocol member** (`last_diagnostics` declared in the `IdentificationScheme` body) — rejected. The Protocol is `@runtime_checkable`, and `isinstance` checks member *presence*: declaring it would break any minimal third-party scheme implementing only `identify()`/`shock_coords()` the moment `enable_runtime_checks()` is on, and would force `Cholesky` to carry an empty dict purely for uniformity. The strictness buys nothing Impulso's own five adapters need.
- **A pure return** (`identify()` returns `(P, diagnostics)`) — rejected for now, not on principle. It would restore strict purity to the scheme contract, but it churns the seam signature at every callsite — `shock_matrix`, `_identify_per_t`, the forecast-side scenario engine, and dozens of direct `identify()` calls across four test files whose interface-level style is the suite's strength. Revisit if a consumer ever needs diagnostics the immediately-read scratchpad cannot express (e.g. concurrent identify calls on one scheme instance).

## Consequences

- CONTEXT.md's identification-scheme entry now reads "pure in its output" with `last_diagnostics` as the one declared side effect, and records the **scheme-prefixed diagnostic keys** convention (`sign_restriction_`, `zero_sign_`, `long_run_`, `proxy_`) so entries surfaced onto shared `attrs` can never mislabel another scheme's diagnostics.
- Public `shock_matrix().attrs` keys are bit-stable, including the historical spelling `sign_restriction_acceptance_rate`. The only additions are the cache-hit flags `long_run_screen_cache_hit` and `proxy_impact_cache_hit` (0.0/1.0 floats — NetCDF-attr-safe).
- Memoisation became observable through the interface: the cache tests that previously monkeypatched `ProxySVAR._aligned_residuals` and read `_impact_cache`/`_lr_cache` directly now assert the cache-hit flag.
- The scratchpad is single-call and not reentrant. Reading it between calls, or after another `identify()`, is undefined behaviour — documented on the Protocol and in CONTEXT.md.
- `_samples_rotations` (the rotation-sampling capability flag) is deliberately untouched: a separate concern with one consumer, documented alongside the diagnostics capability on the Protocol docstring.
