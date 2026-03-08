# Tier 1 Extensions Design: Integrating Foundation Features into Impulso

**Date**: 2026-03-08
**Scope**: Design for integrating the 5 highest-priority methodological extensions into Impulso's architecture.
**Companion**: See `2026-03-08-var-svar-extensions-research.md` for the full research review.

---

## Overview

Five extensions form Impulso's foundation layer. Together they make Impulso competitive with the BVAR R package and the ECB's BEAR toolbox.

| # | Extension | Type | Layer |
|---|-----------|------|-------|
| 1 | Dummy Observation Priors | Prior (data augmentation) | 1 |
| 2 | Conjugate Gibbs Sampler | Sampler (new model class) | 2 |
| 3 | Hierarchical Prior Selection (GLP) | Prior optimisation | 3 |
| 4 | Long-Run Restrictions | Identification | 4 |
| 5 | Conditional Forecasting | Analysis | 5 |

### Dependency Graph

```
Layer 1: VARData.with_dummy_observations()
  └─ Tests: augmented data shapes, dummy values

Layer 2: ConjugateVAR + NIW math
  └─ Tests: posterior matches known analytical results
  └─ Depends on: Layer 1 (optional, for dummy obs support)

Layer 3: ConjugateVAR.optimize_prior()
  └─ Tests: marginal likelihood correctness, optimiser convergence
  └─ Depends on: Layer 2

Layer 4: LongRunRestriction
  └─ Tests: Blanchard-Quah replication
  └─ Independent

Layer 5: ForecastCondition + conditional_forecast()
  └─ Tests: constrained paths respected
  └─ Independent
```

### Architecture Diagram

```
                    VARData
                      │
            ┌─────────┴─────────┐
            │                   │
  .with_dummy_observations()  (unchanged)
            │                   │
        VARData*             VARData
            │                   │
       ConjugateVAR           VAR
     (.optimize_prior())   (.fit() via PyMC)
            │                   │
            └─────────┬─────────┘
                      │
                  FittedVAR
              .forecast()
              .conditional_forecast()  ← NEW
              .set_identification_strategy()
                      │
              IdentifiedVAR
              .impulse_response()
              .conditional_forecast()  ← NEW
              .fevd() / .historical_decomposition()
                      │
          ┌───────────┤
    LongRunRestriction (NEW)
    Cholesky (existing)
    SignRestriction (existing)
```

---

## Layer 1: Dummy Observation Priors

### Problem

Encoding beliefs about unit roots, cointegration, and persistence requires dummy observation priors (Doan, Litterman & Sims 1984; Sims 1993). These maintain conjugacy, making them critical for the ConjugateVAR path.

### Design

A method on `VARData` returns a new `VARData` with appended dummy rows. This is model-agnostic — works with both `VAR` and `ConjugateVAR`.

```python
class VARData(ImpulsoBaseModel):
    def with_dummy_observations(
        self,
        n_lags: int,
        mu: float | None = None,      # sum-of-coefficients tightness
        delta: float | None = None,    # single-unit-root tightness
    ) -> "VARData":
        """Return new VARData with dummy observations appended.

        Args:
            n_lags: Number of VAR lags (needed to construct dummy rows).
            mu: Sum-of-coefficients hyperparameter. Larger = weaker prior.
                Encodes belief that sum of own-lag coefficients is close to 1.
            delta: Single-unit-root hyperparameter. Larger = weaker prior.
                Encodes belief that variables persist at initial levels.

        Returns:
            New VARData with dummy observations appended to endog.
        """
```

### Dummy Types

**Sum-of-coefficients** (controlled by `mu`): Appends `n_vars` rows. Row `i` has `y_bar_i / mu` in position `i`, zeros elsewhere. Repeated across lag positions. Encodes: the sum of own-lag coefficients for variable `i` is close to 1.

**Single-unit-root** (controlled by `delta`): Appends 1 row with `y_bar / delta` across all variables. Encodes: when all variables are at their sample means, they persist at those levels.

Both use `y_bar` = sample means of each variable (standard in the literature).

### Validation

- At least one of `mu` or `delta` must be provided.
- Both must be strictly positive.
- `n_lags` must be a positive integer.
- The returned `VARData` has a synthetic `DatetimeIndex` extension for dummy rows (using the last observed frequency).

### Usage

```python
data = VARData.from_df(df)
augmented = data.with_dummy_observations(n_lags=4, mu=5.0, delta=1.0)

# Works with either estimation path:
fitted = VAR(lags=4).fit(augmented, sampler=NUTSSampler())
fitted = ConjugateVAR(lags=4).fit(augmented)
```

---

## Layer 2: ConjugateVAR

### Problem

NUTS is general but slow. For the standard Minnesota prior with NIW conjugacy, the posterior is available in closed form. Direct sampling yields iid draws — no burn-in, no autocorrelation, orders of magnitude faster.

### Design

A separate model class `ConjugateVAR` alongside `VAR`. Both produce `FittedVAR` for unified downstream analysis. Complete code-path isolation: `ConjugateVAR` never touches PyMC.

```python
class ConjugateVAR(ImpulsoBaseModel):
    lags: int | Literal["aic", "bic", "hq"]
    max_lags: int | None = None
    prior: Literal["minnesota"] | MinnesotaPrior = "minnesota"
    draws: int = Field(2000, ge=1)
    random_seed: int | None = None

    def fit(self, data: VARData) -> FittedVAR:
        """Direct NIW posterior sampling. No PyMC."""

    def optimize_prior(self, data: VARData, optimize_dummy: bool = False) -> MinnesotaPrior:
        """GLP marginal likelihood optimisation (Layer 3)."""

    def marginal_likelihood(self, data: VARData) -> float:
        """Log marginal likelihood p(Y|lambda). Used internally by optimize_prior()."""
```

### NIW Posterior Mathematics

Prior:
- `vec(B) | Sigma ~ N(vec(B_prior), Sigma ⊗ V_prior)`
- `Sigma ~ IW(S_prior, nu_prior)`

Posterior (closed-form):
- `V_posterior = (V_prior^{-1} + X'X)^{-1}`
- `B_posterior = V_posterior @ (V_prior^{-1} @ B_prior + X' @ Y)`
- `nu_posterior = nu_prior + T`
- `S_posterior = S_prior + Y'Y + B_prior' @ V_prior^{-1} @ B_prior - B_posterior' @ V_posterior^{-1} @ B_posterior`

### Sampling Algorithm

For each of `draws` iterations:
1. Draw `Sigma ~ InverseWishart(S_posterior, nu_posterior)` using `scipy.stats.invwishart`
2. Draw `B | Sigma ~ MatrixNormal(B_posterior, Sigma, V_posterior)` using Cholesky of Sigma and V_posterior
3. Extract intercept (first row or column of B depending on design matrix convention)

Each draw is iid. Packed into `az.InferenceData` with `chains=1` for downstream compatibility.

### NIW Parameter Conversion from MinnesotaPrior

`MinnesotaPrior.build_priors()` returns `B_mu` (prior mean) and `B_sigma` (prior standard deviations). `ConjugateVAR` converts these to NIW form:

- `B_prior = B_mu` (prior mean matrix)
- `V_prior = diag(B_sigma^2)` (diagonal prior covariance from Minnesota structure)
- `S_prior = diag(sigma_ols^2)` (OLS residual variances for scale)
- `nu_prior = n_vars + 2` (minimally informative degrees of freedom)

### InferenceData Output

The returned `FittedVAR.idata.posterior` contains:
- `"B"`: shape `(1, draws, n_vars, n_vars * n_lags)`
- `"intercept"`: shape `(1, draws, n_vars)`
- `"Sigma"`: shape `(1, draws, n_vars, n_vars)`

Identical structure to `NUTSSampler` output. All downstream methods (`forecast`, `set_identification_strategy`, etc.) work unchanged.

### File Location

New file: `src/impulso/conjugate.py`

---

## Layer 3: Hierarchical Prior Selection (GLP)

### Problem

Minnesota prior hyperparameters (tightness, cross_shrinkage) are typically set ad hoc. The wrong choice degrades forecasts. Giannone, Lenza & Primiceri (2015) showed that with conjugacy, the marginal likelihood is available in closed form, enabling fast data-driven optimisation.

### Design

A method on `ConjugateVAR` that returns an optimised `MinnesotaPrior`.

```python
def optimize_prior(
    self,
    data: VARData,
    optimize_dummy: bool = False,
) -> MinnesotaPrior:
    """Find Minnesota hyperparameters that maximise the marginal likelihood.

    Args:
        data: The VAR data (may include dummy observations).
        optimize_dummy: If True, also optimise dummy observation
            hyperparameters (mu, delta). Requires data created via
            with_dummy_observations().

    Returns:
        MinnesotaPrior with optimal tightness and cross_shrinkage.
    """
```

### Hyperparameters Optimised

| Parameter | Range | Always optimised? |
|-----------|-------|-------------------|
| `tightness` | `(0.001, 10.0)` | Yes |
| `cross_shrinkage` | `(0.01, 1.0)` | Yes |
| `mu` | `(0.1, 50.0)` | Only if `optimize_dummy=True` |
| `delta` | `(0.1, 50.0)` | Only if `optimize_dummy=True` |

`decay` is discrete ("harmonic" / "geometric") — not optimised, user chooses.

### Marginal Likelihood

With NIW conjugate prior, the log marginal likelihood is:

```
log p(Y|lambda) = -(T * n_vars / 2) * log(pi)
                  + (nu_posterior / 2) * log|S_prior|
                  - (nu_posterior / 2) * log|S_posterior|
                  - (n_vars / 2) * log|V_posterior / V_prior|
                  + sum of log-gamma terms
```

This is a smooth, differentiable function of the hyperparameter vector `lambda`. Optimised via `scipy.optimize.minimize` with method `"L-BFGS-B"` and parameter bounds.

### One-Step Shorthand

Register `"minnesota_optimized"` in `ConjugateVAR` (not in `_PRIOR_REGISTRY` since it's specific to conjugate estimation):

```python
fitted = ConjugateVAR(lags=4, prior="minnesota_optimized").fit(data)
```

This calls `optimize_prior(data)` internally, then fits with the result.

### Usage

```python
# Explicit two-step:
cvar = ConjugateVAR(lags=4)
optimal_prior = cvar.optimize_prior(data)
# Inspect: optimal_prior.tightness, optimal_prior.cross_shrinkage
fitted = ConjugateVAR(lags=4, prior=optimal_prior).fit(data)

# One-step:
fitted = ConjugateVAR(lags=4, prior="minnesota_optimized").fit(data)
```

---

## Layer 4: Long-Run Restrictions

### Problem

Theory sometimes predicts long-run effects rather than short-run orderings. Blanchard & Quah (1989): demand shocks have no permanent effect on output; only supply shocks do. Long-run restrictions identify structural shocks via their cumulative impact as horizon approaches infinity.

### Design

A new `IdentificationScheme` in `identification.py`.

```python
class LongRunRestriction(ImpulsoModel):
    ordering: list[str]  # Variable ordering (same semantics as Cholesky)

    def identify(
        self, idata: az.InferenceData, var_names: list[str]
    ) -> az.InferenceData:
        """Blanchard-Quah long-run identification."""
```

### Algorithm (per posterior draw)

1. Extract coefficient matrix `B` (shape `n_vars, n_vars * n_lags`) and covariance `Sigma` (shape `n_vars, n_vars`).
2. Compute the lag polynomial sum: `lag_coefficient_sum = A_1 + A_2 + ... + A_p` where each `A_j` is the `j`-th lag block of `B`.
3. Compute the long-run multiplier: `long_run_multiplier = inv(I - lag_coefficient_sum)`.
4. Compute the long-run covariance: `long_run_covariance = long_run_multiplier @ Sigma @ long_run_multiplier.T`.
5. Cholesky decompose: `long_run_cholesky = chol(long_run_covariance)`.
6. Recover the structural impact matrix: `structural_impact_matrix = inv(long_run_multiplier) @ long_run_cholesky`.
7. Reorder columns/rows per `self.ordering`.

### Validation

- `ordering` must contain exactly the variables in `var_names` (possibly reordered).
- Stationarity check per draw: eigenvalues of companion matrix must be inside the unit circle. Non-stationary draws emit a warning and fall back to impact Cholesky.

### Output

Same `InferenceData` structure as `Cholesky`:
- `posterior["structural_shock_matrix"]`: shape `(chains, draws, n_vars, n_vars)`
- Coordinates: `{"shock": ordering, "response": ordering}`

### Usage

```python
scheme = LongRunRestriction(ordering=["output", "prices"])
identified = fitted.set_identification_strategy(scheme)
irfs = identified.impulse_response(horizon=40)
```

---

## Layer 5: Conditional Forecasting

### Problem

Policy analysis requires forecasts conditional on assumed paths. "What if the central bank holds rates at 5% for four quarters?" Standard unconditional forecasts cannot answer this.

### Design

A `ForecastCondition` class defines constraints. Methods on both `FittedVAR` (reduced-form) and `IdentifiedVAR` (structural) produce conditional forecasts.

### ForecastCondition

```python
class ForecastCondition(ImpulsoModel):
    variable: str                              # Which variable to constrain
    periods: list[int]                         # Which forecast steps (0-indexed)
    values: list[float]                        # Target values at those periods
    constraint_type: Literal["hard"] = "hard"  # "soft" reserved for future

    @model_validator(mode="after")
    def _validate_periods_values_match(self) -> Self:
        """Ensure periods and values have equal length."""
```

`constraint_type="soft"` and a `tolerance` field are reserved for future use. Initial implementation supports hard constraints only.

**File location**: `src/impulso/conditions.py`

### Methods

On `FittedVAR` (`fitted.py`):

```python
def conditional_forecast(
    self,
    steps: int,
    conditions: list[ForecastCondition],
    exog_future: np.ndarray | None = None,
) -> ConditionalForecastResult:
    """Reduced-form conditional forecast. Constrains observable variable paths."""
```

On `IdentifiedVAR` (`identified.py`):

```python
def conditional_forecast(
    self,
    steps: int,
    conditions: list[ForecastCondition],
    shock_conditions: list[ForecastCondition] | None = None,
    exog_future: np.ndarray | None = None,
) -> ConditionalForecastResult:
    """Structural conditional forecast. Constrains observables and/or shocks."""
```

### Algorithm (Waggoner & Zha 1999, hard constraints, reduced-form)

For each posterior draw:
1. Compute the unconditional forecast path (reuse existing `forecast()` internals).
2. Compute MA coefficient matrices `Phi_0, Phi_1, ..., Phi_{h-1}` (reduced-form impulse responses).
3. Stack constraint equations into a linear system: `R @ shocks = target_values - unconditional_forecast`, where `R` is built from the relevant rows of the MA coefficients.
4. Solve for the constrained shocks via least-squares (`np.linalg.lstsq`).
5. Propagate constrained shocks through the MA representation to produce the conditional forecast.

The structural version on `IdentifiedVAR` additionally uses the structural impact matrix to map between structural and reduced-form shocks, enabling `shock_conditions`.

### Result Type

```python
class ConditionalForecastResult(ForecastResult):
    conditions: list[ForecastCondition]
```

Inherits `.median()`, `.hdi()`, `.to_dataframe()`, `.plot()` from `ForecastResult`. The `.conditions` attribute lets users inspect what was constrained.

### Validation

- All `condition.variable` values must be in `var_names`.
- All `condition.periods` must be in `range(steps)`.
- For `shock_conditions` on `IdentifiedVAR`: shock names must match identification scheme shock names.
- System must not be over-determined (more constraints than degrees of freedom at any period).

### Usage

```python
from impulso import ForecastCondition

conditions = [
    ForecastCondition(variable="interest_rate", periods=[0, 1, 2, 3], values=[5.0, 5.0, 5.0, 5.0]),
]

# Reduced-form:
result = fitted.conditional_forecast(steps=12, conditions=conditions)
result.plot()

# Structural (with shock constraints):
shock_conds = [
    ForecastCondition(variable="monetary_shock", periods=[0, 1, 2, 3], values=[0.0, 0.0, 0.0, 0.0]),
]
result = identified.conditional_forecast(steps=12, conditions=conditions, shock_conditions=shock_conds)
```

---

## New Files Summary

| File | Contents |
|------|----------|
| `src/impulso/conjugate.py` | `ConjugateVAR` class (Layers 2 + 3) |
| `src/impulso/conditions.py` | `ForecastCondition` class (Layer 5) |

## Modified Files Summary

| File | Changes |
|------|---------|
| `src/impulso/data.py` | Add `with_dummy_observations()` method (Layer 1) |
| `src/impulso/identification.py` | Add `LongRunRestriction` class (Layer 4) |
| `src/impulso/fitted.py` | Add `conditional_forecast()` method (Layer 5) |
| `src/impulso/identified.py` | Add `conditional_forecast()` method (Layer 5) |
| `src/impulso/results.py` | Add `ConditionalForecastResult` class (Layer 5) |
| `src/impulso/__init__.py` | Export new public API |

## Key References

- Doan, Litterman & Sims (1984) — Dummy observation priors
- Sims (1993) — Single-unit-root prior
- Kadiyala & Karlsson (1997) — Conjugate NIW estimation
- Waggoner & Zha (1999) — Conditional forecasting
- Blanchard & Quah (1989) — Long-run restrictions
- Giannone, Lenza & Primiceri (2015) — Hierarchical prior selection
- Miranda-Agrippino & Ricco (2018) — Bayesian VAR survey
