# Litterman API Design

## Principles

- Economist-friendly: users think in endogenous/exogenous variables, impulse responses, FEVD — not tensors or MCMC chains.
- PyMC backend, but the user never writes a model function.
- Opinionated defaults (Minnesota prior, 89% HDI) with typed escape hatches.
- Immutable types throughout. Each stage of the workflow is a distinct type.
- Pydantic 2.x at every boundary with strict validation. beartype available for runtime checking.
- Extension points defined via `typing.Protocol` — priors, samplers, and identification schemes are pluggable.

## Module Layout

```
litterman/
  __init__.py              # Re-exports: VAR, VARData, FittedVAR, IdentifiedVAR, select_lag_order
  data.py                  # VARData
  spec.py                  # VAR
  fitted.py                # FittedVAR
  identified.py            # IdentifiedVAR
  results.py               # VARResultBase, IRFResult, FEVDResult, ForecastResult,
                           #   HistoricalDecompositionResult, LagOrderResult, HDIResult
  protocols.py             # Prior, Sampler, IdentificationScheme protocols
  priors.py                # MinnesotaPrior (implements Prior)
  samplers.py              # NUTSSampler (implements Sampler)
  identification.py        # Cholesky, SignRestriction (implement IdentificationScheme)
  plotting/
    __init__.py            # Re-exports: plot_irf, plot_fevd, plot_forecast,
                           #   plot_historical_decomposition
    _irf.py
    _fevd.py
    _forecast.py
    _historical_decomposition.py
```

## Type Progression

```
VAR (immutable spec) → FittedVAR (reduced-form posterior) → IdentifiedVAR (structural)
```

Each stage is immutable (`model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)`). Methods only exist on the stage where they make sense. Static type checkers catch misuse (e.g., calling `impulse_response` on `FittedVAR`). Runtime type checking is available via `litterman.enable_runtime_checks()` (beartype).

## Protocols

Extension points are defined as `typing.Protocol` classes in `litterman.protocols`. v1 ships one implementation of each; v2 can add more without modifying core code.

```python
from litterman.protocols import Prior, Sampler, IdentificationScheme

class Prior(Protocol):
    """Contract for prior specifications."""
    def build_priors(self, n_vars: int, n_lags: int) -> dict: ...

class Sampler(Protocol):
    """Contract for posterior sampling strategies."""
    def sample(self, model: pm.Model) -> az.InferenceData: ...

class IdentificationScheme(Protocol):
    """Contract for structural identification schemes."""
    def identify(self, idata: az.InferenceData, var_names: list[str]) -> az.InferenceData: ...
```

All protocol implementations are Pydantic models (immutable, validated). The `VAR` spec accepts any object satisfying `Prior`; `FittedVAR.set_identification_strategy()` accepts any `IdentificationScheme`.

## Data

```python
from litterman import VARData

# From DataFrame
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"], exog=["oil_price"])

# Raw constructor
data = VARData(
    endog=np.array(...),              # (T, n)
    exog=np.array(...),               # (T, k) or None
    endog_names=["gdp", "inflation", "rate"],
    exog_names=["oil_price"],         # None if no exogenous
    index=pd.DatetimeIndex(...),      # required
)
```

`VARData` is a Pydantic model (`frozen=True`). Validation:

- Array shapes must be consistent: `endog.shape[0] == len(index)`, `len(endog_names) == endog.shape[1]`.
- `exog`, if provided, must have `exog.shape[0] == endog.shape[0]`.
- All array values must be finite (`np.isfinite().all()`); rejects NaN and Inf at construction.
- Minimum 2 endogenous variables (univariate AR is out of scope).
- Arrays are defensively copied on ingestion and set to read-only (`arr.flags.writeable = False`) to enforce immutability.

The `from_df` class method extracts arrays, column names, and the index from a pandas DataFrame.

## Specification

```python
from litterman import VAR
from litterman.priors import MinnesotaPrior

spec = VAR(lags="bic", prior="minnesota")
spec = VAR(lags=4, prior=MinnesotaPrior(tightness=0.1))
spec = VAR(lags="aic", max_lags=12, prior=MinnesotaPrior(tightness=0.1))
```

Fields:

- `lags: int | Literal["aic", "bic", "hq"]` — fixed lag order or automatic selection criterion. When `int`, must be `>= 1`.
- `max_lags: int | None` — upper bound for automatic selection. Only valid when `lags` is a string.
- `prior: Literal["minnesota"] | Prior` — string shorthand resolves to the default constructor via a registry (e.g., `"minnesota"` becomes `MinnesotaPrior()`). Pass the object for full control.

Cross-field validation via `@model_validator(mode='after')`:

```python
@model_validator(mode='after')
def _validate_max_lags(self) -> Self:
    if self.max_lags is not None and isinstance(self.lags, int):
        raise ValueError("max_lags is only valid when lags is a selection criterion ('aic', 'bic', 'hq')")
    return self
```

Immutable. No state, no side effects.

## Priors

```python
from litterman.priors import MinnesotaPrior

prior = MinnesotaPrior()
prior = MinnesotaPrior(tightness=0.2, decay="harmonic", cross_shrinkage=0.5)
```

`MinnesotaPrior` implements the `Prior` protocol. Pydantic model with constrained fields:

- `tightness: float = Field(0.1, gt=0)` — overall shrinkage toward the prior mean.
- `decay: Literal["harmonic", "geometric"] = "harmonic"` — how coefficients shrink on longer lags.
- `cross_shrinkage: float = Field(0.5, ge=0, le=1)` — shrinkage on other variables' lags vs own lags.
- Prior mean is random walk (1.0) for own first lag by default.

## Estimation

```python
from litterman.samplers import NUTSSampler

result: FittedVAR = spec.fit(data)
result: FittedVAR = spec.fit(data, sampler=NUTSSampler(draws=2000, chains=4))
```

Default sampler when omitted: `NUTSSampler()` with PyMC defaults.

`NUTSSampler` implements the `Sampler` protocol. Pydantic model with explicit, whitelisted fields (no `**kwargs` pass-through):

```python
class NUTSSampler(BaseModel):
    model_config = ConfigDict(frozen=True)

    draws: int = Field(1000, ge=1)
    tune: int = Field(1000, ge=0)
    chains: int = Field(4, ge=1)
    cores: int | None = Field(None, ge=1)       # None = auto-detect
    target_accept: float = Field(0.8, gt=0, lt=1)
    random_seed: int | None = None
```

When `lags` is a string, automatic lag selection (OLS-based AIC/BIC/HQ) runs before estimation. The selected lag order is stored on the returned `FittedVAR`.

Sampling stores the full `az.InferenceData` including the `log_likelihood` group (required for future model comparison via `az.compare()`).

Returns `FittedVAR` (immutable).

## FittedVAR

Holds the reduced-form posterior. Exposes:

```python
result.n_lags            # int — actual lag order used (explicit or auto-selected)
result.has_exog          # bool — whether model includes exogenous variables

# Posterior coefficient access
result.coefficients      # posterior draws of B matrices
result.intercepts        # posterior draws of intercept vectors
result.sigma             # posterior draws of residual covariance

# ArviZ InferenceData (full escape hatch)
result.idata             # az.InferenceData — includes posterior, log_likelihood groups

# Forecasting (no identification needed)
fcast = result.forecast(steps=8)
fcast = result.forecast(steps=8, exog_future=future_data)

# Identification
identified = result.set_identification_strategy(Cholesky(ordering=["gdp", "inflation", "rate"]))
```

`exog_future` is required if and only if the model was fitted with exogenous variables (`result.has_exog`). Raises `ValueError` if the argument is missing when needed, or provided when not needed.

Repr excludes large arrays:

```python
model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

def __repr__(self) -> str:
    return f"FittedVAR(n_lags={self.n_lags}, n_vars={...}, n_draws={...}, n_chains={...})"
```

## Identification

```python
from litterman.identification import Cholesky, SignRestriction

identified: IdentifiedVAR = result.set_identification_strategy(Cholesky(ordering=["gdp", "inflation", "rate"]))
```

`set_identification_strategy()` returns a new `IdentifiedVAR` (immutable). The same `FittedVAR` can produce multiple `IdentifiedVAR` instances with different schemes. Both `Cholesky` and `SignRestriction` implement the `IdentificationScheme` protocol.

v1 identification schemes: Cholesky, sign restrictions.

## IdentifiedVAR

All structural post-estimation methods live here:

```python
irf = identified.impulse_response(horizon=20)
irf = identified.impulse_response(horizon=20, shock="gdp")
irf = identified.impulse_response(horizon=20, shock="gdp", response="inflation")

fevd = identified.forecast_error_variance_decomposition(horizon=20)
fevd = identified.fevd(horizon=20)  # alias

hd = identified.historical_decomposition()
hd = identified.historical_decomposition(start=pd.Timestamp("2000-01-01"), end=pd.Timestamp("2020-12-31"))
hd = identified.historical_decomposition(cumulative=True)
```

`impulse_response` returns all variable pairs by default. Optional `shock` and `response` arguments filter to specific pairs.

`forecast_error_variance_decomposition()` is the full name; `fevd()` is a convenience alias.

`historical_decomposition()` accepts optional `start`/`end` timestamps to restrict the decomposition period, and a `cumulative` flag for cumulative vs. period-specific shock contributions.

## Result Objects

All post-estimation methods return result objects inheriting from `VARResultBase`, defined in `litterman.results`:

```python
class HDIResult(BaseModel):
    """Structured HDI output with separate lower/upper bounds."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lower: pd.DataFrame
    upper: pd.DataFrame
    prob: float

class VARResultBase(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData

    def median(self) -> pd.DataFrame: ...
    def hdi(self, prob: float = 0.89) -> HDIResult: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def plot(self) -> Figure: ...  # abstract — each subclass renders differently

    def __repr__(self) -> str:
        """Compact repr — never dumps array contents."""
        ...
```

Subclasses: `IRFResult`, `FEVDResult`, `ForecastResult`, `HistoricalDecompositionResult`.

The `.idata` property exposes the full ArviZ `InferenceData` for custom analysis, including integration with `az.summary()`, `az.plot_trace()`, and `az.compare()`.

`.hdi()` returns an `HDIResult` with `.lower` and `.upper` DataFrames rather than a MultiIndex DataFrame. Default HDI probability is 0.89 (Bayesian convention, configurable per-call).

## Plotting

```python
# Convenient default
irf.plot()

# Full control
from litterman.plotting import plot_irf, plot_fevd, plot_forecast, plot_historical_decomposition
fig, axes = plot_irf(irf, variables=["gdp"], figsize=(12, 8))
```

`.plot()` on result objects delegates to the corresponding function in `litterman.plotting`. Import the function directly for full matplotlib control.

`litterman.plotting` is a package (not a single module) to keep each plot type in its own file. The `__init__.py` re-exports the four public functions.

Plot types:

- IRF: credible bands per variable pair.
- FEVD: stacked area chart.
- Forecast: fan chart with credible bands.
- Historical decomposition: stacked contribution chart.

## Lag Selection (standalone)

```python
from litterman import select_lag_order
from litterman.results import LagOrderResult

ic: LagOrderResult = select_lag_order(data, max_lags=12)
ic.aic   # int — optimal lag order by AIC
ic.bic   # int — optimal lag order by BIC
ic.hq    # int — optimal lag order by Hannan-Quinn
ic.summary()  # formatted table of all criteria by lag order
```

`LagOrderResult` is a Pydantic model:

```python
class LagOrderResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    aic: int
    bic: int
    hq: int
    criteria_table: pd.DataFrame = Field(repr=False)  # excluded from repr

    def summary(self) -> pd.DataFrame:
        return self.criteria_table
```

Uses OLS-based information criteria (fast). Also called internally when `VAR(lags="bic")`.

## Runtime Type Checking

```python
import litterman
litterman.enable_runtime_checks()  # wraps public API with beartype
```

Intended to be on in tests, off in production.

## Serialization

Deferred to a future version. v1 does not provide `save()`/`load()` methods. Users needing persistence should use ArviZ's `InferenceData.to_netcdf()` for posterior data and reconstruct model objects from saved specifications. When persistence is added, it will avoid `pickle` in favor of safe formats (NetCDF + JSON).

## v1 Scope

Included:

- VAR specification and estimation (Minnesota prior, NUTS sampler via PyMC)
- Automatic lag selection (AIC, BIC, HQ via OLS)
- Forecasting with exogenous support
- Impulse response functions (Cholesky, sign restrictions)
- Forecast error variance decomposition
- Historical decomposition (with time range and cumulative options)
- Built-in plotting for all result types (as a package)
- ArviZ-compatible `InferenceData` throughout (log-likelihood stored for future model comparison)
- Extension point protocols for priors, samplers, and identification schemes

Deferred to v2:

- Granger causality
- Conditional forecasting / scenario analysis
- Long-run restrictions (Blanchard-Quah)
- External instruments / proxy SVAR
- Additional priors (Normal-Wishart, steady-state)
- Structural impact matrix (`A0`) property on `IdentifiedVAR`
- Model comparison utilities (`az.compare()` wrappers)
- Serialization / persistence

## Full Example

```python
from litterman import VAR, VARData, select_lag_order
from litterman.priors import MinnesotaPrior
from litterman.samplers import NUTSSampler
from litterman.identification import Cholesky

# Data
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"], exog=["oil_price"])

# Lag selection (optional — can also use lags="bic" on VAR)
ic = select_lag_order(data, max_lags=12)
ic.summary()

# Specification
spec = VAR(lags=ic.bic, prior=MinnesotaPrior(tightness=0.2))

# Estimation
result = spec.fit(data, sampler=NUTSSampler(draws=2000, chains=4))

# Metadata
result.n_lags    # selected lag order
result.has_exog  # True

# Coefficients
result.coefficients
result.sigma

# Forecast
fcast = result.forecast(steps=8)
fcast.median()
fcast.hdi(prob=0.89)        # HDIResult with .lower and .upper DataFrames
fcast.plot()

# Structural analysis
identified = result.set_identification_strategy(Cholesky(ordering=["gdp", "inflation", "rate"]))

irf = identified.impulse_response(horizon=20)
irf.plot()

fevd = identified.fevd(horizon=20)
fevd.plot()

hd = identified.historical_decomposition()
hd.plot()

# Escape hatch
result.idata  # az.InferenceData — full posterior, log-likelihood, etc.
irf.idata     # az.InferenceData — IRF-specific draws
```
