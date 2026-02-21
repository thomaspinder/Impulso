# Litterman API Design

## Principles

- Economist-friendly: users think in endogenous/exogenous variables, impulse responses, FEVD — not tensors or MCMC chains.
- PyMC backend, but the user never writes a model function.
- Opinionated defaults (Minnesota prior, 89% HDI) with typed escape hatches.
- Immutable types throughout. Each stage of the workflow is a distinct type.
- Pydantic at every boundary. beartype available for runtime checking.

## Type Progression

```
VAR (immutable spec) → FittedVAR (reduced-form posterior) → IdentifiedVAR (structural)
```

Each stage is immutable. Methods only exist on the stage where they make sense. Static type checkers catch misuse (e.g., calling `impulse_response` on `FittedVAR`). Runtime type checking is available via `litterman.enable_runtime_checks()` (beartype).

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

`VARData` is a Pydantic model. It validates array shapes, name lengths, and requires a `DatetimeIndex`. The `from_df` class method extracts arrays, column names, and the index from a pandas DataFrame.

## Specification

```python
from litterman import VAR
from litterman.priors import MinnesotaPrior

spec = VAR(lags="bic", prior="minnesota")
spec = VAR(lags=4, max_lags=12, prior=MinnesotaPrior(tightness=0.1))
```

Fields:

- `lags: int | Literal["aic", "bic", "hq"]` — fixed lag order or automatic selection criterion.
- `max_lags: int | None` — upper bound for automatic selection. Only valid when `lags` is a string; raises a validation error otherwise.
- `prior: str | MinnesotaPrior | ...` — string shorthand resolves to the default constructor (e.g., `"minnesota"` becomes `MinnesotaPrior()`). Pass the object for full control.

Immutable. No state, no side effects.

## Priors

```python
from litterman.priors import MinnesotaPrior

prior = MinnesotaPrior()
prior = MinnesotaPrior(tightness=0.2, decay="harmonic", cross_shrinkage=0.5)
```

Pydantic models with documented defaults. Hyperparameters:

- `tightness` (lambda) — overall shrinkage toward the prior mean.
- `decay` — how coefficients shrink on longer lags (`"harmonic"` or `"geometric"`).
- `cross_shrinkage` — shrinkage on other variables' lags vs own lags.
- Prior mean is random walk (1.0) for own first lag by default.

## Estimation

```python
from litterman.samplers import NUTSSampler

result: FittedVAR = spec.fit(data)
result: FittedVAR = spec.fit(data, sampler=NUTSSampler(draws=2000, chains=4))
```

`NUTSSampler` is a Pydantic model whose fields pass through as kwargs to PyMC's sampling routine. When `lags` is a string, automatic lag selection (OLS-based AIC/BIC/HQ) runs before estimation.

Returns `FittedVAR` (immutable).

## FittedVAR

Holds the reduced-form posterior. Exposes:

```python
# Posterior coefficient access
result.coefficients   # posterior draws of B matrices
result.intercepts     # posterior draws of intercept vectors
result.sigma          # posterior draws of residual covariance

# Forecasting (no identification needed)
fcast = result.forecast(steps=8)
fcast = result.forecast(steps=8, exog_future=future_data)

# Identification
identified = result.identification_strategy(Cholesky(ordering=["oil", "gdp", "inflation"]))
```

`exog_future` is required if and only if the model was fitted with exogenous variables.

## Identification

```python
from litterman.identification import Cholesky, SignRestriction

identified: IdentifiedVAR = result.identification_strategy(Cholesky(ordering=[...]))
```

`identification_strategy()` returns a new `IdentifiedVAR` (immutable). The same `FittedVAR` can produce multiple `IdentifiedVAR` instances with different schemes.

v1 identification schemes: Cholesky, sign restrictions.

## IdentifiedVAR

All structural post-estimation methods live here:

```python
irf = identified.impulse_response(horizon=20)
irf = identified.impulse_response(horizon=20, shock="gdp")
irf = identified.impulse_response(horizon=20, shock="gdp", response="inflation")

fevd = identified.fevd(horizon=20)
hd = identified.historical_decomposition()
```

`impulse_response` returns all variable pairs by default. Optional `shock` and `response` arguments filter to specific pairs.

## Result Objects

All post-estimation methods return result objects inheriting from `VARResultBase`:

```python
class VARResultBase(BaseModel):
    idata: xr.Dataset

    def median(self) -> pd.DataFrame: ...
    def hdi(self, prob: float = 0.89) -> pd.DataFrame: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def plot(self) -> ...: ...  # abstract — each subclass renders differently
```

Subclasses: `IRFResult`, `FEVDResult`, `ForecastResult`, `HistoricalDecompositionResult`.

The `.idata` property exposes the raw xarray Dataset for custom analysis.

Default HDI probability is 0.89 (Bayesian convention, configurable).

## Plotting

```python
# Convenient default
irf.plot()

# Full control
from litterman.plotting import plot_irf, plot_fevd, plot_forecast, plot_historical_decomposition
fig, axes = plot_irf(irf, variables=["gdp"], figsize=(12, 8))
```

`.plot()` on result objects delegates to the corresponding module function with defaults. Import the function directly for full matplotlib control.

Plot types:

- IRF: credible bands per variable pair.
- FEVD: stacked area chart.
- Forecast: fan chart with credible bands.
- Historical decomposition: stacked contribution chart.

## Lag Selection (standalone)

```python
from litterman import select_lag_order

ic = select_lag_order(data, max_lags=12)
ic.aic   # int — optimal lag order by AIC
ic.bic   # int — optimal lag order by BIC
ic.hq    # int — optimal lag order by Hannan-Quinn
ic.summary()  # table of all criteria by lag order
```

Uses OLS-based information criteria (fast). Also called internally when `VAR(lags="bic")`.

## Runtime Type Checking

```python
import litterman
litterman.enable_runtime_checks()  # wraps public API with beartype
```

Intended to be on in tests, off in production.

## v1 Scope

Included:

- VAR specification and estimation (Minnesota prior, NUTS sampler via PyMC)
- Automatic lag selection (AIC, BIC, HQ via OLS)
- Forecasting with exogenous support
- Impulse response functions (Cholesky, sign restrictions)
- Forecast error variance decomposition
- Historical decomposition
- Built-in plotting for all result types
- ArviZ-compatible diagnostics via thin wrappers

Deferred to v2:

- Granger causality
- Conditional forecasting / scenario analysis
- Long-run restrictions (Blanchard-Quah)
- External instruments / proxy SVAR
- Additional priors (Normal-Wishart, steady-state)

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

# Coefficients
result.coefficients
result.sigma

# Forecast
fcast = result.forecast(steps=8)
fcast.median()
fcast.hdi(prob=0.89)
fcast.plot()

# Structural analysis
identified = result.identification_strategy(Cholesky(ordering=["oil", "gdp", "inflation"]))

irf = identified.impulse_response(horizon=20)
irf.plot()

fevd = identified.fevd(horizon=20)
fevd.plot()

hd = identified.historical_decomposition()
hd.plot()

# Escape hatch
irf.idata  # raw xarray Dataset
```
