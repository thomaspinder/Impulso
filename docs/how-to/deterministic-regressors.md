# Deterministic Regressors for Climate VARs

Climate series arrive with structure that is not dynamics. A monthly
temperature record for Berlin swings by twenty degrees between January and
July, drifts upward across four decades, and jumps the day the station moved.
None of that is the lag structure a VAR exists to estimate, and a VAR fitted on
the raw series spends its coefficients re-learning the calendar.

This guide shows how to build that structure as *deterministic regressors* —
total functions of a timestamp, with no data in them — and feed them to the
model through the exogenous block.

## Two routes, and when to take each

**Transform it away.** Subtract a month-of-year climatology, standardise, and
fit the VAR on anomalies:

```python
climatology = raw.groupby(raw.index.month).transform("mean")
anomalies = raw - climatology
anomalies = (anomalies - anomalies.mean()) / anomalies.std()
```

This is what the [conjugate VAR tutorial](../tutorials/conjugate-var.py) does.
It works with both estimators, keeps the model small, and gives impulse
responses that read in standard deviations. Its cost is that the seasonal
adjustment is a point estimate: its uncertainty never reaches the posterior,
and you cannot ask how large the annual cycle is or whether it changed.

**Model it.** Put the trend, cycle and breaks in the exogenous block. The
coefficients get posteriors you can read and plot, the seasonal uncertainty
propagates into the forecast bands, and a level-shift coefficient becomes a
dynamic multiplier — the full propagated path of the regime change, not just
its impact. The cost is that only the NUTS estimator (`impulso.VAR`) consumes
exogenous regressors, and each column is a parameter per variable.

The rest of this page is the second route.

## Building a design

The four term types live in `impulso.deterministic`.

A **trend** counts periods elapsed since the start of the sample:

```python
from impulso import Trend

Trend(degree=1, scale=120.0)   # linear, in units of decades on monthly data
Trend(degree=2, scale=120.0)   # adds a squared term
```

`scale` is not cosmetic. The exogenous coefficients carry a fixed
`Normal(0, 1)` prior, so an unscaled 540-month trend reaching 539 puts the
prior in a fight with the data. Divide by a fixed, interpretable constant —
periods per decade — and the coefficient reads as "change per decade".

:::{admonition} Keep `scale` independent of the sample
:class: warning
`scale` must not depend on the sample length. `scale=len(index)` breaks the
continuation property below: the design you extend with is no longer the design
you fitted.
:::

**Fourier harmonics** represent a smooth cycle of known length at two
coefficients per harmonic pair:

```python
from impulso import Fourier

Fourier(period=12, order=2)    # annual cycle on monthly data, 4 columns
Fourier(period=4, order=1)     # annual cycle on quarterly data, 2 columns
```

The cycle length is always explicit — nothing here infers a period from the
data. At most `period / 2` harmonic pairs are identified; more is rejected at
construction.

**Seasonal dummies** are the non-parametric alternative — a free coefficient
per calendar unit:

```python
from impulso import SeasonalDummies

SeasonalDummies(season="month")       # 11 columns (January dropped)
SeasonalDummies(season="quarter")     # 3 columns
SeasonalDummies(season="dayofweek")   # 6 columns (levels 0-6, Monday = 0)
```

One level is always dropped, because every Impulso estimator fits an
unconditional intercept and a full set of indicators sums to it. Use
`reference=` to choose which.

**Break dummies** mark a known, dated discontinuity:

```python
from impulso import BreakDummy

BreakDummy(date="1991-06-15")                  # level shift from that date on
BreakDummy(date="1991-06-15", kind="pulse")    # that period only
```

Compose them into a `DeterministicDesign`:

```python
from impulso import BreakDummy, DeterministicDesign, Fourier, Trend

design = DeterministicDesign(
    terms=[
        Trend(degree=1, scale=120.0),
        Fourier(period=12, order=2),
        BreakDummy(date="1991-06-15"),
    ],
    freq="MS",
)
frame = design.build(anomalies.index)
```

## The column-name contract

`design.column_names` is knowable before any index exists, and it is exactly
what `build` and `extend` emit, in that order:

| Term | Columns |
| --- | --- |
| `Trend(degree=3)` | `trend`, `trend_squared`, `trend_cubed` |
| `Fourier(period=12, order=2)` | `sin(1,12)`, `cos(1,12)`, `sin(2,12)`, `cos(2,12)` |
| `SeasonalDummies(season="month")` | `month_2` … `month_12` |
| `SeasonalDummies(season="quarter")` | `quarter_2`, `quarter_3`, `quarter_4` |
| `SeasonalDummies(season="dayofweek")` | `dow_1` … `dow_6` |
| `BreakDummy(date="1991-06-15")` | `level_1991-06-15` |
| `BreakDummy(date="1991-06-15", kind="pulse")` | `pulse_1991-06-15` |

Those names travel: they become `exog_names` on `VARData`, and PyMC labels the
`exog` coordinate of `B_exog` with them, so the posterior comes back
self-describing.

## Composing into VARData

There is no special constructor — concatenate and hand the column names over:

```python
import pandas as pd
from impulso import VARData

frame = pd.concat([anomalies, design.build(anomalies.index)], axis=1)
data = VARData.from_df(
    frame,
    endog=list(anomalies.columns),
    exog=design.column_names,
)
```

### The no-NaN invariant

Deterministic terms are total functions of a timestamp: they cannot produce a
missing value. So if `VARData` rejects your frame with

```
ValueError: exog contains NaN or Inf values
```

the design and the endogenous block are misaligned — `pd.concat` filled the
holes. The fix is the order of operations:

1. Transform the endogenous data (differences, logs, anomalies).
2. `dropna()`.
3. Build the design **on the index that survived**.
4. Concatenate.

Never `dropna()` after the concatenation: that silently drops observations to
paper over a misalignment.

Note also that `VAR.fit` discards the first `p` rows of both blocks to form the
lag matrices, so a design column that is only non-zero in the first few periods
never reaches the model.

## Fitting and forecasting

The payoff is that one design object serves estimation and forecasting. Because
elapsed time is anchored to the calendar rather than to row position, the block
`extend` writes for the future is exactly the block `build` would have written
had the sample run longer — the *continuation property*:

```python
design.build(index[: T + h]).iloc[T:] == design.extend(index[:T], h)
```

`exog_future` wraps that up and, crucially, reorders the columns **by name** to
match the fitted `exog_names`. `forecast` indexes the block positionally, so a
permuted design would otherwise be a silently wrong forecast rather than an
error.

The following recipe is exercised end to end in
`tests/test_deterministic.py::test_deterministic_design_end_to_end`:

```python
import numpy as np
import pandas as pd

from impulso import VAR, DeterministicDesign, Fourier, NUTSSampler, Trend, VARData

# A short monthly two-variable sample standing in for climate anomalies.
rng = np.random.default_rng(7)
index = pd.date_range("2000-01-01", periods=120, freq="MS")
endog = pd.DataFrame(
    rng.standard_normal((len(index), 2)).cumsum(axis=0) * 0.1,
    index=index,
    columns=["temperature", "precipitation"],
)

# One design, used for estimation and for forecasting.
design = DeterministicDesign(
    terms=[Trend(degree=1, scale=120.0), Fourier(period=12, order=1)],
    freq="MS",
)

frame = pd.concat([endog, design.build(index)], axis=1)
data = VARData.from_df(frame, endog=list(endog.columns), exog=design.column_names)

fitted = VAR(lags=1).fit(data, sampler=NUTSSampler(draws=50, tune=50, chains=2, cores=1, random_seed=42))

# The posterior labels B_exog with the design's own column names.
assert list(fitted.idata.posterior["B_exog"].coords["exog"].values) == design.column_names

forecast = fitted.forecast(steps=12, exog_future=design.exog_future(fitted, 12))

assert forecast.median().shape == (12, 2)
```

The same array feeds `conditional_forecast` and `structural_scenario`.

## The estimator boundary

:::{admonition} The conjugate estimator refuses exogenous regressors
:class: warning
`ConjugateVAR` does not consume exogenous regressors. Handing it a `VARData`
that carries a deterministic design raises:

> ConjugateVAR does not support exogenous regressors: the conjugate engine
> estimates endogenous dynamics only, and silently ignoring the exog block
> would corrupt downstream forecasts. Drop exog from VARData or use the
> PyMC/NUTS estimator (impulso.VAR), which consumes it.

Two ways forward. Fit with `impulso.VAR` and keep the design in the model — the
coefficients get posteriors. Or residualise first: regress each endogenous
series on `design.build(index)` by OLS and fit the conjugate VAR on the
residuals, accepting that the deterministic part's uncertainty is discarded
rather than propagated.
:::

## Collinearity rules

`build` checks the design's rank against the intercept every estimator fits and
refuses a deficient one, because its coefficients are not identified. The
common causes, all named in the error message:

- `SeasonalDummies(drop_first=False)` — the levels sum to the intercept.
- `Fourier(period=12, order=6)` — at exactly the Nyquist limit the top sine is
  identically zero at every sampled point.
- Seasonal dummies **and** harmonics of the same cycle (`season="month"` with
  `Fourier(period=12, ...)`) — the harmonics live in the span of the level
  indicators. Keep one or the other.
- A level break at or before the first observation (constant in-sample) or
  after the last (never occurs); a pulse break on a date the index does not
  contain (zero everywhere).
- Fewer observations than columns.

`extend` deliberately does **not** rank-check: a three-step forecast block
cannot span twelve month dummies, and nothing is being estimated from those
rows.

## Reading the coefficients

`B_exog` is labelled by design column, so posterior summaries name themselves.
For a level break the more interesting object is the dynamic multiplier:
`fitted.dynamic_multiplier(cumulative=True)` propagates a permanent unit step
through the lag dynamics, which for a `level_*` column *is* the full adjustment
path of the regime shift — not merely its impact-period effect. See
[`FittedVAR`](../reference/fitted.md) for the method and
[the result objects](../reference/results.md) for what it returns.

## Scope

This module does calendar arithmetic and nothing else. Holiday and business
calendars, interaction terms, slope breaks and inferring cycle lengths from the
data are all out of scope. So is anything that is *data* rather than
arithmetic: solar forcing, ENSO indices and CO2 concentrations are covariates
to be loaded from a dataset, not generated from a timestamp.

One approximation is worth flagging. On a daily index, `Fourier(period=365.25)`
counts whole days, so the harmonic drifts against the calendar within a leap
cycle. It is fine over multi-year samples and exact for monthly, quarterly and
annual sampling, where a period ordinal *is* the calendar unit.
