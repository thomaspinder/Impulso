# Stationarity Pitfalls in Climate Data

The unit-root tests in [Testing for stationarity and
cointegration](stationarity-testing.md) were developed for macroeconomic
series: a few hundred quarterly observations, no seasonality left after
adjustment, a plausible constant or linear trend. Climate series break most
of those assumptions at once. This page collects the failure modes and what
to do about each.

## Why climate series stress the tests

Four properties of climate data push the Augmented Dickey-Fuller (ADF) and
Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests into their weak regimes:

**Strong deterministic trends.** Forced warming produces a series with an
unmistakable upward slope. ADF with only a constant cannot represent
"trending but mean reverting", so it reports a unit root regardless of
whether the trend is stochastic or deterministic.

**Low power against persistent alternatives.** Ocean heat content, soil
moisture, and multi-decadal oscillation indices are highly persistent but
not necessarily integrated. ADF fails to reject a unit root for an AR(1)
with a coefficient of 0.95 far more often than not, at any sample size a
climate record realistically offers.

**Structural breaks.** Volcanic eruptions, regime shifts in observing
systems, and changes in instrumentation all inject level or variance shifts.
A level shift makes a stationary series look integrated to KPSS while ADF
may still reject — the textbook route to a `conflicting` verdict.

**Long memory.** Several climate series are better described as fractionally
integrated than as cleanly I(0) or I(1). The tests have no way to say so;
they answer the binary question they were asked.

The practical consequence: `conflicting` and `inconclusive` entries in the
`joint_status` column are the normal outcome for climate data, not a sign
that something went wrong. Read the table, do not read only the verdict.

## Trend stationarity versus difference stationarity

A series with an upward slope can be trend stationary (deterministic trend
plus stationary noise) or difference stationary (a stochastic trend, with or
without drift). The tests distinguish these poorly, and the choice changes
what a VAR fitted on the data actually means.

Test both ways before deciding:

```python
from impulso import adf_test, kpss_test

adf_test(data, regression="ct")     # allow a deterministic trend
kpss_test(data, regression="ct")    # test stationarity around that trend
```

Three routes are available once you have decided, and they are not
equivalent:

1. **Fit in levels with `regression="ct"` reasoning behind it.** Appropriate
   if you believe the trend is deterministic. The VAR sees the trending
   series; impulse responses are responses of the *level*, and they need not
   die out.
2. **Difference the series.** Appropriate if you believe the trend is
   stochastic. Impulse responses are now responses of the *growth rate*, and
   they do die out. This is a different question, answered about a different
   object — not a preprocessing detail.
3. **Keep levels and pass the trend as an exogenous regressor.** `VARData`
   accepts `exog`, so a deterministic time trend (or a forcing series) can
   enter the model directly, leaving the endogenous block to carry the
   dynamics.

There is a good argument that anthropogenically forced warming is closer to
trend stationary than to difference stationary — the trend has a physical
driver rather than being an accumulation of shocks. That argument, not the
p-value, is what should decide the specification. State which route you took
and why.

## Anomalies do not make a series stationary

Converting to anomalies (subtracting a climatological baseline, usually per
calendar month) removes the seasonal cycle. It is easy to assume this makes
the series stationary. It does not.

Anomalies remove the *seasonal* unit roots. They leave the zero-frequency
stochastic trend untouched, because subtracting a fixed monthly baseline is a
level shift per month, not a difference. Deseasonalised Mauna Loa carbon
dioxide is the clean illustration: the sawtooth is gone and the series is
still visibly I(1).

Test the anomalies, not the raw series and not your intuition:

```python
from impulso import integration_order

integration_order(anomalies, max_order=2).summary()
```

## Spurious regression

Two independent series that both trend upward will look strongly related.
Regress one on the other and you get a large coefficient, a high
goodness-of-fit, and residuals that are themselves integrated. Nothing about
the relationship is real.

This is the reason the pretests matter for a VAR and not just for a single
regression. A VAR fitted in levels on integrated series inherits the same
problem: the estimated dynamics can reflect shared trending rather than any
transmission mechanism, and standard inference on those coefficients is not
valid. Establishing the integration order first, and the cointegration rank
second, is what separates a real long-run relationship from a shared drift.

## Cointegration and what to do about it

If two climate series share a stochastic trend — cumulative emissions and
global mean surface temperature is the obvious pair, given the near-linear
relationship between cumulative carbon dioxide and warming — they are
cointegrated, and the cointegrating relationship is the physically meaningful
part.

```python
from impulso import johansen_test

result = johansen_test(data, det_order=0, k_ar_diff=p - 1)
result.rank
```

A rank of 1 or more is a warning about differencing. Differencing every
series removes the levels information, and the long-run relationship lives
entirely in the levels. You would be throwing away the part of the data that
answers the question.

**Impulso does not implement a vector error correction model (VECM).** That
is the textbook answer to a cointegrated system, and it is out of scope. Two
routes remain:

**Fit the VAR in levels.** This is the standard Bayesian position (Sims,
Stock and Watson): a VAR in levels is consistent whether or not the series
are cointegrated, and it neither imposes nor discards the cointegrating
relationships. The classical objection is about the validity of hypothesis
tests on the coefficients, which does not bite for posterior inference in the
same way. The Minnesota prior already shrinks each equation toward a random
walk, which is the right prior mean for integrated data. This is the
recommended default.

**Difference and accept the cost.** Legitimate if the long-run relationship
is not what you are asking about — if the question is genuinely about
short-run dynamics between growth rates. Say explicitly that the long-run
relationship was discarded, so that nobody reads a levels interpretation into
the results.

## What to record

Whatever you decide, the pretests are part of the specification, so record
them the way you would record the lag order:

- The **full tables**, not the verdicts. `adf.summary()`, `kpss.summary()`,
  `integration_order(...).summary()`, `johansen_test(...).summary()`.
- Every place the two tests **conflicted**, and what you did about it. The
  `inconclusive` list on `IntegrationOrderResult` is the short version.
- The **significance level** and the **deterministic terms** used. A verdict
  without its `alpha` and its `regression` is not reproducible.
- **`d_max`**, the highest integration order in the system. It is the
  augmentation term a Toda-Yamamoto style procedure needs, and it is far
  easier to carry forward now than to reconstruct later.
- The **cointegration rank** and the lag order it was conditioned on. The
  rank is not invariant to `k_ar_diff`.
