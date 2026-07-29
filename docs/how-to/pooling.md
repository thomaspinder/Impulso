# Pooling Several Models

When two lag orders, two priors, or two estimators all look defensible, you do
not have to pick one. Score them on data they were not fitted to, and let the
scores decide how much of each to keep.

## Split the sample

The held-out window must be the periods immediately following the estimation
sample. Split once, keep the split fixed, and fit every candidate on the same
training data:

```python
import pandas as pd
from impulso import VAR, ConjugateVAR, VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
train = VARData.from_df(df.iloc[:-12], endog=["gdp", "inflation", "rate"])
holdout = VARData.from_df(df.iloc[-12:], endog=["gdp", "inflation", "rate"])
```

## Fit the candidates

Any mix of estimators works — the pool only needs `FittedVAR` objects that share
their variables and their sample end:

```python
fits = {
    "var2": VAR(lags=2, prior="minnesota").fit(train),
    "var4": VAR(lags=4, prior="minnesota").fit(train),
    "conjugate": ConjugateVAR(lags=4).fit(train),
}
```

## Estimate the weights

```python
from impulso import pool_forecasts

pool = pool_forecasts(fits, holdout, method="stacking", seed=0)
pool.summary()
```

`summary()` returns one row per model, heaviest weight first, with the total and
per-period held-out log score:

| | weight | log_score | mean_log_score | rank |
|---|---|---|---|---|
| var4 | 0.62 | -131.1 | -10.9 | 1 |
| var2 | 0.38 | -153.8 | -12.8 | 2 |
| conjugate | 0.00 | -204.5 | -17.0 | 3 |

`pool.log_scores` is the full matrix behind those totals — one row per held-out
date, one column per model — and `pool.to_dataframe()` adds the pooled
predictive's own score alongside them. `pool.plot()` draws the weights as a
ranked bar chart.

## Stacking versus log-score weights

The two methods answer different questions, and on genuinely complementary
models they disagree sharply. Take two models with identical mean forecasts but
mirrored shock scales — one tight on the first variable and wide on the second,
the other the reverse — and a holdout that alternates between the two regions:

```python
stacked = pool_forecasts(fits, holdout, method="stacking", seed=0)
log_scored = pool_forecasts(fits, holdout, method="log_score", seed=0)
```

| | weights | pooled log score |
|---|---|---|
| `method="stacking"` | 0.50 / 0.50 | -40.8 |
| `method="log_score"` | 0.00 / 1.00 | -119.0 |

Neither model scores better than -131.1 on its own. Stacking splits the weight
evenly and the pooled density scores -40.8, because it maximises the score of
the *mixture* and a mixture covers both regions. Log-score weights compare the
models one at a time, so they hand everything to the marginally better single
model and pool no better than that model does. Log-score weights are the right
choice when you believe one candidate is correct and want the evidence to say
which; stacking is the right choice when you want the best combined forecast.

Log-score weights also collapse harder the longer the holdout: total scores
diverge linearly, so the softmax concentrates on one model. That is a property
of the rule, not a bug.

## Forecast with the weights

`pool.holdout_predictive` covers the *held-out window* — useful for plotting the
combination against what actually happened, useless as a forecast, since those
dates have already occurred. For a real forecast, refit on the full sample and
apply the frozen weights:

```python
full = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
refits = {
    "var2": VAR(lags=2, prior="minnesota").fit(full),
    "var4": VAR(lags=4, prior="minnesota").fit(full),
    "conjugate": ConjugateVAR(lags=4).fit(full),
}
forecasts = {label: fit.forecast(steps=8) for label, fit in refits.items()}

combined = pool.combine(forecasts, seed=1)
combined.median()
combined.hdi(0.89)
```

`combine` needs the same model labels and variables as the pool, and all
forecasts must run to the same horizon — but that horizon need not be the one
the weights were scored over. It also insists on density-mode forecasts
(`include_shock_uncertainty=True`, the default), because a mean forecast has no
predictive density to pool.

## What the scores are and are not

- Each horizon's predictive density is a **Gaussian matched to the forecast
  draws**, joint across variables. The true posterior predictive is a
  heavier-tailed mixture over draws, so the scores are comparable across models
  rather than exact. The approximation degrades with few draws, stochastic
  volatility, and fat tails. Pass `density="diagonal"` to score each variable on
  its own marginal when the joint covariance is near-singular.
- Scores are **summed over horizons 1 to H from one fixed origin**. This is not
  a joint-path density and not a rolling-origin evaluation.
- Weights are **static** — one per model, fixed across horizons and time. A
  model that only wins at long horizons cannot be given a horizon-specific
  weight.
- A short holdout makes the weights noisy. Twelve quarters is thin; two
  observations tell you almost nothing.
- Impulso checks that the holdout postdates every model's estimation sample. It
  cannot check that you did not look at the holdout while choosing the
  candidates — that part is on you.
- Nothing records how a series was transformed, so pooling models fitted on
  differently transformed data will produce meaningless scores. Keep the
  transformations identical across candidates.
