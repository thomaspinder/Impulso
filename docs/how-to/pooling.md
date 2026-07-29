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
per-period held-out log score — the shape of the output, with your own data's
numbers in it:

| | weight | log_score | mean_log_score | rank |
|---|---|---|---|---|
| var4 | 0.62 | ... | ... | 1 |
| var2 | 0.38 | ... | ... | 2 |
| conjugate | 0.00 | ... | ... | 3 |

`pool.log_scores` is the full matrix behind those totals — one row per held-out
date, one column per model — and `pool.to_dataframe()` adds the pooled
predictive's own score alongside them. `pool.plot()` draws the weights as a
ranked bar chart.

## Stacking versus log-score weights

The two methods answer different questions. When one candidate simply forecasts
better than the others, they agree and both hand it the weight. Take a tightly
shrunk VAR(1) against a loosely shrunk VAR(4), fitted on the same synthetic
series and scored over sixteen held-out quarters:

```python
import numpy as np
import pandas as pd
from impulso import VARData, pool_forecasts
from impulso.conjugate import ConjugateVAR
from impulso.priors import NIWPrior

rng = np.random.default_rng(0)
T, H = 240, 16
A = np.array([[0.55, 0.15], [-0.20, 0.45]])
y = np.zeros((T + H, 2))
for t in range(1, T + H):
    calm = (t // 4) % 2 == 0
    y[t] = A @ y[t - 1] + rng.standard_normal(2) * (
        np.array([0.3, 1.6]) if calm else np.array([1.6, 0.3])
    )
index = pd.date_range("1980-01-01", periods=T + H, freq="QS")
frame = pd.DataFrame(y, columns=["output", "prices"], index=index)

train = VARData.from_df(frame.iloc[:T], endog=["output", "prices"])
held_out = VARData.from_df(frame.iloc[T:], endog=["output", "prices"])
candidates = {
    "var1_tight": ConjugateVAR(lags=1, prior=NIWPrior(tightness=0.05), draws=500, seed=0).fit(train),
    "var4_loose": ConjugateVAR(lags=4, prior=NIWPrior(tightness=1.0), draws=500, seed=1).fit(train),
}

stacked = pool_forecasts(candidates, held_out, method="stacking", seed=0)
log_scored = pool_forecasts(candidates, held_out, method="log_score", seed=0)
```

The two models score -56.7 and -51.2 over the window, and both rules reach the
same verdict:

| | `var1_tight` | `var4_loose` | pooled log score |
|---|---|---|---|
| `method="stacking"` | 0.000 | 1.000 | -51.2 |
| `method="log_score"` | 0.004 | 0.996 | -51.2 |

This is the common case: one model dominates, the pool finds it, and the pooled
score matches the winner's. Pooling has told you something useful — that the
second candidate adds nothing — even though it did not improve the forecast.

The rules come apart when the candidates are genuinely **complementary**: each
one better over a different part of the held-out window, neither better
throughout. Stacking scores the *mixture*, so it can hold both models at
non-trivial weight and reach a score above anything either achieves alone — a
mixture covers regions that no single member covers. Log-score weights compare
the models one at a time and take a softmax of their totals, so they concentrate
on whichever model has the better total even when a blend would score higher.
Whether the resulting pool beats the best single model depends on how close the
totals are; the point is that log-score weights are not trying to maximise the
pooled score, and stacking is.

Log-score weights also concentrate harder the longer the holdout, because total
scores diverge roughly linearly in the number of held-out periods. That is a
property of the rule, not a bug: it is the behaviour you want if you believe one
candidate is correct and want the evidence to say which. Reach for stacking when
you want the best combined forecast instead.

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
