# Testing for Stationarity and Cointegration

Before specifying a VAR you need to decide whether to fit it in levels or in
differences, and whether the series share a long-run relationship worth
keeping. Impulso ships the standard pretests for that decision. They report;
they do not decide.

## Install the extra

The diagnostics depend on `statsmodels`, which is not part of the core
install:

```
pip install "impulso[diagnostics]"
```

or, with uv:

```
uv add "impulso[diagnostics]"
```

Calling any of these functions without it raises an `ImportError` naming the
extra.

## Unit-root tests

`adf_test` runs the Augmented Dickey-Fuller test on every column. Its null
hypothesis is that the series **has a unit root**, so a small p-value argues
for stationarity.

```python
from impulso import adf_test

result = adf_test(data)          # VARData, DataFrame, or Series
result.summary()                 # one row per variable
result.conclusions               # {"gdp": "non-stationary", ...}
```

`kpss_test` runs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test, whose
null is the **opposite**: that the series is stationary. A small p-value
there argues for a unit root.

```python
from impulso import kpss_test

kpss_test(data).conclusions
```

The `conclusion` column already accounts for the flipped null, so it reads
the same way for both tests: `"stationary"` or `"non-stationary"`.

Running both is standard practice. ADF has poor power against persistent
alternatives, so it fails to reject far more often than the data warrant;
KPSS gives an independent read. When the two agree you have a reasonably
firm answer. When they disagree, see
[Stationarity pitfalls in climate data](climate-pitfalls.md).

### Deterministic terms

Both tests need to know what deterministic behaviour to allow under the
alternative. Get this wrong and the test answers a different question than
you asked:

```python
adf_test(data, regression="ct")    # constant and linear trend
kpss_test(data, regression="ct")   # test trend stationarity
```

A trending series tested with `regression="c"` will almost always look
non-stationary, because ADF has no way to express "trending but mean
reverting around the trend".

### KPSS p-values are bounded

KPSS p-values are interpolated from a published table and clipped to
`[0.01, 0.10]`. When the clip binds, the `pvalue_bounded` column is `True`
and the reported figure is a bound, not an estimate:

```python
table = kpss_test(data).summary()
table.loc[table["pvalue_bounded"], ["statistic", "pvalue"]]
```

Impulso catches the underlying statsmodels warning and turns it into that
column, so the tests never spray warnings into your notebook.

## Integration order

`integration_order` automates the bookkeeping: for each variable it tests the
level, differences, re-tests, and stops when ADF rejects a unit root.

```python
from impulso import integration_order

orders = integration_order(data, max_order=2)
orders.order        # {"gdp": 1, "rate": 1, "spread": 0}
orders.d_max        # 1 — the highest order in the system
orders.inconclusive # variables where the answer is not clean
orders.summary()    # every test at every differencing level
```

Three things to know:

- ADF drives the stopping rule; KPSS is run alongside at every level and
  recorded in the `joint_status` column as one of `stationary`, `unit_root`,
  `conflicting` (both reject), or `inconclusive` (neither rejects).
- `regression` applies to the **level** test only. Differencing removes a
  linear trend, so differenced series are always tested with a constant.
- `inconclusive` lists variables that were still non-stationary at
  `max_order`, or whose two tests conflicted at the level where the search
  stopped. Treat their `order` as a placeholder and look at the table.

`d_max` is the quantity a Toda-Yamamoto style procedure needs, so it is worth
recording alongside the model even when every series turns out to be I(1).

## Cointegration rank

If several series are individually I(1), they may still move together.
`johansen_test` reports the cointegration rank.

```python
from impulso import select_lag_order, johansen_test

p = select_lag_order(data, max_lags=8).bic
result = johansen_test(data, det_order=0, k_ar_diff=p - 1)

result.rank             # rank by the trace statistic
result.rank_max_eigen   # rank by the maximum-eigenvalue statistic
result.summary()        # both sequences, statistic against critical value
```

`k_ar_diff` counts lagged *differences*, so it is `p - 1` for a VAR(p) in
levels — pick `p` first, then subtract one.

Reading the rank, with `n` series:

| Rank | Meaning |
| --- | --- |
| `0` | No cointegration. The series share no long-run relationship. |
| `1` to `n - 1` | Common stochastic trends. Differencing every series discards the long-run relationships. |
| `n` | The system is already stationary in levels. |

The trace and maximum-eigenvalue tests can disagree; `rank` reports the trace
answer by convention, because it is the more robust of the two in small
samples. Check `rank_max_eigen` before relying on it.

Only critical values are tabulated for this test, not p-values, so `alpha`
must be `0.10`, `0.05`, or `0.01`.

## Recording the result

The tables are the deliverable, not the verdicts. Keep them:

```python
adf = adf_test(data, regression="ct")
kpss = kpss_test(data, regression="ct")
orders = integration_order(data)
rank = johansen_test(data, k_ar_diff=p - 1)

diagnostics = {
    "adf": adf.summary(),
    "kpss": kpss.summary(),
    "integration_order": orders.summary(),
    "d_max": orders.d_max,
    "cointegration_rank": rank.summary(),
}
```

Anyone reading your results later needs to know which pretests you ran, at
what significance level, with which deterministic terms — and where the tests
disagreed.
