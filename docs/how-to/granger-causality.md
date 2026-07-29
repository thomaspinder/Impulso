# Granger Causality and Toda-Yamamoto

Granger causality asks whether one variable's past improves the prediction
of another beyond that other variable's own past. Impulso answers it as a
posterior over a *magnitude* rather than as a hypothesis test, and it will
not hand you a probability that there is no causality — the last section
explains why not.

## Query a fitted model

The query lives on the reduced-form object. No identification scheme is
involved: it reads the reduced-form coefficients directly.

```python
from impulso import VAR, VARData

data = VARData.from_df(df, endog=["co2", "temperature"])
fitted = VAR(lags=2).fit(data)

result = fitted.granger_causality("co2", "temperature")
result.summary()
```

The pair is ordered. `granger_causality("co2", "temperature")` tests the
lags of carbon dioxide in the temperature equation; swapping the arguments
asks the other question, and the two answers are unrelated.

## Read the summary

```
       median  hdi_lower  hdi_upper
term
L1       0.31       0.18       0.44
L2      -0.06      -0.19       0.07
norm     0.32       0.20       0.45
```

One row per tested lag, then the headline. `norm` is the Euclidean norm of
the tested coefficients, `‖b‖ = sqrt(sum_k b_k²)`, computed draw by draw —
so its posterior is a posterior for the joint strength of the whole lag
block, not a summary of the per-lag medians. `hdi_lower` and `hdi_upper`
bound the highest-density interval (HDI), 89% by default; pass
`summary(prob=0.95)` for a different mass.

Keeping the per-lag rows matters: a strong first lag with an offsetting
second lag is a different finding from two moderate ones, and the norm alone
cannot tell them apart.

By default the draws are standardised — multiplied by
`sd(cause) / sd(effect)`, both sample standard deviations of the estimation
data — so a magnitude reads as standard deviations of the effect per
standard deviation of the cause. The factor is on the result as `scale`.
Pass `standardize=False` for raw coefficient units.

## Put a number on "practically zero"

Supply a region of practical equivalence (ROPE) — the magnitude below which
you would call the relationship negligible — and the result also reports
`p_rope`:

```python
result = fitted.granger_causality("co2", "temperature", rope=0.05)
result.p_rope        # P(||b|| < 0.05 | data)
```

There is deliberately no default. The ROPE is where your judgement about
what counts as a small effect enters, and it belongs in the write-up next to
the number it produced. In standardised units it is read as "a shift of one
standard deviation in the cause moves the effect by less than `rope`
standard deviations".

## Why there is no probability of no causality

The obvious thing to want is `P(no causality | data)`. Impulso does not
report it, because under the priors it fits that quantity is zero by
construction and would be zero whatever the data said.

Every coefficient in Impulso has a continuous prior — Normal under
`MinnesotaPrior`, Normal-Inverse-Wishart under `NIWPrior`. A continuous
distribution assigns probability zero to any single point, so
`P(b = 0) = 0` before seeing the data. Conditioning cannot raise a
probability from zero. A model that can answer the question needs a prior
that puts a lump of mass on the null itself — a spike-and-slab, or an
edge-inclusion prior over which coefficients are present at all — which is a
different model, not a different summary of this one.

So the honest reformulation is the ROPE one: not "is it exactly zero?" but
"is it smaller than I would care about?". That is what `p_rope` answers, and
it is only meaningful because you chose the threshold. Report `p_rope`
together with the `rope` that produced it; alone it is uninterpretable.

## Toda-Yamamoto for integrated systems

Standard Granger inference assumes the VAR's asymptotics are the stationary
ones. On integrated series they are not, and the usual fix — difference
everything first — changes the question to one about growth rates and
discards any long-run relationship.

Toda and Yamamoto (1995) offer a way around it: fit the VAR in levels with
`p + d` lags, where `p` is the lag order you would have chosen and `d` the
highest integration order in the system, then test only the first `p`. The
extra `d` lags are not part of the hypothesis. They exist to restore the
standard asymptotics.

```python
from impulso import toda_yamamoto

result = toda_yamamoto(data, "co2", "temperature", lags=2, rope=0.05)

result.n_lags_tested       # 2 — what the answer is about
result.n_lags_fitted       # 3 — what was estimated
result.augmentation        # 1
result.augmentation_source # "integration_order"
result.integration_order_result.summary()
```

The test lag order is never silently changed to match the fitted one. Both
numbers are on the result, and only the tested lags appear in `summary()`.

`d` comes from `integration_order` unless you pass it. That call needs the
optional `diagnostics` extra (`pip install "impulso[diagnostics]"`).

### When it refuses

`integration_order` lists a variable in `inconclusive` when it is still
non-stationary at `max_order`, or when the Augmented Dickey-Fuller (ADF) and
Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests disagreed where the search
stopped. In that case `d_max` is a floor rather than a finding: the true
augmentation may be higher, and an under-augmented Toda-Yamamoto test is
invalid.

Rather than guess, `toda_yamamoto` raises, naming the variables. Read the
table, decide, and pass the augmentation yourself:

```python
from impulso import integration_order

integration_order(data).summary()          # look at every level, per variable
toda_yamamoto(data, "co2", "temperature", lags=2, d=2)
```

Passing `d=` skips the diagnostics entirely — the decision is recorded as
`augmentation_source="user"` — so it also works without `statsmodels`
installed. `d=0` is legitimate: it is the plain Granger test, and it is what
the diagnostics themselves return for a stationary system.

If you already ran the diagnostics, hand them over instead of paying for
them twice:

```python
diagnostics = integration_order(data)
toda_yamamoto(data, "co2", "temperature", lags=2, integration_order_result=diagnostics)
```

### The manual route

`toda_yamamoto` fits with the conjugate estimator, which draws in closed
form — augmentation inflates the lag order, and this keeps that cheap. It
therefore does not accept exogenous regressors, the NUTS estimator, or a
stochastic-volatility process. For any of those, run the same three steps by
hand:

```python
from impulso import VAR, integration_order

d = integration_order(data).d_max          # check .inconclusive first
fitted = VAR(lags=2 + d).fit(data)
fitted.granger_causality("co2", "temperature", test_lags=2)
```

`test_lags=2` is the whole trick: 2 + `d` lags are estimated, two are
tested. The result records the untested lags as `augmentation`.

## A worked example, and what it does not license

Take deseasonalised Mauna Loa carbon dioxide and a global mean surface
temperature series, both annual, both in levels:

```python
from impulso import VARData, integration_order, toda_yamamoto

data = VARData.from_df(climate_df, endog=["co2", "temperature"])
integration_order(data).summary()          # both I(1), typically

forward = toda_yamamoto(data, "co2", "temperature", lags=2, rope=0.05)
reverse = toda_yamamoto(data, "temperature", "co2", lags=2, rope=0.05)

forward.median(), forward.p_rope
reverse.median(), reverse.p_rope
```

Suppose the forward direction comes back with a large norm and a `p_rope`
near zero, and the reverse with a small one. Here is precisely what may be
said: *past carbon dioxide improves the prediction of temperature beyond
temperature's own past, in this bivariate system, at these lags.* Nothing
more. In particular:

**Granger causality is predictive precedence, not intervention.** It ranks
information sets, not policies. It cannot tell you what temperature would do
under a counterfactual emissions path — that is what
`counterfactual` and `structural_scenario` are for, and they need an
identification scheme.

**The bivariate system omits the other forcings.** Solar variability,
volcanic and anthropogenic aerosols, and El Niño-Southern Oscillation all
drive temperature and are correlated with the industrial era. A driver
omitted from the system can manufacture apparent causality between the two
variables that remain, or mask it.

**The physical coupling runs both ways.** Carbon dioxide forces temperature
radiatively; temperature drives carbon dioxide back through ocean solubility
and the response of respiration and the terrestrial carbon sink. A test that
finds one direction "stronger" has found something about the sampling
frequency and the lag structure, not about which mechanism is real.

**Aggregation distorts lead-lag structure.** Annual means, and the smoothing
inherent in ice-core and other proxy records, compress the timescales the
test measures. A mechanism that operates within a year is invisible to
annual data, and the smoothing can shift apparent leads by whole periods.

The general warnings about unit-root testing on climate series apply here
too, because Toda-Yamamoto consumes an integration order: see
[Stationarity pitfalls in climate data](climate-pitfalls.md).

## What to record

- The **ordered pair** and the **direction**, both ways round if you ran
  both. "X and Y are Granger-related" is not a result.
- The **`rope`** alongside every `p_rope`. The number means nothing without
  its threshold.
- **`n_lags_tested` and `n_lags_fitted`**, not just the lag order. Under
  augmentation these differ, and the difference is the point.
- **`augmentation_source`**, and the integration-order table when the
  diagnostics were consulted.
- Whether the magnitudes are **standardised**, and the `scale` if not
  obvious. Under lag augmentation the model is in levels, so the standard
  deviations carry the series' trends and magnitudes compare best within one
  fit.
