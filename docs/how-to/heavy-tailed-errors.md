# Heavy-Tailed Observation Errors

Macroeconomic samples contain observations that a Gaussian VAR cannot
accommodate: 2020Q2, the 2008 collapse, a devaluation, a data revision. Under
Gaussian errors a single such quarter pulls the coefficient estimates and
inflates the estimated covariance for the whole sample. Student-t observation
errors downweight it automatically, inside the model, with no dummies to pick
by hand.

## When to reach for it

- The residuals from a Gaussian fit have a few very large values rather than a
  uniformly wide spread.
- You would otherwise be adding outlier dummies, and would rather not choose
  the dates yourself.
- The estimated innovation covariance looks implausibly large relative to the
  bulk of the sample.

If instead the volatility level *drifts* over the sample — quiet decades and
turbulent ones — that is a stochastic-volatility problem, not a heavy-tails
problem. The two cannot currently be combined (see
[Limitations](#limitations)).

## Fitting

The string shorthand infers the degrees of freedom from the data:

```python
from impulso import VAR

fitted = VAR(lags=4, error_dist="student_t").fit(data)
```

Fixing them instead is the robust choice on short samples, where `nu` is only
weakly identified:

```python
from impulso import VAR, StudentT

fitted = VAR(lags=4, error_dist=StudentT(nu=5.0)).fit(data)
```

`nu` must be strictly greater than 2 — below that the t has infinite variance
and forecast bands, variance decompositions and the innovation covariance all
stop being defined. Values around 4–6 are aggressively robust; above roughly 30
the fit is indistinguishable from Gaussian.

## Reading the posterior for nu

The degrees of freedom land in the posterior as `nu` under **both**
parameterisations, so the same code reads them either way:

```python
import arviz as az

az.summary(fitted.idata, var_names=["nu"])
```

Under inference the free parameter is `nu_excess` and `nu = 2 + nu_excess`.
The shift means the prior has zero density at the boundary, so the sampler is
never dragged toward the infinite-variance edge. A posterior median below about
10 says the data genuinely want heavy tails; a median that has drifted up toward
the prior mean (22 by default) says the sample carries little information about
the tail, and a fixed `nu` is the more honest specification.

:::{admonition} If you see divergences
:class: tip
`nu` and the innovation scale trade off against each other, which can make the
posterior geometry awkward on short samples. Raise `target_accept` to 0.9, or
fix `nu` rather than inferring it.
:::

## What changes, and what does not

Under the t, `Ω = L Lᵀ` is the **scale** matrix, not the covariance. The
distinction propagates predictably:

| Quantity | Effect |
| --- | --- |
| `fitted.sigma()` | Returns Ω unchanged — the scale matrix, no longer the covariance |
| `fitted.innovation_covariance()` | New accessor; returns `nu/(nu−2)·Ω`, the actual second moment |
| `fitted.forecast()` | Innovations follow the t, so bands have fatter tails at the same interquartile width |
| `identified.impulse_response()` | A "unit shock" is one *scale* unit = `sqrt((nu−2)/nu)` unconditional sd |
| `identified.fevd()` | **Exactly unchanged** — shares are ratios, and the scale cancels |
| `identified.historical_decomposition()` | **Exactly unchanged** — shocks are backed out and re-propagated |
| `identified.counterfactual()` | **Exactly unchanged** for zero edits; non-zero `ShockPath` values are in scale units |
| `fitted.conditional_forecast()` | Raises `NotImplementedError` |
| `identified.structural_scenario()` | Raises `NotImplementedError` |

To restore the one-standard-deviation shock convention for impulse responses,
scale them by `sqrt(nu/(nu−2))`:

```python
import numpy as np

nu = fitted.idata.posterior["nu"].values          # (chain, draw)
irf = identified.impulse_response(horizon=20)
factor = np.sqrt(nu / (nu - 2.0))[:, :, None, None, None]
sd_units = irf.idata.posterior_predictive["irf"].values * factor
```

## Is it worth it? Comparing against the Gaussian fit

Both fits carry a pointwise log-likelihood, so ArviZ compares them directly:

```python
import arviz as az

from impulso import VAR, NUTSSampler

sampler = NUTSSampler(nuts_sampler="pymc")
gaussian = VAR(lags=4).fit(data, sampler=sampler)
student = VAR(lags=4, error_dist="student_t").fit(data, sampler=sampler)

az.compare({"gaussian": gaussian.idata, "student_t": student.idata})
```

:::{admonition} Sampler backend
:class: note
`nuts_sampler="pymc"` is required here. The nutpie backend ignores the
`idata_kwargs` that request the `log_likelihood` group, so a nutpie-sampled
`InferenceData` has nothing for `az.compare` to read. This is a pre-existing
quirk of the nutpie integration, not specific to Student-t errors.
:::

(limitations)=
## Limitations

Two combinations are refused outright rather than approximated.

**Stochastic volatility.** `VAR(volatility="sv", error_dist="student_t")` raises
at construction. The degrees of freedom and the log-volatility innovation
variance both absorb outliers, so the two are weakly identified jointly and
NUTS mixes poorly. Pick the mechanism that matches the problem: drifting
volatility level → stochastic volatility with Gaussian errors; isolated extreme
observations → constant volatility with Student-t errors.

**Conditional forecasts and structural scenarios.** Both raise
`NotImplementedError` under heavy tails. The Waggoner–Zha constrained draw and
the ADPRR three-way partition are Gaussian conditional-law results, and the
plausibility statistic's `χ²` reference assumes Gaussian shocks. Under a t the
conditional law has updated degrees of freedom and a Mahalanobis-inflated scale,
and the plausibility reference becomes an F/Hotelling statistic. Returning a
half-valid scenario would be worse than the error, because nothing in the output
would look wrong. Unconditional density forecasts via `forecast()` and in-sample
`counterfactual()` are both fully valid under the t.

See ADR-0007 for the full derivation of which quantities change and which are
exactly invariant.
