# Long-Run Restrictions

Cholesky identification assumes a **contemporaneous** causal ordering: the first variable isn't affected by any other variable within the same period, the second is affected only by the first, and so on. This is a strong assumption that may not match your economic theory.

Sometimes theory says nothing about contemporaneous effects but makes clear predictions about **long-run** effects. For example, in the Blanchard-Quah (1989) framework:

- **Supply shocks** can have permanent effects on both output and prices
- **Demand shocks** have no permanent effect on output (only transitory)

Long-run restrictions implement this idea. Instead of applying Cholesky to the contemporaneous impact matrix, we apply it to the **long-run cumulative impact matrix** — forcing some shocks to have zero permanent effect on certain variables.

## Defining the scheme

The ordering determines which shocks can have permanent effects:

```python
from impulso.identification import LongRunRestriction

scheme = LongRunRestriction(ordering=["output", "prices"])
```

This means:
- The **first shock** (associated with "output") can have permanent effects on both output and prices
- The **second shock** (associated with "prices") has **no permanent effect on output** — only on prices

The ordering encodes your identifying assumption about long-run neutrality.

!!! tip "Reading the ordering"
    Think of it as: "shocks later in the ordering cannot permanently affect variables earlier in the ordering." The last shock has the most restrictions; the first shock is unrestricted.

## Applying to a fitted model

The workflow is identical to Cholesky — only the economics differ:

```python
from impulso import VAR, VARData
from impulso.identification import LongRunRestriction

# Fit reduced-form VAR
data = VARData.from_df(df, endog=["output", "prices"])
fitted = VAR(lags=4, prior="minnesota").fit(data)

# Apply long-run identification
scheme = LongRunRestriction(ordering=["output", "prices"])
identified = fitted.set_identification_strategy(scheme)
```

The resulting `IdentifiedVAR` supports all the same analysis methods — impulse responses, variance decomposition, and historical decomposition.

## Impulse response analysis

```python
irf = identified.impulse_response(horizon=40)
irf.plot()
```

When you examine the impulse responses, you should see the long-run restriction at work: the cumulative response of "output" to the second shock (the "prices" shock) converges to zero as the horizon increases. The first shock (the "output" shock) can have a permanent effect on both variables.

!!! note "Convergence to zero"
    The restriction forces the **cumulative** long-run effect to be zero, not the response at each individual horizon. The impulse response at horizon $h$ may be non-zero — it's only the sum across all horizons that vanishes.

## Variance decomposition and historical decomposition

These work identically to other identification schemes:

```python
# What fraction of forecast error variance is due to each shock?
fevd = identified.fevd(horizon=40)
fevd.plot()

# What drove the historical movements in each variable?
hd = identified.historical_decomposition()
hd.plot()
```

## When to use long-run restrictions

| Situation | Recommended scheme |
|-----------|-------------------|
| Theory specifies contemporaneous ordering | `Cholesky` |
| Theory specifies long-run neutrality | `LongRunRestriction` |
| Theory specifies signs but not ordering | `SignRestriction` |
| Multiple schemes plausible | Try several and compare IRFs |

!!! warning "Stationarity required"
    Long-run restrictions require the VAR to be stationary (all eigenvalues of the companion matrix inside the unit circle). If the model has a unit root, the long-run multiplier matrix is undefined. If some posterior draws are near-non-stationary, results may be numerically unstable.
