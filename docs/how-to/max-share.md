# Using Max-Share Identification

Max-share identification picks out the single structural shock that explains the largest possible share of one variable's variance over a stated band of frequencies. Instead of asserting a sign or an ordering, you nominate a target variable and a band, and the shock is whatever maximises that share.

## Define the target and the band

```python
from impulso.identification import MaxShare

scheme = MaxShare(target="unemployment", band=(6, 32))
```

`band` is `(low_period, high_period)` in **periods of the sampling interval**, shortest first — not radians. With quarterly data, `(6, 32)` is the conventional business-cycle window of six to thirty-two quarters. The monthly analogue is `(18, 96)`. The lower bound may not go below 2, which is the Nyquist frequency; the upper bound may be `inf`.

## Apply it

```python
identified = fitted.set_identification_strategy(scheme)
irf = identified.impulse_response(horizon=40)
```

The scheme needs the posterior lag coefficients to build the transfer function; `set_identification_strategy` passes them through automatically.

Only one shock is identified. The other columns are an orthogonal completion, labelled `unidentified_1`, `unidentified_2`, and so on. Impulse responses report them, since they are a valid decomposition of the residual covariance, but forecast error variance decompositions (FEVDs) mask their shares to `NaN` and the historical decomposition collapses them into a single `unidentified_remainder` column. The identified column is signed so the shock raises the target variable on impact.

Rows of the shock matrix stay in the data's own variable order. There is no ordering parameter, so nothing can be silently permuted.

## Read the diagnostics

Identification succeeds or fails per posterior draw, so the diagnostics are posterior quantities. Summary statistics ride along on the shock matrix:

```python
P = identified.shock_matrix()
P.attrs["max_share_share_median"]        # band variance share achieved
P.attrs["max_share_share_q05"]
P.attrs["max_share_eigen_ratio_median"]  # lambda_2 / lambda_1
P.attrs["max_share_explosive_draws"]
P.attrs["max_share_singular_draws"]
```

For the full per-draw picture:

```python
posterior = fitted.idata.posterior
diagnostics = scheme.max_share_diagnostics(
    identified.volatility.cholesky_at(posterior, t=None),
    fitted.var_names,
    posterior,
)
diagnostics["share"]            # shape (chains, draws)
diagnostics["eigen_ratio"]
diagnostics["spectral_radius"]
diagnostics["condition_max"]
```

The eigenvalue ratio is the one to watch. It compares the second-largest eigenvalue of the band variance form to the largest. Near zero, the maximiser is a well-separated direction. Near one, two directions explain almost the same share, and which of them is returned is down to the eigensolver rather than to the data — Impulso warns when the posterior median exceeds 0.9, and the honest response is to widen the band, change the target, or stop treating the answer as a point estimate.

A spectral radius above one means that draw is explosive, so its spectral density is not that of a stationary process and the share has no interpretation. Those draws are warned about but kept: posteriors on persistent data routinely put mass above one, and dropping them would quietly condition the posterior on stability.

## When draws are undefined

The transfer function is `C(w) = (I - sum_j A_j e^{-i w j})^-1`. If a draw has a root essentially on the unit circle *at a frequency inside your band*, that inverse is numerically meaningless there and the draw is blanked:

```python
scheme = MaxShare(
    target="unemployment",
    band=(6, 32),
    on_undefined="nan",     # default; "raise" errors instead
    max_condition=1e8,      # condition number above which C(w) is refused
)
```

`NaN` draws propagate: impulse responses and FEVDs report `NaN` for them rather than a misleading zero, and the scenario methods reject them outright. If any of that is in your plans, run with `on_undefined="raise"` so the problem surfaces at identification time instead of three steps later.

## Tuning the quadrature

The band integral is a uniform midpoint rule with `n_frequencies` nodes, 192 by default, which is comfortably converged for typical macro VARs. Raise it for very narrow bands, or when draws sit close to a unit root and the spectral density has a sharp peak inside the band:

```python
scheme = MaxShare(target="unemployment", band=(28, 32), n_frequencies=512)
```

The cost is linear in `n_frequencies`, and the sweep is computed once per posterior and reused, so raising it is cheap relative to sampling.

## A climate example

Nothing about the scheme is macroeconomic. With annual data on global-mean surface temperature, ocean heat content and an El Niño-Southern Oscillation (ENSO) index, the same machinery separates variance by timescale:

```python
forced = MaxShare(target="temperature", band=(10, float("inf")))
internal = MaxShare(target="temperature", band=(2, 8))
```

The first asks which single shock explains most of temperature's decadal-and-longer variance; the second asks the same question of its interannual variance. The two bands are different questions about the same posterior, and each returns its own structural column.

:::{admonition} A band is not a mechanism
:class: warning
The low-frequency shock is not "the forced response" and the
interannual one is not "ENSO" — those are labels you supply, and the
maximisation cannot check them. All the scheme establishes is that a
particular direction in shock space accounts for most of the variance in
a particular frequency range. Treat the band as a way of *organising* the
variance, then argue separately, from physics or from an independent
series, that the direction it selects is the thing you named.
:::
