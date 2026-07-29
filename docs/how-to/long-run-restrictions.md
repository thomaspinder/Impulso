# Using Long-Run Restrictions

Long-run restrictions identify structural shocks by what they do *eventually*, not on impact: a shock is defined by having no permanent effect on the level of some variable. This is the Blanchard-Quah scheme.

## Define the restriction

Give the variables in order, most restricted first, and name the shocks:

```python
from impulso.identification import LongRunRestriction

scheme = LongRunRestriction(
    ordering=["output_growth", "unemployment"],
    shock_names=["supply", "demand"],
)
```

That single zero says the demand shock has no permanent effect on the level of output. Shock `j` has no long-run effect on any variable ordered before it, so the ordering is the restriction — with two variables there is exactly one.

If naming the zeros directly is clearer than reasoning about an ordering, use the alternative constructor. It recovers the ordering from the pattern, and refuses patterns that no ordering can produce:

```python
scheme = LongRunRestriction.from_zero_restrictions(
    restrictions={"output_growth": ["demand"]},
    var_names=["output_growth", "unemployment"],
    shock_names=["supply", "demand"],
)
```

:::{admonition} Your variables must be differenced
:class: warning
The restriction is on the long-run level of the variables *as they enter
the model*. "Demand has no permanent effect on output" therefore requires
output to enter as a growth rate — the level then accumulates, and a
zero cumulative effect on the growth rate is a zero permanent effect on
the level. If you pass output in levels, the restriction says the demand
shock has no permanent effect on the *growth rate*, which is a much
weaker and rarely intended claim. Impulso cannot detect the difference.
:::

## Apply it

```python
identified = fitted.set_identification_strategy(scheme)
irf = identified.impulse_response(horizon=40)
```

The scheme needs the posterior lag coefficients to build the long-run multiplier; `set_identification_strategy` passes them through automatically.

Rows of the structural shock matrix stay in the data's variable order, whatever `ordering` says. Only the shock columns follow `shock_names`.

## Read the diagnostics

Identification succeeds or fails per posterior draw, so the diagnostics are posterior quantities. Summary statistics ride along on the shock matrix:

```python
P = identified.shock_matrix()
P.attrs["long_run_singular_draws"]      # draws where C(1) is undefined
P.attrs["long_run_explosive_draws"]     # draws with spectral radius > 1
P.attrs["long_run_condition_q95"]       # conditioning of I - sum_j A_j
P.attrs["long_run_spectral_radius_max"]
```

For the full per-draw picture:

```python
diagnostics = scheme.long_run_diagnostics(fitted.idata.posterior)
diagnostics["condition"]         # shape (chains, draws)
diagnostics["spectral_radius"]   # shape (chains, draws)
```

A spectral radius above one means that draw's moving-average sum diverges, so the long run does not exist for it. Impulso warns and reports those draws but keeps them: posteriors on persistent data routinely put mass above one, and dropping them would quietly condition the posterior on stability.

## When draws are undefined

The long-run multiplier is `C(1) = (I - sum_j A_j)^-1`. When that matrix is close to singular, the inverse is numerically meaningless. Those draws are blanked:

```python
scheme = LongRunRestriction(
    ordering=["output_growth", "unemployment"],
    shock_names=["supply", "demand"],
    on_undefined="nan",     # default; "raise" errors instead
    max_condition=1e8,      # condition number above which C(1) is refused
)
```

`NaN` draws propagate: impulse responses and forecast error variance decompositions (FEVD) report `NaN` for them rather than a misleading zero, and highest-density intervals over a mixed set of draws may come back `NaN` too. The scenario methods (`counterfactual`, `structural_scenario`) reject them outright. If any of that is in your plans, run with `on_undefined="raise"` so the problem surfaces at identification time instead of three steps later.

## A climate example

The scheme is not specific to macroeconomics. Take global-mean surface temperature and an El Niño-Southern Oscillation (ENSO) index, with temperature differenced:

```python
scheme = LongRunRestriction(
    ordering=["temperature_change", "enso_index"],
    shock_names=["forced", "internal"],
)
identified = fitted.set_identification_strategy(scheme)
```

The restriction: internal variability has no permanent effect on the level of global-mean temperature, while forced variability may. That is one assumption, stated plainly, and it is doing all the identifying work — so be clear about what it commits you to:

- Temperature must enter as a change, not a level, or the restriction means something else.
- With two variables this is exactly one restriction. Adding a third variable would assert three zeros at once, which is a far stronger joint claim.
- The zero is exact and permanent. If internal variability has a small but genuinely permanent effect, the scheme will attribute it to the forced shock.
