# Conditioning on Probabilities

Some scenarios are not paths. "Assume a 40% chance of recession next year", "assume the market's mean rate path", "what shocks would push inflation below target?" — none of these pin a variable to a number on every draw. They are statements about the forecast *distribution*, and Impulso imposes them by **entropic tilting** {cite:p}`robertsonTallmanWhiteman2005,kruegerClarkRavazzolo2017`: reweight the draws you already have so the statement holds, using the weights closest (in relative entropy) to leaving them alone.

## State a probability target

Tilt a density forecast onto the probability you want:

```python
from impulso.scenario import ProbabilityTarget

forecast = fitted.forecast(steps=8, seed=0)

tilted = forecast.tilt([
    ProbabilityTarget(variable="gdp", horizon=4, threshold=0.0, probability=0.4)
])

tilted.median()  # weighted median
tilted.hdi()  # weighted HDI
tilted.plot()  # tilted fan against the untilted median
```

`horizon` is 1-based, the same convention as `VariablePath`. The event is `gdp < 0.0` at step 4; pass `direction="above"` for the other side. Use `MomentTarget(variable, horizon, mean)` to fix a mean instead of a probability.

## Read the diagnostics before the forecast

Tilting cannot create draws. If you ask for a probability the sample barely supports, the weights pile onto a handful of draws and every summary silently becomes a summary of those few draws. `summary()` reports what happened:

```python
tilted.summary()
# {'ess': 612.4, 'ess_fraction': 0.153, 'kl_divergence': 0.44, 'n_draws': 4000,
#  'targets': [{'target': 'P(gdp[h=4] < 0)', 'requested': 0.4,
#               'achieved': 0.4, 'draws_in_event': 380}]}
```

- **`ess`** is the Kish effective sample size, `1 / Σ wᵢ²` — the number of draws the tilt is effectively using. `ess_fraction` expresses it as a share of `n_draws`, and a tilt below 10% of the sample warns.
- **`kl_divergence`** is how far the tilt moved the forecast. Zero means the target was already true; large values mean the model disagrees with you.
- **`achieved`** against **`requested`** confirms the solver hit the target. `draws_in_event` is how many of the original draws satisfied it before reweighting.

A target no draw satisfies is refused outright, naming the counts:

```python
forecast.tilt([ProbabilityTarget(variable="gdp", horizon=4, threshold=-50.0, probability=0.4)])
# ValueError: 0 of 4000 draws satisfy P(gdp[h=4] < -50); the target is
# unachievable by reweighting — widen the threshold or increase the number of draws.
```

## Chain hard pins with soft targets

Hard conditioning and probability targets combine by chaining, not by a mixed call:

```python
from impulso.scenario import VariablePath

conditional = fitted.conditional_forecast(
    steps=8,
    conditions=[VariablePath(variable="rate", values=rate_path)],
    seed=0,
)
both = conditional.tilt([
    ProbabilityTarget(variable="gdp", horizon=4, threshold=0.0, probability=0.4)
])
```

The order does not matter to the result and the pins are safe: they hold pathwise on *every* draw, and reweighting never moves a draw, so no weighting can break them. `ScenarioResult` inherits `tilt()` too, so a structural scenario can carry a probability target the same way.

:::{admonition} Tilting needs a density forecast
:class: warning
`tilt()` refuses a mean forecast (`include_shock_uncertainty=False`). Mean-mode draws carry parameter uncertainty only, so probabilities read off them are not predictive probabilities.
:::

## Run the scenario backwards

Reverse stress testing asks the inverse question: not "what happens if this shock hits?" but "what shocks would deliver this outcome?"

```python
result = identified.reverse_stress(
    variable="inflation",
    threshold=1.0,
    steps=12,
    horizon=8,
    direction="below",
    seed=0,
)

result.shock_cocktail()  # step x shock, in one-standard-deviation units
result.summary()
result.plot()
```

Impulso draws an unconditional forecast together with the structural shocks behind it, conditions on the event (`probability=1.0` by default — exact conditioning), and averages the shocks of the draws that produced the outcome. That average is the **shock cocktail**: the structural configuration most associated with the stress event, in one-standard-deviation units.

Reading the output:

- `baseline_probability` is how likely the event was before conditioning. If it is already 0.4, the "stress" is not stressful.
- The cocktail's entries are shock sizes. A `-2.1` on a demand shock at step 3 means the outcome is associated with a 2.1-standard-deviation negative demand shock there.
- `q = ‖cocktail‖²` is the cocktail's total magnitude in the same units as the scenario plausibility statistic, so `q = 9` reads as "a 3-standard-deviation configuration".
- `q_cal` is a separate reading: it applies the ADPRR binomial calibration to the *tilt's relative entropy* — how far conditioning on the event moved the forecast — mapping it onto `[0.5, 1]`, with 0.5 meaning the event cost nothing to impose.

Softening the conditioning keeps more of the sample:

```python
result = identified.reverse_stress(
    variable="inflation", threshold=1.0, steps=12, probability=0.6, seed=0
)
```

At `probability=0.6` the event carries 60% of the weight instead of all of it, the effective sample size rises, and the cocktail shrinks toward zero — the outcome is being made likely rather than certain.

:::{admonition} Cocktails are associations, not causes
:class: note
The cocktail is a conditional mean over draws, so it reports which shock configurations *accompany* the outcome under the estimated model. It is not a unique cause: other configurations in the retained set may look nothing like the average, and a wide `hdi()` around the conditioned path is the signal to look at them.
:::
