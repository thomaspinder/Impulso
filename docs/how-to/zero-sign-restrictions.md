# Combining Zero and Sign Restrictions

`ZeroSignRestriction` imposes exact zeros on the impact matrix alongside sign restrictions on impulse responses. Use it when part of your identification rests on a timing or exclusion argument you can defend outright, and the rest on the direction a response should take.

## A worked example

Take a three-variable system: a temperature anomaly, a measure of economic activity, and emissions. Economic activity moves emissions immediately, and emissions eventually move temperature, but the thermal inertia of the ocean means an activity shock cannot register in this year's temperature anomaly. That is a zero, not a sign.

```python
from impulso.identification import ZeroSignRestriction

scheme = ZeroSignRestriction(
    shock_names=["climate", "activity", "emissions"],
    zero_restrictions={
        # An activity shock has no contemporaneous effect on temperature:
        # the physical lag between emissions and warming is far longer
        # than the sampling frequency.
        "temperature": ["activity"],
    },
    sign_restrictions={
        "temperature": {"climate": "+"},
        "activity":    {"climate": "-", "activity": "+"},
        "emissions":   {"activity": "+", "emissions": "+"},
    },
    random_seed=42,
)

identified = fitted.set_identification_strategy(scheme)
irf = identified.impulse_response(horizon=20)
```

## Specifying restrictions

- `shock_names` fixes the column order of the returned structural matrix. Name fewer shocks than you have variables and the rest are labelled `unidentified_1`, `unidentified_2`, ... — those columns carry no restrictions and are rotation-arbitrary, so `fevd()` masks their shares.
- `zero_restrictions` maps a variable to the shocks that do not move it on impact. Zeros bind at horizon 0 only; long-run zeros are not supported.
- `sign_restrictions` uses the same format as `SignRestriction`: variable → shock → `"+"` or `"-"`.
- `restriction_horizon=H` imposes the *signs* at horizons `0..H`. The zeros stay at impact.

A cell cannot be restricted to zero and to a sign at once — that is a contradiction at horizon 0, and construction fails with a `ValueError`.

## How many zeros are admissible

Sort the shocks by how many zeros they carry, most first. The shock in position `j` may carry at most `n - j` zeros. Break that and identification is impossible for *any* orthogonal matrix, so `identify()` raises before sampling starts rather than burning through rotations. At the limit — `n - 1`, `n - 2`, ..., `0` — the zeros exactly identify the system and reproduce the Cholesky factor.

## When draws fail

A draw fails when no candidate satisfies the sign restrictions within `n_rotations` attempts.

:::{admonition} Failed draws become NaN, not Cholesky
:class: warning
`ZeroSignRestriction` fills failed draws with `NaN` and warns once with the count and fraction. It deliberately does **not** fall back to the unrotated factor the way `SignRestriction` does: that fallback would silently break the zero restrictions, which are the whole point of the scheme.

`NaN` draws propagate into impulse-response and variance-decomposition summaries and are rejected by the scenario methods, so treat a non-trivial failure fraction as a result to act on, not a warning to suppress. Raise `n_rotations`, relax the signs, or set `on_failure="raise"` to stop at the first failure.
:::

## Tuning

- Each unpinned column sign is effectively a coin flip, so a scheme with `k` sign-restricted shocks accepts roughly one candidate in `2**k` even when the restrictions are otherwise easy. Budget `n_rotations` accordingly.
- Read the acceptance rate off the shock matrix:

  ```python
  attrs = identified.shock_matrix().attrs
  attrs["zero_sign_acceptance_rate"]     # fraction of draws identified
  attrs["zero_sign_mean_attempts"]       # candidates drawn per draw
  attrs["zero_sign_max_zero_violation"]  # largest |zero cell|, expect ~1e-14
  ```

- Set `random_seed` for reproducibility.
- Naming a shock without giving it any sign restriction leaves its column sign unidentified, so posterior summaries average over both directions. Impulso warns when this happens.

:::{admonition} Draws are unweighted
:class: note
No importance weight corrects for the volume element of the zero-restricted manifold, so set-identified results are not the uniform-conditional prior of Arias, Rubio-Ramírez and Waggoner (2018). See [the identification explanation](../explanation/identification.md) for what this does and does not affect.
:::
