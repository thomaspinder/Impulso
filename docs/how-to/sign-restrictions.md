# Using Sign Restrictions

Sign restrictions identify structural shocks by imposing qualitative constraints on the impact matrix, rather than a recursive ordering.

## Define restrictions

Specify which direction each variable should respond to each named shock:

```python
from impulso.identification import SignRestriction

scheme = SignRestriction(
    restrictions={
        "gdp":       {"supply": "+", "demand": "+"},
        "inflation": {"supply": "-", "demand": "+"},
    },
    n_rotations=1000,
    random_seed=42,
)
```

- `"+"` means the variable must increase on impact
- `"-"` means the variable must decrease on impact
- Omitted entries are unrestricted

## Apply to a fitted model

```python
identified = fitted.set_identification_strategy(scheme)
irf = identified.impulse_response(horizon=20)
```

## Tips

- More restrictions = fewer valid rotations found per draw. If identification is too tight, consider relaxing some constraints.
- Increase `n_rotations` if many draws fail to find a valid rotation (the default fallback is plain Cholesky).
- Set `random_seed` for reproducibility.
