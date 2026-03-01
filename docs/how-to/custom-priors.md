# Writing a Custom Prior

Impulso uses `typing.Protocol` for extensibility. You can write your own prior by implementing the `Prior` protocol.

## The Prior protocol

```python
from impulso.protocols import Prior

class MyPrior:
    def build_priors(self, n_vars: int, n_lags: int) -> dict:
        ...
```

Your `build_priors` method must return a dictionary with keys `"B_mu"` and `"B_sigma"`, both NumPy arrays of shape `(n_vars, n_vars * n_lags)`.

- `B_mu`: Prior mean for VAR coefficient matrix
- `B_sigma`: Prior standard deviation for VAR coefficient matrix

## Example: Flat prior

```python
import numpy as np

class FlatPrior:
    def build_priors(self, n_vars: int, n_lags: int) -> dict:
        n_coeffs = n_vars * n_lags
        return {
            "B_mu": np.zeros((n_vars, n_coeffs)),
            "B_sigma": np.ones((n_vars, n_coeffs)) * 10.0,
        }
```

## Using your custom prior

```python
from impulso import VAR

spec = VAR(lags=2, prior=FlatPrior())
fitted = spec.fit(data)
```
