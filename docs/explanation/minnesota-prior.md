# The Minnesota Prior

The **Minnesota prior** (Impulso, 1986) is the most widely used prior for Bayesian VARs. It encodes the belief that each variable follows a random walk, with coefficients on other variables' lags shrunk toward zero.

## Key hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tightness` | 0.1 | Overall shrinkage. Smaller = more shrinkage toward prior. |
| `decay` | `"harmonic"` | How fast coefficients shrink on longer lags. `"harmonic"`: $1/l$. `"geometric"`: $1/l^2$. |
| `cross_shrinkage` | 0.5 | Relative shrinkage on other variables' lags vs own lags. 0 = only own lags matter, 1 = equal treatment. |

## Intuition

The prior mean for the coefficient on a variable's own first lag is 1.0 (random walk). All other coefficients have prior mean 0.0. The prior standard deviation controls how far the posterior can move from these defaults.

## Usage in Impulso

```python
from impulso import VAR
from impulso.priors import MinnesotaPrior

# Use defaults
spec = VAR(lags=4, prior="minnesota")

# Customize hyperparameters
prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.3)
spec = VAR(lags=4, prior=prior)
```
