# What Is a Bayesian VAR?

A **Vector Autoregression (VAR)** models multiple time series as a system of equations where each variable depends on its own lags and the lags of all other variables in the system.

A **Bayesian VAR** adds prior distributions over the model parameters. This serves two purposes:

1. **Regularization** — VARs have many parameters (grows as $n^2 \times p$ where $n$ is the number of variables and $p$ is the lag order). Priors shrink estimates toward sensible values, reducing overfitting.
2. **Uncertainty quantification** — instead of point estimates, you get a full posterior distribution over coefficients, forecasts, and structural quantities.

## When to use a Bayesian VAR

- You have a moderate number of macroeconomic or financial time series (2--20 variables)
- You want probabilistic forecasts with credible intervals
- You want to study how shocks propagate through a system (impulse responses)
- You want to decompose forecast error variance or historical variation by shock source

## The Impulso pipeline

Impulso models this as a sequence of immutable types:

```
VARData -> VAR -> FittedVAR -> IdentifiedVAR
```

Each step adds information. You cannot skip steps or go backward.

## Heavy-tailed observation errors

The standard VAR assumes Gaussian observation errors, $y_t \sim N(\mu_t, \Omega)$. Macroeconomic samples routinely violate that assumption in one specific way: a handful of quarters — a financial collapse, a pandemic, a devaluation — sit far outside the bulk of the distribution. Under a Gaussian likelihood those observations dominate estimation, because the Gaussian score $\Omega^{-1}(y_t - \mu_t)$ grows without bound as the observation moves into the tail.

Replacing the likelihood with a multivariate Student-t, $y_t \sim \mathrm{MvT}_\nu(\mu_t, \Omega)$, bounds that influence. The t score is

$$
\frac{\partial \log p}{\partial \mu_t} = \frac{\nu + n}{\nu + q_t} \, \Omega^{-1}(y_t - \mu_t),
\qquad q_t = (y_t - \mu_t)^\top \Omega^{-1} (y_t - \mu_t),
$$ (t-score)

so the weight $(\nu + n)/(\nu + q_t)$ falls as the squared Mahalanobis distance $q_t$ rises. The score is bounded and eventually *redescending*: a sufficiently extreme observation pulls the estimate less, not more. This is automatic outlier downweighting, and it removes the need to choose dummy dates by hand.

The t is a scale mixture of normals,

$$
y_t = \mu_t + L \xi_t, \qquad \xi_t = z_t / \sqrt{g_t / \nu},
\qquad z_t \sim N(0, I_n), \quad g_t \sim \chi^2_\nu,
$$ (t-mixture)

with one mixing variable $g_t$ shared across the whole observation vector. A draw with small $g_t$ inflates every component at once — which is what makes the joint law a multivariate t rather than a product of independent t marginals.

One consequence deserves care: $\Omega = LL^\top$ is the **scale** matrix, not the covariance. The covariance is $\nu/(\nu-2)\,\Omega$, finite only for $\nu > 2$. Impulso keeps `sigma()` returning $\Omega$ — identification factorises the scale matrix — and exposes the covariance separately as `innovation_covariance()`. Variance decompositions and historical decompositions are exactly invariant to the distinction; impulse responses are in scale units. See ADR-0007 and the [heavy-tailed errors how-to](../how-to/heavy-tailed-errors.md).
