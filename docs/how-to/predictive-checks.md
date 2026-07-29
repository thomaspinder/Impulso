# Prior and Posterior Predictive Checks

Predictive checks ask whether the model can generate data that looks like the data you have. Impulso exposes them at the two natural points in the pipeline: `VAR.prior_predictive` before fitting, and `FittedVAR.posterior_predictive` after.

Both return a plain ArviZ `InferenceData`, so `az.plot_ppc` and friends work directly.

## Checking the prior before you fit

`VAR.prior_predictive` builds the same PyMC graph `fit` builds and draws from it with `pymc.sample_prior_predictive`. The prior you check is by construction the prior you sample:

```python
import arviz as az
from impulso import VAR

spec = VAR(lags=4, prior="minnesota")
prior = spec.prior_predictive(data, draws=500, random_seed=0)

az.plot_ppc(prior, group="prior", num_pp_samples=50)
```

The returned object has `prior` (every latent — inspect `prior["B"]` to see what the Minnesota prior actually implies for the coefficients), `prior_predictive` (the simulated `obs`), and `observed_data`.

If the prior band does not contain the data, the prior is fighting the likelihood. If it is orders of magnitude wider, the prior is uninformative and you are paying for it in sampling efficiency.

Compare with quantiles, not with mean ± k·sd: the default scale prior is HalfCauchy, which has no finite moments, so a prior predictive mean is meaningless.

```python
import numpy as np

draws = prior.prior_predictive["obs"].values[0]   # (draws, time, var)
lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)

observed = prior.observed_data["obs"].values      # already lag-trimmed
covered = ((observed >= lower) & (observed <= upper)).mean()
```

:::{note}
`pymc.sample_prior_predictive` returns a single chain, so `obs` has shape `(1, draws, T - lags, n_vars)`.
:::

:::{note}
With `volatility="sv"`, the per-variable log-volatility priors are seeded from the OLS residuals of `data`. That "prior" is therefore mildly data-informed in its scale. The constant-volatility default is not.
:::

## Checking the fit afterwards

`FittedVAR.posterior_predictive` replicates the estimation sample from the posterior:

```python
fitted = spec.fit(data)
ppc = fitted.posterior_predictive(seed=0)

az.plot_ppc(ppc, num_pp_samples=100)
```

Each replicate is **one-step-ahead conditioned on the observed lags**:

$$
y^{rep}_t = c + B x^{obs}_t + B_{exog} z_t + L_t \varepsilon_t, \qquad \varepsilon_t \sim N(0, I)
$$

That is the standard posterior-predictive object for a conditional model, and the one `az.plot_ppc` expects. It is *not* a path simulated forward from initial conditions — for that, use `fitted.forecast(steps=...)`, which iterates its own predictions.

`L_t` comes from the volatility process, so under `volatility="sv"` the replicate spread genuinely varies with `t`.

### Coverage

A quick calibration check — what fraction of the observed data lands inside the 95% predictive band:

```python
import numpy as np

draws = ppc.posterior_predictive["obs"].values          # (chain, draw, time, var)
flat = draws.reshape(-1, *draws.shape[2:])              # (chain*draw, time, var)
lower, upper = np.quantile(flat, [0.025, 0.975], axis=0)

observed = ppc.observed_data["obs"].values
coverage = ((observed >= lower) & (observed <= upper)).mean()
print(f"95% band covers {coverage:.1%} of observations")
```

Substantially below 95% means the model is over-confident; substantially above means it is over-dispersed.

### Mean mode for residual diagnostics

`simulate_innovations=False` drops the innovation term and returns the conditional mean under parameter uncertainty. Subtracting it from the observed data gives the reduced-form residuals per draw:

```python
fit_only = fitted.posterior_predictive(simulate_innovations=False)

residuals = fit_only.observed_data["obs"].values - fit_only.posterior_predictive["obs"].values
```

Mean mode consumes no randomness at all, so `seed` is irrelevant there and repeated calls agree exactly.

## Attaching the result to the fit

`posterior_predictive` never mutates `fitted.idata`; it returns a fresh object. Attach it yourself when you want a single `InferenceData` for downstream ArviZ work:

```python
fitted.idata.extend(ppc)
az.plot_ppc(fitted.idata)
```

:::{admonition} Memory
:class: warning
The replicate array is dense: `chains × draws × T × n_vars` float64 values — roughly 19 MB at 4 chains, 1000 draws, 200 dates and 3 variables, and it grows linearly in all four. Thin the posterior before calling if that is too large.
:::

## Which method for which question

| Question | Method |
| --- | --- |
| Is my prior plausible before I see the data? | `VAR.prior_predictive(data)` |
| Can the fitted model reproduce the sample it was fitted to? | `FittedVAR.posterior_predictive()` |
| What happens after the sample ends? | `FittedVAR.forecast(steps=...)` |
| What happens if a variable follows a pinned path? | `FittedVAR.conditional_forecast(...)` |
