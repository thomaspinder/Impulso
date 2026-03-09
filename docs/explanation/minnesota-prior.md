# The Minnesota Prior

The **Minnesota prior** (Litterman, 1986) is the most widely used prior for Bayesian VARs. It encodes the belief that each variable follows a random walk, with coefficients on other variables' lags shrunk toward zero.

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

## Conjugate estimation

When the Minnesota prior is paired with a Normal-Inverse-Wishart (NIW) likelihood, the posterior belongs to the same family — this is called **conjugacy**. The practical consequence is dramatic: instead of running an iterative MCMC sampler, we can compute the posterior parameters in closed form and draw from it directly.

The prior specifies:

$$B \mid \Sigma \sim \mathcal{MN}(B_0, \Sigma, V_0), \qquad \Sigma \sim \mathcal{IW}(S_0, \nu_0)$$

where $\mathcal{MN}$ is the matrix normal distribution and $\mathcal{IW}$ is the inverse Wishart. Given data matrices $Y$ (observations) and $X$ (lagged regressors), the posterior parameters are:

$$V_{\text{post}} = (V_0^{-1} + X^\top X)^{-1}$$

$$B_{\text{post}} = V_{\text{post}}(V_0^{-1} B_0 + X^\top Y)$$

$$\nu_{\text{post}} = \nu_0 + T$$

$$S_{\text{post}} = S_0 + Y^\top Y + B_0^\top V_0^{-1} B_0 - B_{\text{post}}^\top V_{\text{post}}^{-1} B_{\text{post}}$$

The posterior mean for $B$ is a precision-weighted average of the prior mean $B_0$ and the OLS estimate $(X^\top X)^{-1} X^\top Y$. When the prior is tight (small $V_0$), the posterior stays close to the random walk prior. When data is abundant (large $X^\top X$), the posterior approaches OLS. The posterior for $\Sigma$ is an inverse Wishart with updated scale and degrees of freedom.

Sampling is straightforward: draw $\Sigma \sim \mathcal{IW}(S_{\text{post}}, \nu_{\text{post}})$, then $B \mid \Sigma \sim \mathcal{MN}(B_{\text{post}}, \Sigma, V_{\text{post}})$. Each draw is independent — no burn-in, no autocorrelation, no convergence diagnostics. `ConjugateVAR` implements this approach.

## Data-driven prior selection

The Minnesota prior's hyperparameters — `tightness`, `cross_shrinkage`, and `decay` — are often chosen by convention or trial-and-error. Giannone, Lenza & Primiceri (2015) proposed an empirical Bayes approach: choose hyperparameters by maximising the **marginal likelihood**.

The marginal likelihood is the probability of the observed data $Y$ given hyperparameters $\lambda$, after integrating out all model parameters:

$$p(Y \mid \lambda) = \int p(Y \mid B, \Sigma) \, p(B, \Sigma \mid \lambda) \, dB \, d\Sigma$$

Thanks to conjugacy, this integral has a closed-form solution involving determinants and multivariate gamma functions. Optimising $\log p(Y \mid \lambda)$ over $\lambda = (\text{tightness}, \text{cross\_shrinkage})$ using standard numerical optimisation (L-BFGS-B) is fast and reliable.

The marginal likelihood naturally balances two forces:

- **Fit**: a loose prior (high tightness) lets the model fit the data closely
- **Parsimony**: a loose prior also spreads probability mass over implausible parameter values, reducing the marginal likelihood

The optimal hyperparameters sit at the sweet spot where the prior is just flexible enough to capture the data's patterns without wasting probability on noise.

This approach is sometimes called **GLP** after the authors' initials. In Impulso, use `ConjugateVAR.optimize_prior()` or the shorthand `prior="minnesota_optimized"`.

## Dummy observation priors

An elegant trick from the classical VAR literature encodes prior beliefs not by modifying the prior distribution directly, but by **appending synthetic observations** to the dataset. These "dummy observations" push the posterior in the desired direction while preserving conjugacy.

### Sum-of-coefficients prior

The sum-of-coefficients prior (Doan, Litterman & Sims, 1984) encodes the belief that if all variables have been at their initial sample values forever, they should persist at those values. Formally, it adds observations implying:

$$\sum_{j=1}^{p} A_j \approx I$$

where $A_j$ are the lag coefficient matrices. This discourages the model from predicting rapid mean-reversion when it isn't supported by the data. The hyperparameter $\mu$ controls the prior's strength: smaller $\mu$ imposes the belief more tightly.

### Single-unit-root prior

The single-unit-root prior (Sims, 1993) encodes the belief that each variable follows an independent random walk. It adds a single observation per variable that makes the model reluctant to introduce cointegrating relationships not strongly supported by the data. The hyperparameter $\delta$ controls its strength.

### Practical usage

In Impulso, dummy observations are added via `VARData.with_dummy_observations()`:

```python
data_augmented = data.with_dummy_observations(n_lags=4, mu=1.0, delta=1.0)
```

The augmented data can then be passed to any estimation method — `ConjugateVAR`, `VAR`, or used with `optimize_prior()`. Because the dummy observations are simply extra rows in the data matrices, they preserve conjugacy and integrate seamlessly with marginal likelihood optimisation.
