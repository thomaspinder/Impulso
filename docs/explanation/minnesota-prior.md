# The Minnesota Prior

The **Minnesota prior** {cite:p}`doan1984,litterman1986` is the most widely used prior for Bayesian VARs, and it is what Impulso applies by default. It encodes the belief that each variable follows a random walk, with coefficients on other variables' lags shrunk toward zero.

This page is the reference statement of the prior. For a worked introduction with figures, prior predictive checks, and a bias–variance experiment, see [The Minnesota Prior, From Scratch](../tutorials/minnesota-prior.py).

## Setup

Impulso estimates the VAR($p$)

$$
y_t = c + \sum_{l=1}^{p} A_l\, y_{t-l} + u_t,
\qquad u_t \sim \mathcal{N}(0, \Sigma),
$$ (eq-mn-var)

with $y_t \in \mathbb{R}^n$ and each $A_l \in \mathbb{R}^{n \times n}$. The lag matrices are stacked into the single coefficient matrix that `VAR.fit` samples,

$$
B = \begin{bmatrix} A_1 & A_2 & \cdots & A_p \end{bmatrix} \in \mathbb{R}^{n \times np}.
$$ (eq-mn-stack)

Write $\beta_{ij}^{(l)}$ for the entry of $A_l$ in row $i$ and column $j$: the coefficient on lag $l$ of variable $j$ in the equation for variable $i$. Columns of $B$ are ordered lag-major — all $n$ variables at lag 1, then all $n$ at lag 2, and so on — so $\beta_{ij}^{(l)}$ sits at column $(l-1)n + j$.

## The prior

Every coefficient gets an independent normal prior,

$$
\beta_{ij}^{(l)} \sim \mathcal{N}\!\left( m_{ij}^{(l)},\; \big(s_{ij}^{(l)}\big)^{2} \right),
$$ (eq-mn-normal)

with a random-walk mean,

$$
m_{ij}^{(l)} =
\begin{cases}
1 & \text{if } i = j \text{ and } l = 1, \\
0 & \text{otherwise,}
\end{cases}
$$ (eq-mn-mean)

and a standard deviation built from three multiplicative factors,

$$
s_{ij}^{(l)} = \lambda \cdot d(l) \cdot
\begin{cases} 1 & i = j \\ \kappa & i \neq j \end{cases},
\qquad
d(l) = \begin{cases} 1/l & \texttt{decay="harmonic"} \\ 1/l^{2} & \texttt{decay="geometric"}. \end{cases}
$$ (eq-mn-sd)

`MinnesotaPrior.build_priors(n_vars, n_lags)` returns {eq}`eq-mn-mean` and {eq}`eq-mn-sd` as the arrays `B_mu` and `B_sigma`, both of shape `(n_vars, n_vars * n_lags)`. `VAR.fit` passes them straight to PyMC as the `mu` and `sigma` of a normal prior on `B`.

## Hyperparameters

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| `tightness` | $\lambda$ | `0.1` | Overall shrinkage. $\lambda \to 0$ freezes the model at the random walk; $\lambda \to \infty$ recovers OLS. Must be $> 0$. |
| `decay` | $d(l)$ | `"harmonic"` | How fast the prior tightens on longer lags. `"harmonic"`: $1/l$. `"geometric"`: $1/l^2$. |
| `cross_shrinkage` | $\kappa$ | `0.5` | Shrinkage on other variables' lags relative to own lags. $\kappa = 0$ reduces the VAR to $n$ independent AR($p$) models; $\kappa = 1$ treats own and cross lags alike. Must lie in $[0, 1]$. |

Rough guidance: tighten $\lambda$ as the system grows, since the number of coefficients rises quadratically in $n$ while the sample does not — {cite:t}`giannoneLenzaPrimiceri2015` show the optimal tightness falls with dimension. Prefer `"geometric"` decay when you have many lags and no reason to expect long-cycle dynamics.

## What the prior implies

The prior mean is a random walk, whose companion matrix has spectral radius exactly one. The Minnesota prior therefore sits **on the boundary of stationarity** by construction, and $\lambda$ controls how far around that boundary the prior mass spreads. It is not a stationarity prior — most prior draws are technically explosive at any $\lambda$, because the largest of several near-unit eigenvalues is biased upward. What small $\lambda$ buys is that they are only barely so: at $\lambda = 0.05$ in a 3-variable VAR(4), the median spectral radius is 1.05 and only about an eighth of draws exceed 1.10. At $\lambda = 1$ the median draw doubles a shock every period.

## Scope and caveats

**No scale adjustment.** The classical Litterman formula multiplies the cross-variable standard deviation by $\sigma_i / \sigma_j$, the ratio of the two variables' residual scales. {eq}`eq-mn-sd` has no such term — `build_priors` receives only `n_vars` and `n_lags` and never sees your data. Put your variables on comparable scales before fitting (standardise them, or express everything in percent). If you want the estimator to handle scaling itself, `NIWPrior` computes per-variable AR(1) residual standard deviations internally.

**Levels, not growth rates.** The mean in {eq}`eq-mn-mean` is a statement about levels. On differenced, de-meaned, or standardised series a prior mean of one on the own first lag is too persistent; the honest centre is nearer zero. Write a custom zero-mean prior instead — see [Writing a Custom Prior](../how-to/custom-priors.md).

**Coefficients only.** `MinnesotaPrior` governs the lag coefficients. Intercepts receive a fixed $\mathcal{N}(0, 1)$ prior, and the residual covariance $\Sigma$ is handled by the volatility process (`Constant` by default).

**Not conjugate.** {eq}`eq-mn-normal` places independent normal priors on the coefficients with a separately parameterised covariance, so the posterior has no closed form and is sampled with NUTS. That independence is what buys per-equation own-versus-cross asymmetry via $\kappa$. The conjugate alternative, `NIWPrior`, gives up that asymmetry in exchange for a closed-form posterior and marginal likelihood — which in turn lets the data select $\lambda$. See [The Conjugate VAR](../tutorials/conjugate-var.py).

## Usage

```python
from impulso import VAR
from impulso.priors import MinnesotaPrior

# Use defaults
spec = VAR(lags=4, prior="minnesota")

# Customize hyperparameters
prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.3)
spec = VAR(lags=4, prior=prior)
```
