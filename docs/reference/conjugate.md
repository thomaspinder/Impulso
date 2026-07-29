# Conjugate VAR

`ConjugateVAR` is the sibling estimator to `VAR`. It pairs a natural-conjugate
Normal-Inverse-Wishart prior (`NIWPrior`) with the VAR likelihood, so the
coefficient and covariance posterior is available in closed form and no PyMC
model is ever built; Monte Carlo is reserved for the single low-dimensional
Minnesota tightness hyperparameter, which the data selects by marginal
likelihood {cite:p}`giannoneLenzaPrimiceri2015`. Prefer it over the
NUTS-sampled `VAR` when the system is large, when the fit has to be repeated
many times (rolling windows, real-time exercises), or when the closed-form
marginal likelihood is itself the quantity of interest. Stay on `VAR` when you
need per-equation own/cross shrinkage asymmetry, stochastic volatility, or a
prior the conjugate Kronecker structure cannot express. Both paths return the
same `FittedVAR`, so identification, impulse responses, variance
decompositions, and forecasting behave identically downstream.

Time-varying volatility on the conjugate path is deterministic rather than
latent, which is what keeps the posterior closed-form. `ConjugateVolatility`
is the adapter surface a conjugate fit attaches when the residual scale
breaks: it reports a per-period multiplier `s_t` on a base Cholesky factor, so
`Sigma_t = s_t**2 * Sigma_base`. `PandemicBreak` is the concrete adapter, the
COVID-19 break of {cite:t}`lenzaPrimiceri2022` — free outbreak scales for
March, April, and May 2020 followed by a geometric decay back toward the
pre-break scale. The latent volatility processes used by the `VAR` path live
on the [volatility](volatility.md) page.

```{eval-rst}
.. currentmodule:: impulso.conjugate

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ConjugateVAR

.. currentmodule:: impulso.priors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NIWPrior

.. currentmodule:: impulso.conjugate_volatility

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ConjugateVolatility
   PandemicBreak
```
