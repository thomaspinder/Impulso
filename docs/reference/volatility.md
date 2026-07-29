# Volatility Processes

The volatility argument to `VAR` is a seam: the model specification owns the
conditional mean, and a volatility process owns the residual covariance. Every
adapter implements the `VolatilityProcess` protocol on the
[protocols](protocols.md) page, which is what lets downstream code reconstruct
the Cholesky factor at a given period without knowing which process produced
it. `Constant` is the homoscedastic default — one Σ shared across all
periods, built from a HalfCauchy prior on the diagonal scales and a Normal
prior on the lower-triangular off-diagonals. `StochasticVolatility` makes Σ_t
time-varying by placing latent log-volatility dynamics (a random walk or an
AR(1)) on each variable's scale; it doubles as a standalone univariate
stochastic-volatility model with its own fit and forecast surface.

Both adapters are sampled with PyMC, so they belong to the `VAR` path. The
conjugate estimator cannot use them — a closed-form Normal-Inverse-Wishart
posterior admits only a deterministic scale path — and rejects them at
construction. Its deterministic counterparts, `ConjugateVolatility` and
`PandemicBreak`, are documented on the [conjugate](conjugate.md) page.

```{eval-rst}
.. currentmodule:: impulso.volatility

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Constant

.. currentmodule:: impulso.sv.spec

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StochasticVolatility
```
