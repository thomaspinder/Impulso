# Posterior predictive is computed in NumPy, not on the PyMC graph

`FittedVAR.posterior_predictive` reconstructs in-sample replicates from the posterior arrays — conditional mean from `B`/`intercept`/`B_exog`, innovations from the volatility seam's `cholesky_path` — rather than calling `pymc.sample_posterior_predictive` on the stored `pymc_model`. `VAR.prior_predictive` does the opposite: it builds the graph and calls `pymc.sample_prior_predictive`.

**Why the asymmetry**: a prior predictive *is* the prior, and the prior only exists in the graph — hand-rolling it is precisely the duplication issue #56 complains about. A posterior predictive, by contrast, needs nothing but the posterior draws and Σ, both of which are already NumPy.

**What NumPy buys**:

- **`ConjugateVAR` parity.** `ConjugateVAR` draws in closed form and builds no PyMC graph at all (`pymc_model is None`; ADR-0004). A graph-based implementation would either exclude the sibling estimator or force it to build a graph it otherwise never needs.
- **`simulate_innovations=False`.** The conditional-mean mode is a slice of the same computation, exact and RNG-free. On the graph it would need a separate `pm.do`-style intervention.
- **Backend independence.** No dependence on which NUTS backend produced the draws, on `idata` group layout beyond `posterior`, or on the graph surviving serialisation. `FittedVAR` round-tripped through NetCDF keeps working.
- **The volatility seam.** `volatility.cholesky_path(posterior, T)` already returns `L_t` per draw per `t` (ADR-0001), which is exactly the innovation scale needed. Under stochastic volatility this gives genuinely time-varying replicate spread; the issue's original "residual covariance of each posterior draw" wording would have flattened it.

**Cost accepted — drift.** Two implementations of the same likelihood can diverge: a change to the observation equation (Student-t errors, a mean shift, a new exogenous block) must land in both `spec.py`'s graph and `_residuals.fitted_values`. The fence is `tests/test_predictive.py::TestPredictiveAgainstPyMC::test_matches_sample_posterior_predictive`, a slow test that fits a real VAR and compares the NumPy replicates against `pm.sample_posterior_predictive` on moments. Its fixture uses strongly cross-correlated shocks on purpose: with a near-diagonal Σ, `L` and `L.T` imply nearly the same covariance and a Cholesky-orientation bug would slip through.

**Considered and rejected**: routing through `pymc_model` when it is present and falling back to NumPy when it is not. Two code paths returning subtly different objects for the same public method is worse than one path plus a fence — and it would make `simulate_innovations=False` mean different things depending on the estimator.

**Side benefit**: nutpie needs no caveat. The replicates never touch the sampler's backend, so a nutpie-sampled `idata` and a PyMC-sampled one behave identically here.
