# Covariance factors are parameterised manually, not with `LKJCholeskyCov`

Every Cholesky factor Impulso builds inside PyMC — the constant volatility process's `L`
(`volatility.py:Constant.build_pymc_latent`) and Clark-style SV's correlation factor `R_chol`
(`sv/spec.py:StochasticVolatility.build_pymc_latent`) — is parameterised by hand from primitive
distributions rather than with PyMC's purpose-built `LKJCholeskyCov` / `LKJCorr`.

`LKJCholeskyCov` and `LKJCorr` are broken on the dependency set Impulso supports: an einsum
unpacking bug surfaces with PyMC 5.28 + PyTensor 2.38 + NumPy 2.4. The manual form is a
HalfCauchy on the diagonal scales and a Normal on the lower-triangular off-diagonals, which
builds the same object out of parts that work.

This is recorded because it is a deliberate deviation from the obvious path. A reader who knows
PyMC will see hand-rolled triangular assembly and reasonably assume nobody looked for the
built-in — and "fix" it. The workaround should be revisited, and preferably reverted, once the
upstream bug is resolved across the supported `pymc` range in `pyproject.toml`.

## Consequences

- The two call sites parameterise their factors differently on purpose: `Constant` needs a full
  covariance factor, so its diagonal is free; Clark-style SV needs a *correlation* factor, so its
  Gram diagonal is pinned to 1 and all volatility scaling lives in `h`. Both avoid LKJ for the
  same reason, but neither is a copy of the other.
- The manual form carries no LKJ shape prior over correlations. The off-diagonal Normal is a
  different prior, not a reparameterisation of the same one — switching back to `LKJCholeskyCov`
  later would change posteriors, so it is a breaking change and not a pure refactor.
