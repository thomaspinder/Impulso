# Deterministic Regressors

Calendar-anchored exogenous design matrices: `Trend`, `Fourier`,
`SeasonalDummies` and `BreakDummy`, composed by `DeterministicDesign` into the
exogenous block of a `VARData`. Elapsed time is counted in integer period
ordinals from the first timestamp of the estimation index, so `extend` writes
exactly the rows `build` would have written on a longer sample — which is what
lets `exog_future` hand `FittedVAR.forecast` a column-aligned block.

See the how-to guide, [Deterministic Regressors for Climate
VARs](../how-to/deterministic-regressors.md), for the recipe.

```{eval-rst}
.. currentmodule:: impulso.deterministic

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DeterministicDesign
   Trend
   Fourier
   SeasonalDummies
   BreakDummy
```
