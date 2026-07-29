# Scenario Conditions

The condition vocabulary for scenario analysis. `ShockPath` sets a
structural shock's path (in-sample counterfactual edits; forecast-side
prescriptions for `structural_scenario`) and `VariablePath` pins a future
endogenous path (`conditional_forecast`, `structural_scenario`) — both
hard conditions, holding on every draw. The *targets* are soft: they state
a fact about the forecast distribution that entropic tilting imposes by
reweighting draws (`ProbabilityTarget` for an event probability,
`MomentTarget` for a mean).

```{eval-rst}
.. currentmodule:: impulso.scenario

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ShockPath
   VariablePath
   ProbabilityTarget
   MomentTarget
```
