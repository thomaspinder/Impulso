# Diagnostics

Convergence and dynamic-stability diagnostics for a fitted VAR posterior.
`convergence_report` reports R-hat and effective sample size *per parameter
block* with the offending coordinate named, counts divergences globally, and
summarises the posterior distribution of the companion-matrix spectral
radius. Reach for it through `FittedVAR.convergence_report()` or
`IdentifiedVAR.convergence_report()`; the free function is the entry point
for posteriors built by hand.

```{eval-rst}
.. currentmodule:: impulso.diagnostics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   convergence_report
   ConvergenceReport
   BlockDiagnostics
   StabilitySummary
   DiagnosticMessage
   ConvergenceThresholds
   assign_blocks
```
