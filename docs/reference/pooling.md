# Predictive Pooling

Combine several fitted models into one predictive distribution. `pool_forecasts`
forecasts every candidate from their shared forecast origin, scores those
densities against a held-out window, and turns the resulting log-score matrix
into weights — by stacking (the log score of the *pooled* predictive, maximised
over the simplex) or by a softmax of each model's total score.

The returned `PredictivePool` carries the weights, the full score matrix, and a
pooled predictive sample over the held-out window. Its weights are frozen once
estimated: `PredictivePool.combine` applies them to new forecasts, which is how
you get a genuine out-of-sample forecast from full-sample refits without
rescoring anything.

```{eval-rst}
.. currentmodule:: impulso.pooling

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pool_forecasts
   PredictivePool
```
