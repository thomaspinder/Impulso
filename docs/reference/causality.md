# Granger Causality

Directional predictive-strength queries on a fitted VAR. The per-pair query
lives on `FittedVAR.granger_causality`; `toda_yamamoto` is the lag-augmented
entry point for possibly-integrated systems.

`toda_yamamoto` resolves its augmentation from the integration-order
diagnostics unless you pass `d=` yourself, so that path needs the optional
`diagnostics` extra:

```
pip install "impulso[diagnostics]"
```

```{eval-rst}
.. currentmodule:: impulso

.. autosummary::
   :toctree: generated/
   :nosignatures:

   toda_yamamoto
```

Both entry points return a
{class}`~impulso.results.GrangerCausalityResult`, documented on the
[Results](results.md) page — read its docstring for exactly what `p_rope`
claims and what it does not.

See [Granger causality and Toda-Yamamoto](../how-to/granger-causality.md)
for the recipes.
