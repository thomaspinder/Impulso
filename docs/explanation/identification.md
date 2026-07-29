# Structural Identification

A reduced-form VAR estimates the joint dynamics of a set of variables, but its residuals are correlated. **Structural identification** decomposes these correlated residuals into uncorrelated structural shocks with economic interpretations.

## Why identification matters

Without identification, you can describe correlations but not causation. Identification lets you answer:

- "What happens to inflation when there is a monetary policy shock?" (impulse responses)
- "How much of GDP variation is due to supply vs demand shocks?" (variance decomposition)

## Cholesky identification

The simplest approach. Uses the lower-triangular Cholesky factor of the residual covariance matrix. This implies a **recursive causal ordering**: the first variable is not contemporaneously affected by any other, the second is affected only by the first, and so on.

```python
from impulso.identification import Cholesky

scheme = Cholesky(ordering=["gdp", "inflation", "rate"])
```

The ordering encodes your assumptions. Changing it changes the results.

:::{admonition} Ordering sensitivity
:class: info
The ordering encodes your causal assumptions. Changing the ordering
changes the results — not because the data changed, but because the
identifying restrictions changed. Always justify your ordering with
domain knowledge.
:::
## Sign restrictions

A more agnostic approach. Instead of imposing a full recursive structure, you specify qualitative constraints: "a supply shock raises GDP and lowers inflation." The algorithm searches over random rotation matrices to find decompositions consistent with your restrictions.

```python
from impulso.identification import SignRestriction

scheme = SignRestriction(
    restrictions={
        "gdp":       {"supply": "+", "demand": "+"},
        "inflation": {"supply": "-", "demand": "+"},
    },
)
```

Sign restrictions are weaker than Cholesky (they don't uniquely identify the model), but they require fewer assumptions.

## Long-run restrictions

Cholesky and sign restrictions both constrain the *impact* matrix $\Theta(0) = P$ — what happens on the day of the shock. Long-run restrictions constrain the *cumulative* matrix $\Theta(1)$ instead: the total effect of each shock on the level of each variable, summed over all horizons. Adding up the moving-average coefficients of a stable VAR gives the long-run multiplier $C(1) = \sum_h \Phi_h = (I - \sum_j A_j)^{-1}$, and the cumulative structural impact is $\Theta(1) = C(1) P$.

The restriction is that $\Theta(1)$ is lower-triangular in the order you supply: a shock has no long-run effect on any variable ordered before it. Two variables, one restriction — the {cite:t}`blanchardQuah1989` case:

```python
from impulso.identification import LongRunRestriction

scheme = LongRunRestriction(
    ordering=["output_growth", "unemployment"],
    shock_names=["supply", "demand"],
)
```

Here the single zero says the demand shock has no permanent effect on the level of output. The supply shock is unrestricted, and both shocks may do whatever they like to unemployment.

Nothing is searched for. Since $\Theta(1)\Theta(1)' = C(1)\Sigma C(1)'$, the lower-triangular positive-diagonal $\Theta(1)$ is that matrix's Cholesky factor, and $P$ follows. So the scheme is a *rotation of the Cholesky factor* — the same object sign restrictions sample over — picked out in closed form. The positive diagonal is the sign convention: shock $j$ raises variable $j$'s long-run level. The diagonal of $P$ itself may be negative, since the normalisation applies to the cumulative matrix rather than to impact.

:::{admonition} Your variables must be differenced
:class: warning
$C(1)$ is the long-run multiplier on the levels of the variables *as modelled*. "No permanent effect on output" is a statement about the level of output, so output must enter the VAR as a growth rate — otherwise the restriction says something else entirely. Impulso cannot check this for you.
:::

Triangularity is $n(n-1)/2$ restrictions, exactly the number needed for point identification. With two variables that is the one restriction you wanted. With three it asserts three zeros at once, which is a much stronger joint claim than it looks. Arbitrary (non-recursive) long-run zero patterns are not supported.

Two things can go wrong, and Impulso reports them separately. If $I - \sum_j A_j$ is close to singular, $C(1)$ is numerically undefined and those draws are blanked (or, with `on_undefined="raise"`, refused). If a draw is explosive — companion spectral radius above one — the arithmetic is fine but $\sum_h \Phi_h$ diverges, so "long-run effect" has no meaning for it; those draws are always reported and never blanked, because posteriors near a unit root routinely contain them.
