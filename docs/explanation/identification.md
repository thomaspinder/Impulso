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

## Long-run restrictions (Blanchard-Quah)

Cholesky and sign restrictions both constrain the **contemporaneous** (impact) response to structural shocks. Long-run restrictions take a different approach: they constrain the **cumulative** response as the horizon goes to infinity.

The key object is the **long-run multiplier matrix**:

$$C(1) = (I - A_1 - A_2 - \cdots - A_p)^{-1}$$

This matrix captures the total cumulative effect of a one-time shock. If the VAR is stationary, all shocks are transitory and $C(1)$ is finite. The long-run impact of structural shocks is $C(1) P$, where $P$ is the structural impact matrix.

Blanchard & Quah (1989) proposed forcing $C(1) P$ to be **lower triangular**. This is achieved by applying the Cholesky decomposition not to the residual covariance $\Sigma$ (as in short-run Cholesky) but to the long-run covariance:

$$C(1) \, \Sigma \, C(1)^\top = L \, L^\top$$

The structural impact matrix is then $P = C(1)^{-1} L$.

The interpretation depends on the variable ordering:
- Shocks later in the ordering have **zero long-run cumulative effect** on variables earlier in the ordering
- The first shock is unrestricted in its long-run effects

This is exactly the same Cholesky math, applied to a different matrix. The economics, however, are very different: you're restricting permanent effects rather than contemporaneous effects. In the classic Blanchard-Quah example, ordering output before prices means the second shock (interpreted as a demand shock) has no permanent effect on output — only supply shocks do.
