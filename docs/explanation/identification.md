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
## Zero and sign restrictions together

Cholesky and sign restrictions sit at opposite ends of one spectrum. Cholesky imposes $n(n-1)/2$ zeros, which is exactly enough to pin a single answer. Sign restrictions impose no zeros at all and leave a set of answers. Most credible identification schemes sit in between: a few zeros you can defend on institutional or physical grounds, plus signs on the responses you have a firm prior about.

`ZeroSignRestriction` covers that middle ground, following {cite:t}`ariasRubioRamirezWaggoner2018`:

```python
from impulso.identification import ZeroSignRestriction

scheme = ZeroSignRestriction(
    shock_names=["supply", "demand", "policy"],
    zero_restrictions={"gdp": ["policy"]},
    sign_restrictions={
        "gdp":       {"supply": "+", "demand": "+"},
        "inflation": {"supply": "-", "demand": "+", "policy": "-"},
    },
)
```

This says output does not respond to a policy shock within the period, on top of the usual signs.

### How the construction works

Write the structural impact matrix as $P = LQ$, where $L$ is the Cholesky factor of the residual covariance and $Q$ is orthogonal. The response of variable $i$ to shock $j$ on impact is $e_i' L q_j$, so a zero restriction is a *linear* condition on the $j$-th column of $Q$:

$$Z_j L q_j = 0$$ (eq-zero-condition)

where $Z_j$ selects the rows carrying zeros on shock $j$. Because {eq}`eq-zero-condition` is linear, it does not have to be searched for. Columns are built one at a time, in decreasing order of how many zeros they carry. At step $k$ the column must satisfy its own zero conditions *and* be orthogonal to the $k-1$ columns already drawn:

$$R_k = \begin{pmatrix} Z_k L \\ q_1' \\ \vdots \\ q_{k-1}' \end{pmatrix}, \qquad q_k \sim \text{Uniform}\!\left(\mathcal{S} \cap \mathcal{N}(R_k)\right)$$ (eq-arw-recursion)

with $\mathcal{N}(R_k)$ the null space of $R_k$ and $\mathcal{S}$ the unit sphere. Drawing a standard Gaussian in an orthonormal basis of that null space and normalising gives the uniform draw. The zeros then hold to the precision of the singular value decomposition (SVD) used to find the basis, and orthogonality holds by construction, so the accept/reject step only ever has to test the *signs*.

### The rank condition

$R_k$ has $z_k + (k-1)$ rows, where $z_k$ is the number of zeros on the $k$-th shock, so its null space is non-trivial only when

$$z_j \le n - j, \qquad j = 1, \dots, n$$ (eq-rwz-rank)

with shocks indexed in decreasing order of $z_j$. This is the counting condition of {cite:t}`rubioRamirezWaggonerZha2010`. It depends on the restriction pattern alone, so Impulso checks it once before any sampling and raises a `ValueError` naming the offending shock. Equality throughout — $z_j = n - j$ for every shock — makes every null space one-dimensional, the answer unique up to column signs, and reproduces the Cholesky factor. That is the exactly-identified end of the spectrum; anything looser leaves a set.

:::{admonition} Draws are unweighted
:class: warning
Accepted draws are kept as the recursion produced them, with no importance weight. {cite:t}`ariasRubioRamirezWaggoner2018` derive such a weight — a volume-element correction for the manifold the zero restrictions carve out — which their uniform-conditional prior over the identified set requires. Without it, a set-identified posterior from `ZeroSignRestriction` is not that prior exactly.

Two cases are unaffected. With no zero restrictions the recursion reduces to Gram-Schmidt on Gaussian columns, which is exactly the Haar measure. With zeros that exactly identify the system, the answer is a point up to column signs, and no weight can move a point. In between, treat the spread across draws as reflecting the restrictions and the parameter posterior, not a calibrated prior over rotations.

Two further differences from the paper are worth stating. Impulso retries rotations *within* each posterior draw, as `SignRestriction` does, rather than rejecting the joint $(\theta, Q)$ pair; and it draws from the orthogonal group $O(n)$ rather than the special orthogonal group $SO(n)$, which is immaterial whenever at least one column's sign is left unpinned.
:::
