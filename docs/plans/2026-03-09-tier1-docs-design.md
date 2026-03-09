# Tier 1 Extensions Documentation Design

**Date**: 2026-03-09
**Goal**: Document the 5 Tier 1 extensions (ConjugateVAR, dummy obs, GLP, long-run restrictions, conditional forecasting) across all Diataxis layers.

## Audience

Both applied economists/central bank researchers (know VAR theory, learning the API) and Python data scientists (comfortable with Python, may need econometric concepts explained).

## File Map

### New files
- `docs/how-to/conjugate-estimation.md` — ConjugateVAR + GLP + dummy obs workflow
- `docs/how-to/long-run-restrictions.md` — Blanchard-Quah identification
- `docs/how-to/conditional-forecasting.md` — conditional forecasts on FittedVAR and IdentifiedVAR
- `docs/reference/conjugate.md` — mkdocstrings for `impulso.conjugate`
- `docs/reference/conditions.md` — mkdocstrings for `impulso.conditions`
- `docs/explanation/conditional-forecasting.md` — Waggoner-Zha theory

### Modified files
- `docs/index.md` — update features list and hero example
- `docs/explanation/minnesota-prior.md` — add conjugate NIW + GLP + dummy obs theory
- `docs/explanation/identification.md` — add long-run restrictions theory
- `mkdocs.yml` — add new pages to nav

## Landing Page

Update `docs/index.md`:
- Add ConjugateVAR to the hero code example showing the fast estimation path alongside existing PyMC path
- Update features list to include: conjugate estimation, data-driven prior selection, long-run identification, conditional forecasting, dummy observation priors

## How-To Guides

All guides are pedagogical — explain *why* at each step, not just *how*. Each code block preceded by a paragraph explaining the motivation. Use admonitions for practical insights. Interpret output, don't just show it. Close with "When to use this" comparing alternatives.

### "Fast Estimation with ConjugateVAR" (`conjugate-estimation.md`)
- Open with the problem: NUTS is slow, conjugacy gives closed-form posterior
- Walk through: basic fit → explain draws → show forecasts work identically
- GLP section: explain hyperparameter subjectivity problem, show `optimize_prior()`, interpret results, explain marginal likelihood intuitively
- Dummy obs section: economic intuition (persistence/unit root beliefs), show `with_dummy_observations()`, explain mu/delta economically
- Compare: when ConjugateVAR vs VAR+NUTS

### "Long-Run Restrictions" (`long-run-restrictions.md`)
- Open with the problem: Cholesky assumes contemporaneous ordering, but theory may specify long-run effects instead
- Walk through: construct scheme, explain ordering meaning economically
- Show and interpret IRFs: long-run restriction visible in convergence to zero
- Tips: stationarity requirement, ordering sensitivity, when to prefer over alternatives

### "Conditional Forecasting" (`conditional-forecasting.md`)
- Open with the problem: standard forecasts are unconditional, policy analysis needs "what if" scenarios
- Reduced-form example: condition on interest rate path
- Structural example: condition on structural shock paths
- Interpret results: constrained variable hits target, unconstrained variables show conditional prediction
- Tips: hard constraints, degrees of freedom considerations

## Explanation Page Extensions

### Extend `minnesota-prior.md`
- **Conjugate estimation**: NIW posterior update equations in LaTeX, intuitive explanation ("blends prior with OLS, weighted by precision")
- **Data-driven prior selection (GLP)**: marginal likelihood concept, why it works (penalises under/overfitting), closed-form availability from conjugacy
- **Dummy observation priors**: the trick of encoding beliefs as fake data rows, sum-of-coefficients (persistence), single-unit-root (random walk), mu/delta control strength

### Extend `identification.md`
- **Long-run restrictions (Blanchard-Quah)**: identify via cumulative long-run effects, C(1) = (I - A_1 - ... - A_p)^{-1}, Cholesky on long-run covariance, contrast with short-run Cholesky

### New `explanation/conditional-forecasting.md`
- Waggoner-Zha (1999) algorithm: unconditional forecast + MA representation + linear constraint system
- Structural extension: conditioning on structural shock paths
- Connection to scenario analysis and counterfactual exercises

## Reference Pages

Two new pages following existing pattern:
- `docs/reference/conjugate.md` — `::: impulso.conjugate`
- `docs/reference/conditions.md` — `::: impulso.conditions`

Existing reference pages (`identification.md`, `fitted.md`, `identified.md`, `results.md`) auto-pick up new classes via mkdocstrings.

## mkdocs.yml Nav

Add to nav:
- How-To: conjugate-estimation, long-run-restrictions, conditional-forecasting
- Explanation: conditional-forecasting
- Reference: conjugate, conditions
