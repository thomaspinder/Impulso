# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary (confirmed 2026-08-12): practicing macro and finance economists and econometricians who want
Bayesian VARs without writing PyMC — people who think in variables, lags, and shocks, not tensors and
MCMC chains.

Secondary audiences observed in repository evidence (not ranked by the user): technical due-diligence
reviewers assessing whether the methods are inspectable and correct; researchers and students
replicating structural-VAR papers (e.g. Känzig 2021); downstream library authors (pymc-marketing
consumes a Stage-2 VARX in two MMM notebooks).

## Product Purpose

Impulso is an open-source Python library for Bayesian Vector Autoregression (VAR) built on PyMC. It
gives economists full posterior inference — informative priors, structural identification, impulse
responses, forecasting with credible intervals — behind a validated, immutable, economist-friendly
API.

The docs site (this surface) is upper-funnel marketing (confirmed 2026-08-12): its job is to build
credibility for the project within the econometrics and statistics communities. It is not a
conversion surface and does not need explicit calls to action.

## Positioning

Economist-friendly Bayesian VAR with honest uncertainty: a validated immutable pipeline
(`VARData → VAR → FittedVAR → IdentifiedVAR`), Minnesota-prior defaults with typed escape hatches,
and full posterior uncertainty on every output. Sits between frequentist VAR tooling (statsmodels)
and hand-rolled PyMC models: Bayesian rigour without writing a model function.

The 2026-07-08 GTM plan (services-led consultancy; three-layer open-library / private-connectors /
case-studies stack; climate-conditioned macro risk as lead domain) is **partially superseded**. The
confirmed posture is OSS-credibility-first with this site as upper funnel. Which GTM specifics still
bind is an open decision (see Capabilities and Constraints).

## Operating Context

Users work in Python and Jupyter on macroeconomic and financial time series (FRED-style data),
producing forecasts, impulse responses, and decompositions for research and institutional forecasting
work.

The docs surface: Sphinx + MyST-NB, Diátaxis layout (tutorials / how-to / explanation / reference),
`shibuya` theme, hosted at https://thomaspinder.github.io/Impulso/. Tutorials are jupytext
`py:percent` notebooks executed at build time via jupyter-cache; no rendered outputs are committed.
CI smoke-renders docs on PRs (`IMPULSO_DOCS_CI=1`) and full-renders with real MCMC on push to
`main`. Docs figures use qc_core's ledger plotting style (adopted repo-wide, commit 8c9e66b) — an
observed convention, not confirmed as a binding brand commitment.

## Capabilities and Constraints

- Capabilities: Bayesian VAR/VARX estimation (NUTS via PyMC, optional nutpie), Minnesota and
  conjugate priors, automatic lag selection (AIC/BIC/HQ), probabilistic forecasting, structural
  identification (Cholesky, sign, zero-sign, long-run, proxy-SVAR), IRF / FEVD / historical
  decomposition, scenario analysis, stochastic volatility, model checking, Granger causality,
  stationarity testing.
- Python >=3.11; PyMC/ArviZ pinned as matched pairs per Python version (see pyproject.toml).
- Pre-v0.1 policy: breaking public-API changes are allowed when well-justified and documented.
- Extension seams are public protocols: `Prior`, `Sampler`, `IdentificationScheme`,
  `VolatilityProcess`.
- Docs constraint: heavy tutorials are bounded by CI (limited cores, nutpie); build correctness is
  gated by `make docs-ci` (warnings as errors).
- Open decisions (recorded, not invented):
  - Whether climate-conditioned macro risk remains the lead case-study domain (GTM §8, not
    reconfirmed on 2026-08-12).
  - Consulting-CTA removal (resolved for the docs surface 2026-08-12): the CTA blocks in
    `docs/index.md` and all seven tutorial notebooks are removed, with their styling support,
    to match the no-explicit-CTAs posture. `README.md` carries its own CTA block and remains a
    separate open decision.
  - Binding status of the AI-driven-development disclosure in `README.md`.

## Brand Commitments

- Name: Impulso.
- Posture (confirmed 2026-08-12): credibility-first, community-facing open source. The docs surface
  carries no explicit calls to action; standing in the econ and stats communities is the goal.

## Evidence on Hand

- Executable tutorials (`docs/tutorials/*.py`): quickstart, minnesota-prior, forecasting,
  structural-analysis, monetary-policy, proxy-svar, scenario-analysis, stochastic-volatility,
  conjugate-var, model-checking, post-march-2020.
- Twelve how-to guides (`docs/how-to/`), including climate-pitfalls, sign-restrictions, and
  long-run-restrictions.
- Explanations (`docs/explanation/`): bayesian-var, identification, minnesota-prior; bibliography in
  `docs/references.bib`.
- ADRs in `docs/adr/`, domain glossary in `CONTEXT.md`, design history in `plans/`.
- `papers/` and `demos/` directories exist as untracked work-in-progress.
- Absences future work must not fabricate: no published testimonials, client case studies,
  benchmarks, or pricing.

## Product Principles

1. Credibility over conversion — the docs earn standing in the econ and stats communities; upper
   funnel, no hard sells.
2. Speak the economist's language — variables, lags, and shocks, never tensors and chains.
3. Honest uncertainty — full posteriors and calibrated bands; never overstate what the model knows.
4. Show, don't assert — docs execute their notebooks at build time; every result on the page is
   reproduced, not pasted.
5. The methods ship open — credibility depends on the methodology being fully inspectable.
