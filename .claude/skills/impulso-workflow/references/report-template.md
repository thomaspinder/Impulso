# Audit Report Template

Generate the report in EXACTLY this format. Replace placeholders with actual findings.

```markdown
# Impulso Codebase Audit Report

**Date**: YYYY-MM-DD
**Commit**: <short SHA from `git rev-parse --short HEAD`>
**Impulso version**: <from pyproject.toml>
**PyMC version**: <pinned version>
**Pandas version**: <pinned version>
**Python target**: <requires-python value>

---

## Executive Summary

<2–4 sentences: overall health of the codebase, most critical findings, and top recommendation.>

---

## What's Working Well

<Bullet list of 3–6 genuine strengths. Examples: good test coverage of core pipeline, clean Pydantic model design, well-structured CI pipeline. Only list things that are genuinely strong — do not pad this section.>

---

## Findings

### 1. Code Standards

#### CS-<NNN> [P<0-3>]: <Short title>

**What**: <1–2 sentences describing the issue. Include file paths and line numbers.>

**Why it matters**: <1–2 sentences on impact — correctness, maintainability, performance, developer experience.>

**Proposed solution**: <Concrete actionable steps. Include code snippets if helpful.>

---

<Repeat for each code standards finding, numbered sequentially CS-001, CS-002, etc.>

### 2. Pydantic / Models

#### PD-<NNN> [P<0-3>]: <Short title>

**What**: <Description of the Pydantic usage issue. Include model class names, file paths, and line numbers.>

**Why it matters**: <Impact on validation, performance, serialization, or maintainability.>

**Proposed solution**: <Concrete steps. Include code snippets showing before/after if helpful.>

---

<Repeat for each Pydantic finding, numbered sequentially PD-001, PD-002, etc.>

### 3. Test Quality

#### TQ-<NNN> [P<0-3>]: <Short title>

**What**: <Description with specifics.>

**Why it matters**: <Impact statement.>

**Proposed solution**: <Actionable steps.>

---

<Repeat for each test finding.>

### 4. CI/CD and Tooling

#### CI-<NNN> [P<0-3>]: <Short title>

**What**: <Description.>

**Why it matters**: <Impact.>

**Proposed solution**: <Steps.>

---

### 5. Documentation and Type Hints

#### DT-<NNN> [P<0-3>]: <Short title>

**What**: <Description.>

**Why it matters**: <Impact.>

**Proposed solution**: <Steps.>

---

### 6. Upstream Dependencies

#### UP-<NNN> [P<0-3>]: <Short title>

**What**: <Description of the upstream feature/change and how it relates to Impulso's current code.>

**Why it matters**: <What Impulso gains by adopting this — performance, API quality, future-proofing, new capabilities.>

**Proposed solution**: <Concrete migration/adoption steps. Reference upstream docs/examples.>

---

## Summary Table

| ID | Priority | Category | Title |
|----|----------|----------|-------|
| CS-001 | P1 | Code Standards | ... |
| PD-001 | P2 | Pydantic / Models | ... |
| TQ-001 | P2 | Tests | ... |
| CI-001 | P2 | CI/CD | ... |
| DT-001 | P3 | Docs / Types | ... |
| UP-001 | P1 | Upstream | ... |
| ... | ... | ... | ... |

## Recommended Action Order

<Numbered list of the top 5–10 findings to address first, considering priority AND dependency order (e.g., "bump Python minimum to 3.11 before updating PyMC/Pandas pins").>
```

## Priority Definitions

- **P0 — Critical**: Broken behaviour, security issue, deprecated API that will break on next dependency update. Fix immediately.
- **P1 — High**: Significant quality/maintainability issue, missing tests for critical paths, upstream feature that closes a known gap. Fix this release cycle.
- **P2 — Medium**: Modernisation opportunity, test improvement, non-critical upstream feature. Plan for next 1–2 release cycles.
- **P3 — Low**: Style nitpick, nice-to-have improvement, upstream feature of marginal value. Address opportunistically.

## Quantity Guidelines

Aim for a focused, high-signal report:
- **P0**: 0–3 findings (these should be rare)
- **P1**: 3–8 findings
- **P2**: 5–12 findings
- **P3**: 3–8 findings
- **Total**: 15–30 findings is the sweet spot. More than 40 suggests insufficient prioritisation.
