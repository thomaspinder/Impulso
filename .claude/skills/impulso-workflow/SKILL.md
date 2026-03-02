---
name: impulso-audit
description: >
  Conduct a comprehensive audit of the Impulso codebase (or a similarly structured probabilistic Python library).
  The audit covers: (1) modern Python code standards, (2) Pydantic model usage, (3) CI/CD pipeline and uv configuration,
  (4) unit test quality and coverage, (5) docstring and type hint completeness,
  (6) upstream dependency analysis for PyMC, ArviZ, Pandas, Pydantic, and NumPy to identify new features or patterns
  Impulso should adopt. Produces a prioritised markdown report at ./reports/audit-YYYY-MM-DD.md.
  Trigger on: "audit impulso", "run audit", "codebase audit", "code review impulso",
  "check upstream deps", "audit my library".
---

# Impulso Codebase Audit

Full-scope audit of a Bayesian Vector Autoregressive (VAR) and structural VAR library. Run against a local checkout.

## Workflow

Execute phases 1–7 in order. Each phase gathers findings. Phase 8 compiles the report.

### Phase 1: Repository Discovery

Understand the repo before auditing. Read these files (skip missing ones):

```
pyproject.toml
setup.cfg / setup.py (if present)
.github/workflows/*.yml
.pre-commit-config.yaml
uv.lock (first 50 lines)
Makefile / justfile (if present)
```

Record:
- Python version constraints (requires-python)
- Dependency list and version pins
- Dev/test dependency groups
- Build backend (hatchling, setuptools, etc.)
- uv workspace configuration if any
- Linter/formatter config (ruff, mypy, ty, etc.)

### Phase 2: Code Standards Audit

Read `references/code-standards.md` for the full checklist, then scan the source tree.

For each `.py` file under the main package directory:
1. Sample files proportionally — read ALL files in small packages (<30 files), or ~40% in larger ones, prioritising core modules (spec, data, fitted, identified, priors, samplers, results).
2. Check against each item in the checklist, including the **Probabilistic Programming / Domain-Specific** and **Protocols and Extensibility** sections.
3. Record concrete findings with file paths and line numbers where possible.

### Phase 3: Pydantic Usage Audit

Read `references/pydantic-checklist.md` for the full checklist, then audit all Pydantic models.

1. Use the **Grep** tool to find all `BaseModel` subclasses: pattern `class \w+\(BaseModel\)` in `src/impulso/`.
2. For each model, check against the checklist items: model_config consistency, Field constraints, validator patterns, serialization, arbitrary_types_allowed usage, v2 feature adoption, and performance.
3. Pay special attention to:
   - Whether a shared base model class would reduce `model_config` duplication
   - Whether `arbitrary_types_allowed` is justified for each model
   - Whether `model_construct()` is used in internal code paths that build models from already-validated data
   - Whether `@computed_field` could replace any manual `@property` definitions
4. Record concrete findings with file paths and line numbers.

### Phase 4: Test Quality Audit

Read `references/testing-checklist.md`, then scan the `tests/` directory.

1. List all test files using the **Glob** tool: pattern `tests/**/*.py`.
2. Identify which source modules have corresponding tests and which do not.
3. Sample test files (same proportional strategy as Phase 2).
4. Evaluate against the checklist.
5. Use the **Grep** tool to count test functions: pattern `def test_` in `tests/`.
6. Check for a coverage configuration and any coverage reports.

### Phase 5: CI/CD and uv Audit

Read `references/cicd-uv-checklist.md`, then inspect:

1. All GitHub Actions workflow files in `.github/workflows/`
2. `pyproject.toml` — build system, scripts, dependency groups
3. `.pre-commit-config.yaml` if present
4. Any `Makefile`, `justfile`, or `noxfile.py`

Evaluate against the checklist.

### Phase 6: Documentation and Type Hints Audit

Scan for:
1. **Docstrings**: Check public classes and functions for missing/incomplete docstrings. Use the **Grep** tool with pattern `def \w+|class \w+` on the source tree to sample, then read those locations.
2. **Type hints**: Check function signatures for missing parameter/return type annotations. Look for `Any` overuse or imprecise types (e.g., `dict` instead of `dict[str, float]`).
3. **Consistency**: Docstring style (Google, NumPy, or Sphinx) should be consistent across the codebase.
4. **Public API**: Check `__init__.py` exports match what's documented.

### Phase 7: Upstream Dependency Analysis

This is critical. Read `references/upstream-analysis.md` for the detailed procedure.

For **PyMC**, **ArviZ**, **Pandas**, **Pydantic**, and **NumPy**:
1. **Web search** recent releases, changelogs, and migration guides (last 6–12 months) using the **WebSearch** tool.
2. **Fetch documentation** directly using **WebFetch** or the **Context7** MCP tools — do NOT clone entire upstream repos.
3. Compare upstream capabilities against Impulso's current usage patterns.
4. Identify features Impulso is NOT using but SHOULD consider.

Focus areas:
- PyMC: inference algorithms, distributions, model primitives, LKJCholeskyCov bug status, new sampling backends
- ArviZ: 1.0 modular restructure, new diagnostics, InferenceData changes
- Pandas: Copy-on-Write default, new string dtype, deprecated APIs
- Pydantic: computed_field, validate_call, TypeAdapter, performance improvements
- NumPy: 2.0 breaking changes, deprecated functions

### Phase 8: Report Generation

Compile all findings into a single prioritised report. Read `references/report-template.md` for the exact output format.

1. Create `./reports/` directory if it doesn't exist.
2. Write the report to `./reports/audit-YYYY-MM-DD.md` using today's date.
3. Assign priority (P0-Critical, P1-High, P2-Medium, P3-Low) to each finding.
4. Sort findings by priority within each section.
5. Each finding MUST include: what the issue is, why it matters, and the proposed solution.

## Important Guidelines

- Be specific: cite file paths, line numbers, function names. Vague findings are useless.
- Be actionable: every finding must have a concrete proposed solution.
- Be honest: if something is already well-done, don't manufacture issues. Note strengths briefly in a "What's Working Well" section.
- Be proportional: don't list 50 P3 nitpicks. Focus on what moves the needle.
- Upstream analysis should be forward-looking: what should Impulso adopt in the NEXT release cycle?
- Use the right tools: Grep and Glob for searching, Read for file contents, WebSearch and WebFetch for upstream research. Avoid unnecessary bash commands.
