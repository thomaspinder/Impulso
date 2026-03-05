# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Impulso is a Python library for Bayesian Vector Autoregression (VAR). Early stage (v0.0.2), scaffolded from cookiecutter-uv.

## Commands

```bash
# Install dependencies and prek hooks
make install

# Run all code quality checks (lock file, pre-commit, type checking)
make check

# Run tests with coverage
make test

# Run a single test
uv run python -m pytest tests/test_foo.py::test_foo -v

# Run only fast tests (skip MCMC sampling)
uv run python -m pytest -m "not slow" -v

# Type check only
uv run ty check

# Lint/format only
uv run ruff check . && uv run ruff format .

# Build docs locally
make docs

# Test docs build
make docs-test

# Multi-version test (Python 3.11-3.14)
uv run tox
```

## Architecture

- **Source**: `src/impulso/` — library code, built as a wheel via Hatchling
- **Tests**: `tests/` — pytest with `--cov`, 90% coverage target (codecov.yaml)
- **Docs**: `docs/` — MkDocs Material, docstrings auto-rendered via mkdocstrings

### Core Pipeline

The library follows an immutable pipeline where each stage produces the next:

```
VARData → VAR.fit() → FittedVAR → .set_identification_strategy() → IdentifiedVAR
```

- **`VARData`** (`data.py`): Validated, frozen Pydantic model holding endogenous/exogenous arrays + DatetimeIndex. Constructor `from_df()` for DataFrame input.
- **`VAR`** (`spec.py`): Model specification (lags, prior). Calling `.fit(data, sampler)` builds a PyMC model, samples, and returns `FittedVAR`.
- **`FittedVAR`** (`fitted.py`): Reduced-form posterior. Provides `.forecast(steps)` → `ForecastResult` and `.set_identification_strategy(scheme)` → `IdentifiedVAR`.
- **`IdentifiedVAR`** (`identified.py`): Structural VAR. Provides `.impulse_response()` → `IRFResult`, `.fevd()` → `FEVDResult`, `.historical_decomposition()` → `HistoricalDecompositionResult`.

### Extensibility via Protocols

Three `Protocol` classes in `protocols.py` define the extension points:

- **`Prior`**: must implement `build_priors(n_vars, n_lags) → dict`. Concrete: `MinnesotaPrior`.
- **`Sampler`**: must implement `sample(model) → InferenceData`. Concrete: `NUTSSampler`.
- **`IdentificationScheme`**: must implement `identify(idata, var_names) → InferenceData`. Concrete: `Cholesky`, `SignRestriction`.

### Result Objects

All result types (`results.py`) inherit from `VARResultBase` and provide `.median()`, `.hdi()`, `.to_dataframe()`, and `.plot()`. Plot methods delegate to `plotting/` subpackage.

### Key Patterns

- **All models are frozen Pydantic `BaseModel`s** with `arbitrary_types_allowed=True`. Use `object.__setattr__` only for internal setup (e.g., making arrays read-only in validators).
- **Lazy imports**: PyMC, plotting modules, and cross-module types are imported inside methods to avoid circular imports and reduce import time.
- **Prior registry**: `spec.py` maps string shorthands (e.g., `"minnesota"`) to `Prior` classes via `_PRIOR_REGISTRY`.

## Tooling

- **Package manager**: uv (lock file must stay in sync — `uv lock --locked`)
- **Linter/Formatter**: Ruff — line length 120, target py311, auto-fix enabled
- **Type checker**: ty (configured for `.venv`, Python 3.11)
- **Prek**: Ruff checks + standard hooks (trailing whitespace, TOML/YAML/JSON validation)
- **CI**: GitHub Actions runs quality, tests (3.11–3.14), and docs checks on push/PR

## Code Conventions

- Ruff lint rules: `YTT, S, B, A, C4, T10, SIM, I, C90, E, W, F, PGH, UP, RUF, TRY`
- `assert` is allowed in tests (`S101` ignored for `tests/*`)
- `E501` (line length), `E731` (lambda assignment), and `TRY003` (long exception messages) are globally ignored
- Docstrings follow Google style (Args/Returns sections)

## PyMC / Sampling Gotchas

- **LKJCholeskyCov/LKJCorr are broken** with PyMC 5.28 + PyTensor 2.38 + NumPy 2.4 (einsum unpacking bug). Use manual Cholesky parameterization instead (HalfCauchy diagonal + Normal off-diagonal), as done in `spec.py`.
- **Parallel sampling segfaults**: Use `cores=1` in `NUTSSampler` for tests to avoid multiprocessing crashes. Tests marked `@pytest.mark.slow` exercise MCMC.

## PR Review Policy

When reviewing pull requests, structure all feedback into two categories:

### In-Scope (block merge until resolved)
- Bugs, logic errors, or correctness issues in the diff
- Violations of the coding standards defined in this file
- Security concerns in changed code
- Missing or broken tests for changed behaviour
- Type errors, API-contract mismatches, or CI-breaking changes

### Out-of-Scope (track as separate issues)
- Pre-existing tech debt not introduced by this PR
- Refactoring in unchanged code
- Performance improvements unrelated to the PR
- Feature ideas or enhancements
- Documentation gaps outside changed files
- Architectural concerns requiring design discussion

**Default to out-of-scope when uncertain.** Do not block PRs for tangential work.

Format in-scope items as a checklist with severity (P0/P1/P2) and line references.
Format out-of-scope items as complete issue drafts with title, labels, context,
problem description, suggested approach, and affected files.

Use the `/pr-review` command for the full review workflow.
