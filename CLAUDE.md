# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Litterman is a Python library for Bayesian Vector Autoregression (VAR). Early stage (v0.0.1), scaffolded from cookiecutter-uv.

## Commands

```bash
# Install dependencies and pre-commit hooks
make install

# Run all code quality checks (lock file, pre-commit, type checking)
make check

# Run tests with coverage
make test

# Run a single test
uv run python -m pytest tests/test_foo.py::test_foo -v

# Type check only
uv run ty check

# Lint/format only
uv run ruff check . && uv run ruff format .

# Build docs locally
make docs

# Test docs build
make docs-test

# Multi-version test (Python 3.10-3.14)
uv run tox
```

## Architecture

- **Source**: `src/litterman/` — library code, built as a wheel via Hatchling
- **Tests**: `tests/` — pytest with `--cov`, 90% coverage target (codecov.yaml)
- **Docs**: `docs/` — MkDocs Material, docstrings auto-rendered via mkdocstrings

## Tooling

- **Package manager**: uv (lock file must stay in sync — `uv lock --locked`)
- **Linter/Formatter**: Ruff — line length 120, target py310, auto-fix enabled
- **Type checker**: ty (configured for `.venv`, Python 3.10)
- **Pre-commit**: Ruff checks + standard hooks (trailing whitespace, TOML/YAML/JSON validation)
- **CI**: GitHub Actions runs quality, tests (3.10–3.14), and docs checks on push/PR

## Code Conventions

- Ruff lint rules: `YTT, S, B, A, C4, T10, SIM, I, C90, E, W, F, PGH, UP, RUF, TRY`
- `assert` is allowed in tests (`S101` ignored for `tests/*`)
- `E501` (line length) and `E731` (lambda assignment) are globally ignored
- Docstrings follow Google style (Args/Returns sections)
