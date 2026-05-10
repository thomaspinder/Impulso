# Upstream Dependency Analysis Procedure

## Overview

Identify features in Impulso's key dependencies that it could benefit from but isn't currently using.
Use BOTH web search and direct documentation fetching for thoroughness. Do NOT clone entire upstream repos.

## Step 1: Determine Current Usage

Before checking upstream, understand what Impulso currently uses:

Use the **Grep** tool (not bash grep) to search `src/impulso/` for:
- All PyMC imports: pattern `from pymc|import pymc`
- All ArviZ imports: pattern `from arviz|import arviz`
- All Pandas imports: pattern `from pandas|import pandas`
- All Pydantic imports: pattern `from pydantic|import pydantic`
- All NumPy imports: pattern `from numpy|import numpy`
- All PyTensor imports: pattern `from pytensor|import pytensor`

Record:
- Which PyMC distributions, model-building primitives, and sampling APIs are used
- Which ArviZ functions are used (plotting, diagnostics, InferenceData manipulation)
- Which Pandas APIs are used (DataFrame construction, indexing, time series features)
- Which Pydantic features are used (BaseModel, validators, Field, ConfigDict)
- Current version pins for all dependencies in `pyproject.toml`

## Step 2: Web Search — Recent Developments

Search for recent changes (last 6–12 months) using the **WebSearch** tool:

### PyMC
Search queries to run:
- `pymc changelog 2025 2026`
- `pymc new features release notes`
- `pymc 5.x migration guide`
- `pymc new distributions inference`

Look for:
- New or improved distributions relevant to VAR modelling (e.g., LKJ fixes, multivariate normals)
- New sampling backends or inference algorithms (ADVI improvements, pathfinder, etc.)
- Changes to model building API (`pm.Model`, `pm.Data`, `pm.MutableData`)
- New model diagnostics or convergence checking utilities
- Any deprecations of patterns Impulso currently uses
- Fixes for known issues (e.g., LKJCholeskyCov bug)

### ArviZ
Search queries to run:
- `arviz changelog 2025 2026`
- `arviz 1.0 migration`
- `arviz new features release`

Look for:
- ArviZ 1.0 modular restructure — which subpackages replace monolithic imports?
- New plotting backends or plot types
- Changes to InferenceData structure or xarray backend
- New diagnostic functions (ESS, R-hat improvements)
- Deprecations that affect Impulso's current usage

### Pandas
Search queries to run:
- `pandas changelog 2025 2026`
- `pandas 3.0 migration guide breaking changes`
- `pandas copy-on-write`

Look for:
- Copy-on-Write (CoW) as default — does Impulso's code assume in-place mutation?
- New string dtype (pyarrow-backed) — does Impulso use string columns?
- Deprecated APIs that Impulso still uses
- New time series or DataFrame features that could simplify Impulso's code

### Pydantic
Search queries to run:
- `pydantic changelog 2025 2026`
- `pydantic v2 new features`
- `pydantic computed field validate_call`

Look for:
- New features: `@computed_field`, `@validate_call`, `TypeAdapter`, discriminated unions
- Performance improvements in model validation/construction
- Changes to serialization API (`model_dump`, `model_validate`)
- Any deprecations of patterns Impulso currently uses

### NumPy
Search queries to run:
- `numpy 2.0 migration guide breaking changes`
- `numpy changelog 2025 2026`

Look for:
- NumPy 2.0 breaking changes (type promotion, array API changes)
- Deprecated functions Impulso uses
- New array creation or manipulation utilities

## Step 3: Documentation Fetching

Use **WebFetch** to read key upstream documentation pages directly:

### PyMC
- Fetch the PyMC release notes / changelog from GitHub
- Fetch any migration guides for recent major versions

### ArviZ
- Fetch the ArviZ changelog from GitHub
- Fetch ArviZ 1.0 migration docs if available

### Pandas
- Fetch the Pandas "What's New" page for the current major version

### Pydantic
- Fetch the Pydantic v2 migration guide and changelog

Alternatively, use the **Context7** MCP tool (`resolve-library-id` then `query-docs`) to query up-to-date documentation for each library.

## Step 4: Gap Analysis

For each upstream feature identified, evaluate:

1. **Relevance**: Does this feature solve a problem Impulso users face?
2. **Effort**: How much work to adopt? (trivial API swap vs. architectural change)
3. **Risk**: Does adopting this break backward compatibility?
4. **Priority**: Assign P0–P3 based on impact and effort.

Categories of findings:
- **Deprecated patterns**: Impulso uses something upstream has deprecated -> P0/P1
- **Missing capabilities**: Upstream offers something valuable Impulso lacks -> P1/P2
- **Modernisation**: Newer pattern is cleaner/faster but current approach still works -> P2/P3
- **Version bumps**: Impulso pins an old version and should bump -> assess based on changelog
- **Bug fixes**: Upstream fixed a bug Impulso works around (e.g., LKJCholeskyCov) -> P0/P1
