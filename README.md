# Impulso

[![Release](https://img.shields.io/github/v/release/thomaspinder/impulso)](https://img.shields.io/github/v/release/thomaspinder/impulso)
[![Build status](https://img.shields.io/github/actions/workflow/status/thomaspinder/impulso/main.yml?branch=main)](https://github.com/thomaspinder/impulso/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/thomaspinder/impulso/branch/main/graph/badge.svg)](https://codecov.io/gh/thomaspinder/impulso)
[![Commit activity](https://img.shields.io/github/commit-activity/m/thomaspinder/impulso)](https://img.shields.io/github/commit-activity/m/thomaspinder/impulso)
[![License](https://img.shields.io/github/license/thomaspinder/impulso)](https://img.shields.io/github/license/thomaspinder/impulso)

Bayesian Vector Autoregression (VAR) in Python.

> 🚧 **Experimental — under heavy development.** This project is an experiment in AI-driven software development. The vast majority of the code, tests, and documentation were written by AI (Claude Code). Humans direct architecture, priorities, and design decisions, but have not reviewed most of the code line-by-line. Treat this accordingly — there will be bugs, rough edges, and things that don't work.

## Overview

**impulso** provides a modern, Pythonic interface for Bayesian Vector Autoregression modeling. Built on PyMC, it enables full posterior inference for VAR models with informative priors, structural identification, impulse response analysis, and forecast error variance decomposition.

### Core Pipeline

The library follows an immutable, type-safe pipeline:

```
VARData → VAR.fit() → FittedVAR → .set_identification_strategy() → IdentifiedVAR
```

- **`VARData`**: Validated time series data (endogenous/exogenous variables + DatetimeIndex)
- **`VAR`**: Model specification (lags, priors, exogenous variables)
- **`FittedVAR`**: Reduced-form posterior estimates with forecasting capabilities
- **`IdentifiedVAR`**: Structural VAR with impulse responses, FEVD, and historical decomposition

### Key Features

- **Full Bayesian inference** via PyMC (NUTS sampling, automatic diagnostics)
- **Minnesota priors** for regularization in high-dimensional VARs
- **Flexible identification schemes**: Recursive (Cholesky), sign restrictions
- **Forecasting**: Point forecasts, credible intervals, and scenario analysis
- **Impulse response functions** (IRFs) with uncertainty quantification
- **Forecast error variance decomposition** (FEVD)
- **Historical decomposition** of variables into structural shocks
- **Extensible protocols**: Plug in custom priors, samplers, and identification schemes
- **Type-safe**: Frozen Pydantic models with full type hints

## Installation

```bash
pip install impulso
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install impulso
```

### Faster sampling with nutpie

For significantly faster NUTS sampling, install with the optional [nutpie](https://github.com/pymc-devs/nutpie) backend:

```bash
pip install "impulso[nutpie]"
```

Or with uv:

```bash
uv add impulso --extra nutpie
```

When nutpie is installed, it is used automatically as the default sampler. You can also select the backend explicitly:

```python
from impulso.samplers import NUTSSampler

sampler = NUTSSampler(nuts_sampler="nutpie")   # or "pymc"
fitted = var.fit(data, sampler=sampler)
```

## Quick Start

```python
import pandas as pd
from impulso import VARData, VAR

# Load your time series data
df = pd.read_csv("data.csv", index_col="date", parse_dates=True)

# Create validated VAR data
data = VARData.from_df(df, endog_vars=["gdp", "inflation", "interest_rate"])

# Specify and fit a VAR(4) model with Minnesota prior
var = VAR(lags=4, prior="minnesota")
fitted = var.fit(data)

# Generate forecasts
forecast = fitted.forecast(steps=12)
forecast.plot()

# Structural identification and impulse responses
identified = fitted.set_identification_strategy("cholesky")
irf = identified.impulse_response(steps=20)
irf.plot()

# Forecast error variance decomposition
fevd = identified.fevd(steps=20)
fevd.plot()
```

## Documentation

Full documentation, tutorials, and API reference: [https://thomaspinder.github.io/impulso](https://thomaspinder.github.io/impulso)

## Development

See [CLAUDE.md](CLAUDE.md) for development setup, testing, and contribution guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use impulso in your research, please cite:

```bibtex
@software{impulso,
  author = {Pinder, Thomas},
  title = {impulso: Bayesian Vector Autoregression in Python},
  year = {2026},
  url = {https://github.com/thomaspinder/impulso}
}
```
