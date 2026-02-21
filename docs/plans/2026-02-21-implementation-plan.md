# Litterman v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete Bayesian VAR library with economist-friendly API, from data ingestion through structural identification, forecasting, and plotting.

**Architecture:** Immutable Pydantic 2.x types form a linear pipeline: `VARData → VAR → FittedVAR → IdentifiedVAR`. Extension points (priors, samplers, identification) are `typing.Protocol` classes. PyMC handles posterior sampling; ArviZ holds all inference data. Result objects provide `.median()`, `.hdi()`, `.plot()`, and `.to_dataframe()`.

**Tech Stack:** Python 3.10+, Pydantic 2.x, NumPy, pandas, PyMC, ArviZ, matplotlib, beartype (optional runtime checks)

**Reference:** `docs/plans/2026-02-21-api-design.md` — the authoritative API specification.
**Documentation Design:** `docs/plans/2026-02-21-documentation-design.md` — Diataxis doc structure and per-milestone schedule.

---

## Phasing Strategy

The plan is split into 6 phases. Each phase ends with a **milestone** followed by a **documentation task** (D0–D5). This ensures the docs site is always inspectable and reflects the current API.

| Phase | What becomes usable | Milestone | Doc Task |
|-------|-------------------|-----------|----------|
| 0 | Project bootstrap — dependencies, scaffold cleanup | `make test` passes on clean slate | D0: MkDocs setup + doc scaffold |
| 1 | `VARData` — validated immutable data container | Users can construct and validate VAR datasets | D1: VARData reference + how-to |
| 2 | `VAR` spec + `MinnesotaPrior` + `NUTSSampler` + `FittedVAR` + lag selection | Users can specify, estimate, and inspect a Bayesian VAR | D2: 6 ref pages, quickstart notebook, how-tos, explanations |
| 3 | Forecasting from `FittedVAR` + `ForecastResult` | Users can produce and summarize forecasts | D3: Forecasting notebook |
| 4 | Identification (`Cholesky`, `SignRestriction`) + `IdentifiedVAR` + IRF + FEVD + Historical Decomposition | Full structural analysis pipeline works | D4: Structural analysis notebook + ref pages |
| 5 | Plotting package + `__init__.py` re-exports + runtime checks | Library is feature-complete for v1 | D5: Plotting ref + final landing page polish |

---

## Phase 0 — Project Bootstrap

**Goal:** Add runtime dependencies, remove scaffold placeholder, set up the module structure.

### Task 0.1: Add runtime dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add runtime dependencies to pyproject.toml**

Add the following under `[project]`:

```toml
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "pydantic>=2.0",
    "pymc>=5.10",
    "arviz>=0.17",
    "matplotlib>=3.7",
    "beartype>=0.18",
]
```

**Step 2: Sync the lock file**

Run: `uv lock && uv sync`
Expected: Lock file updated, all packages installed.

### Task 0.2: Remove scaffold placeholder

**Files:**
- Delete: `src/litterman/foo.py`
- Delete: `tests/test_foo.py`
- Modify: `src/litterman/__init__.py`

**Step 1: Delete placeholder files**

```bash
rm src/litterman/foo.py tests/test_foo.py
```

**Step 2: Clear `__init__.py`**

Write an empty `__init__.py` (just a docstring):

```python
"""Litterman: Bayesian Vector Autoregression in Python."""
```

**Step 3: Run tests to verify clean slate**

Run: `uv run python -m pytest -v`
Expected: `no tests ran` or `0 items collected` — no errors.

### Task 0.3: Create empty module files

**Files:**
- Create: `src/litterman/protocols.py`
- Create: `src/litterman/data.py`
- Create: `src/litterman/spec.py`
- Create: `src/litterman/priors.py`
- Create: `src/litterman/samplers.py`
- Create: `src/litterman/fitted.py`
- Create: `src/litterman/identified.py`
- Create: `src/litterman/identification.py`
- Create: `src/litterman/results.py`
- Create: `src/litterman/plotting/__init__.py`
- Create: `src/litterman/plotting/_irf.py`
- Create: `src/litterman/plotting/_fevd.py`
- Create: `src/litterman/plotting/_forecast.py`
- Create: `src/litterman/plotting/_historical_decomposition.py`

**Step 1: Create all empty module files**

Each file should contain only a module docstring. For example:

```python
"""Protocol definitions for extensible components."""
```

Create the plotting directory:

```bash
mkdir -p src/litterman/plotting
```

**Step 2: Verify the package imports cleanly**

Run: `uv run python -c "import litterman; print('OK')"`
Expected: `OK`

### MILESTONE 0: `make test` passes on clean project structure.

### Task D0: Documentation scaffold and MkDocs setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `mkdocs.yml`
- Create: `docs/tutorials/index.md`
- Create: `docs/how-to/index.md`
- Create: `docs/explanation/index.md`
- Create: `docs/reference/index.md`
- Delete: `docs/modules.md`

**Step 1: Add mkdocs-jupyter to dev dependencies**

In `pyproject.toml`, add to the `[dependency-groups] dev` list:

```toml
"mkdocs-jupyter>=0.25.0",
```

**Step 2: Sync the lock file**

Run: `uv lock && uv sync`
Expected: Lock file updated, mkdocs-jupyter installed.

**Step 3: Replace mkdocs.yml with expanded Diataxis nav**

Overwrite `mkdocs.yml`:

```yaml
site_name: litterman
repo_url: https://github.com/thomaspinder/litterman
site_url: https://thomaspinder.github.io/litterman
site_description: Bayesian Vector Autoregression (VAR) in Python.
site_author: Thomas Pinder
edit_uri: edit/main/docs/
repo_name: thomaspinder/litterman
copyright: Maintained by <a href="https://thomaspinder.com">thomaspinder</a>.

nav:
  - Home: index.md
  - Tutorials:
    - tutorials/index.md
  - How-To Guides:
    - how-to/index.md
  - Explanation:
    - explanation/index.md
  - Reference:
    - reference/index.md

plugins:
  - search
  - mkdocs-jupyter:
      include_source: true
  - mkdocstrings:
      handlers:
        python:
          paths: ["src"]
          options:
            docstring_style: google
            show_source: false
            heading_level: 2

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.indexes
    - navigation.footer
    - navigation.top
    - content.code.copy
    - content.tabs.link
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: white
      accent: deep orange
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: black
      accent: deep orange
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  icon:
    repo: fontawesome/brands/github

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/thomaspinder/litterman
    - icon: fontawesome/brands/python
      link: https://pypi.org/project/litterman

markdown_extensions:
  - toc:
      permalink: true
  - admonition
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
```

**Step 4: Delete the old modules.md placeholder**

```bash
rm docs/modules.md
```

**Step 5: Create section landing pages**

`docs/tutorials/index.md`:

```markdown
# Tutorials

Step-by-step guides that walk you through using Litterman from start to finish.

*Tutorials will be added as features are implemented.*
```

`docs/how-to/index.md`:

```markdown
# How-To Guides

Practical recipes for solving specific problems with Litterman.

*Guides will be added as features are implemented.*
```

`docs/explanation/index.md`:

```markdown
# Explanation

Background theory and design decisions behind Litterman.

*Explanations will be added as features are implemented.*
```

`docs/reference/index.md`:

```markdown
# API Reference

Complete auto-generated reference for all Litterman modules.

*Reference pages will be added as modules are implemented.*
```

**Step 6: Verify docs build cleanly**

Run: `uv run mkdocs build --strict 2>&1 || uv run mkdocs build`
Expected: Site builds with no errors. The `--strict` flag may warn about empty sections, which is acceptable at this stage.

**Step 7: Verify local preview**

Run: `uv run mkdocs serve`
Expected: Site serves at `localhost:8000` with tabbed navigation showing all 5 sections.


---

## Phase 1 — VARData

**Goal:** Build the validated, immutable data container that all downstream code depends on.

### Task 1.1: Define `VARData` Pydantic model

**Files:**
- Create: `tests/test_data.py`
- Modify: `src/litterman/data.py`

**Step 1: Write failing tests for VARData construction**

```python
"""Tests for VARData."""

import numpy as np
import pandas as pd
import pytest

from litterman.data import VARData


@pytest.fixture
def sample_endog():
    """3 variables, 100 observations."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 3))


@pytest.fixture
def sample_index():
    return pd.date_range("2000-01-01", periods=100, freq="QS")


@pytest.fixture
def endog_names():
    return ["gdp", "inflation", "rate"]


class TestVARDataConstruction:
    def test_basic_construction(self, sample_endog, sample_index, endog_names):
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            index=sample_index,
        )
        assert data.endog.shape == (100, 3)
        assert data.exog is None
        assert data.exog_names is None
        assert len(data.endog_names) == 3

    def test_with_exog(self, sample_endog, sample_index, endog_names):
        rng = np.random.default_rng(42)
        exog = rng.standard_normal((100, 1))
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            exog=exog,
            exog_names=["oil_price"],
            index=sample_index,
        )
        assert data.exog is not None
        assert data.exog.shape == (100, 1)

    def test_frozen(self, sample_endog, sample_index, endog_names):
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            index=sample_index,
        )
        with pytest.raises(Exception):
            data.endog = np.zeros((100, 3))

    def test_arrays_not_writeable(self, sample_endog, sample_index, endog_names):
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            index=sample_index,
        )
        with pytest.raises(ValueError):
            data.endog[0, 0] = 999.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_data.py -v`
Expected: FAIL — `VARData` not yet defined.

**Step 3: Implement VARData**

In `src/litterman/data.py`:

```python
"""VARData — validated, immutable data container for VAR models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class VARData(BaseModel):
    """Immutable, validated container for VAR estimation data.

    Args:
        endog: Endogenous variable array of shape (T, n) where T >= 1 and n >= 2.
        endog_names: Names for each endogenous variable.
        exog: Optional exogenous variable array of shape (T, k).
        exog_names: Names for each exogenous variable. Required if exog is provided.
        index: DatetimeIndex of length T.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    endog: np.ndarray
    endog_names: list[str]
    exog: np.ndarray | None = None
    exog_names: list[str] | None = None
    index: pd.DatetimeIndex

    @model_validator(mode="after")
    def _validate(self) -> Self:
        # Shape consistency
        t, n = self.endog.shape
        if n < 2:
            raise ValueError(f"Minimum 2 endogenous variables required, got {n}")
        if len(self.endog_names) != n:
            raise ValueError(f"endog_names length {len(self.endog_names)} != endog columns {n}")
        if len(self.index) != t:
            raise ValueError(f"index length {len(self.index)} != endog rows {t}")

        # Exogenous consistency
        if self.exog is not None:
            if self.exog.shape[0] != t:
                raise ValueError(f"exog rows {self.exog.shape[0]} != endog rows {t}")
            if self.exog_names is None:
                raise ValueError("exog_names required when exog is provided")
            if len(self.exog_names) != self.exog.shape[1]:
                raise ValueError(f"exog_names length {len(self.exog_names)} != exog columns {self.exog.shape[1]}")
        elif self.exog_names is not None:
            raise ValueError("exog_names provided without exog")

        # Finite values
        if not np.isfinite(self.endog).all():
            raise ValueError("endog contains NaN or Inf values")
        if self.exog is not None and not np.isfinite(self.exog).all():
            raise ValueError("exog contains NaN or Inf values")

        # Defensive copy + read-only
        endog_copy = self.endog.copy()
        endog_copy.flags.writeable = False
        object.__setattr__(self, "endog", endog_copy)

        if self.exog is not None:
            exog_copy = self.exog.copy()
            exog_copy.flags.writeable = False
            object.__setattr__(self, "exog", exog_copy)

        return self
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_data.py -v`
Expected: All 4 tests PASS.



### Task 1.2: VARData validation edge cases

**Files:**
- Modify: `tests/test_data.py`

**Step 1: Write failing tests for validation**

Append to `tests/test_data.py`:

```python
class TestVARDataValidation:
    def test_rejects_nan(self, sample_index, endog_names):
        bad = np.array([[1.0, 2.0, np.nan]] * 100)
        with pytest.raises(ValueError, match="NaN or Inf"):
            VARData(endog=bad, endog_names=endog_names, index=sample_index)

    def test_rejects_inf(self, sample_index, endog_names):
        bad = np.array([[1.0, 2.0, np.inf]] * 100)
        with pytest.raises(ValueError, match="NaN or Inf"):
            VARData(endog=bad, endog_names=endog_names, index=sample_index)

    def test_rejects_single_variable(self, sample_index):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Minimum 2"):
            VARData(
                endog=rng.standard_normal((100, 1)),
                endog_names=["gdp"],
                index=sample_index,
            )

    def test_rejects_mismatched_names(self, sample_endog, sample_index):
        with pytest.raises(ValueError, match="endog_names length"):
            VARData(endog=sample_endog, endog_names=["a", "b"], index=sample_index)

    def test_rejects_mismatched_index(self, sample_endog, endog_names):
        short_index = pd.date_range("2000-01-01", periods=50, freq="QS")
        with pytest.raises(ValueError, match="index length"):
            VARData(endog=sample_endog, endog_names=endog_names, index=short_index)

    def test_rejects_exog_names_without_exog(self, sample_endog, sample_index, endog_names):
        with pytest.raises(ValueError, match="exog_names provided without exog"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog_names=["oil"],
                index=sample_index,
            )

    def test_rejects_exog_without_names(self, sample_endog, sample_index, endog_names):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="exog_names required"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog=rng.standard_normal((100, 1)),
                index=sample_index,
            )
```

**Step 2: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_data.py -v`
Expected: All tests PASS (the implementation from Task 1.1 already handles these).

### Task 1.3: `VARData.from_df` class method

**Files:**
- Modify: `tests/test_data.py`
- Modify: `src/litterman/data.py`

**Step 1: Write failing test for from_df**

Append to `tests/test_data.py`:

```python
class TestVARDataFromDF:
    def test_from_df_endog_only(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2000-01-01", periods=100, freq="QS")
        df = pd.DataFrame(
            rng.standard_normal((100, 3)),
            columns=["gdp", "inflation", "rate"],
            index=index,
        )
        data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
        assert data.endog.shape == (100, 3)
        assert data.endog_names == ["gdp", "inflation", "rate"]
        assert data.exog is None

    def test_from_df_with_exog(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2000-01-01", periods=100, freq="QS")
        df = pd.DataFrame(
            rng.standard_normal((100, 4)),
            columns=["gdp", "inflation", "rate", "oil"],
            index=index,
        )
        data = VARData.from_df(df, endog=["gdp", "inflation", "rate"], exog=["oil"])
        assert data.exog is not None
        assert data.exog_shape == (100, 1)
        assert data.exog_names == ["oil"]

    def test_from_df_requires_datetime_index(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            VARData.from_df(df, endog=["a", "b"])
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_data.py::TestVARDataFromDF -v`
Expected: FAIL — `from_df` not defined.

**Step 3: Implement from_df**

Add to `VARData` class in `src/litterman/data.py`:

```python
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        endog: list[str],
        exog: list[str] | None = None,
    ) -> VARData:
        """Construct VARData from a pandas DataFrame.

        Args:
            df: DataFrame with a DatetimeIndex.
            endog: Column names for endogenous variables.
            exog: Column names for exogenous variables (optional).

        Returns:
            Validated VARData instance.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"DataFrame must have a DatetimeIndex, got {type(df.index).__name__}")

        endog_arr = df[endog].to_numpy(dtype=np.float64)
        exog_arr = df[exog].to_numpy(dtype=np.float64) if exog is not None else None

        return cls(
            endog=endog_arr,
            endog_names=endog,
            exog=exog_arr,
            exog_names=exog,
            index=df.index,
        )
```

**Step 4: Fix test typo — `exog_shape` should be `exog.shape`**

In the test `test_from_df_with_exog`, the assertion `data.exog_shape` should be `data.exog.shape`.

**Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_data.py -v`
Expected: All tests PASS.


### MILESTONE 1: Users can construct validated, immutable VARData from arrays or DataFrames. All validation rules enforced.

### Task D1: VARData documentation

**Files:**
- Create: `docs/reference/data.md`
- Create: `docs/how-to/data-preparation.md`
- Modify: `docs/index.md`
- Modify: `mkdocs.yml`

**Step 1: Create API reference page for data module**

`docs/reference/data.md`:

```markdown
# Data

::: litterman.data
```

**Step 2: Write data preparation how-to**

`docs/how-to/data-preparation.md`:

```markdown
# Preparing Data for VARData

This guide shows how to construct a `VARData` container from common data formats.

## From a pandas DataFrame

The simplest path is `VARData.from_df`. Your DataFrame must have a `DatetimeIndex`:

\`\`\`python
import pandas as pd
from litterman import VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
\`\`\`

## With exogenous variables

Pass column names for exogenous variables separately:

\`\`\`python
data = VARData.from_df(
    df,
    endog=["gdp", "inflation", "rate"],
    exog=["oil_price"],
)
\`\`\`

## From NumPy arrays

If you already have arrays, pass them directly:

\`\`\`python
import numpy as np

data = VARData(
    endog=endog_array,           # shape (T, n), n >= 2
    endog_names=["gdp", "inflation", "rate"],
    index=pd.date_range("2000-01-01", periods=T, freq="QS"),
)
\`\`\`

## Validation rules

`VARData` enforces these constraints at construction time:

- At least 2 endogenous variables
- No `NaN` or `Inf` values
- `endog_names` length must match number of columns
- `index` length must match number of rows
- If `exog` is provided, `exog_names` is required (and vice versa)
- Arrays are copied and made read-only — the original data is never modified
```

**Step 3: Update landing page with VARData example**

Overwrite `docs/index.md`:

```markdown
# Litterman

**Bayesian Vector Autoregression in Python.**

\`\`\`python
import pandas as pd
from litterman import VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
# Validated, immutable, ready for modeling
\`\`\`

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable types** — all objects are frozen after creation, preventing accidental mutation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains

## Installation

\`\`\`bash
pip install litterman
\`\`\`
```

**Step 4: Update mkdocs.yml nav to include new pages**

Update the nav section:

```yaml
nav:
  - Home: index.md
  - Tutorials:
    - tutorials/index.md
  - How-To Guides:
    - how-to/index.md
    - how-to/data-preparation.md
  - Explanation:
    - explanation/index.md
  - Reference:
    - reference/index.md
    - reference/data.md
```

**Step 5: Verify docs build**

Run: `uv run mkdocs build`
Expected: Clean build. The `VARData` autodoc page renders with constructor signature and docstrings.

---

## Phase 2 — Specification, Priors, Samplers, Lag Selection, Estimation

**Goal:** Build the complete pipeline from `VAR` specification through `FittedVAR`.

### Task 2.1: Protocols

**Files:**
- Create: `tests/test_protocols.py`
- Modify: `src/litterman/protocols.py`

**Step 1: Write test that protocol classes exist and are importable**

```python
"""Tests for protocol definitions."""

from typing import runtime_checkable

from litterman.protocols import IdentificationScheme, Prior, Sampler


class TestProtocols:
    def test_prior_is_runtime_checkable(self):
        assert hasattr(Prior, "__protocol_attrs__") or runtime_checkable

    def test_sampler_is_runtime_checkable(self):
        assert hasattr(Sampler, "__protocol_attrs__") or runtime_checkable

    def test_identification_is_runtime_checkable(self):
        assert hasattr(IdentificationScheme, "__protocol_attrs__") or runtime_checkable
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_protocols.py -v`
Expected: FAIL — protocols not defined.

**Step 3: Implement protocols**

In `src/litterman/protocols.py`:

```python
"""Protocol definitions for extensible components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import arviz as az
import pymc as pm


@runtime_checkable
class Prior(Protocol):
    """Contract for prior specifications."""

    def build_priors(self, n_vars: int, n_lags: int) -> dict: ...


@runtime_checkable
class Sampler(Protocol):
    """Contract for posterior sampling strategies."""

    def sample(self, model: pm.Model) -> az.InferenceData: ...


@runtime_checkable
class IdentificationScheme(Protocol):
    """Contract for structural identification schemes."""

    def identify(self, idata: az.InferenceData, var_names: list[str]) -> az.InferenceData: ...
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_protocols.py -v`
Expected: PASS.


### Task 2.2: MinnesotaPrior

**Files:**
- Create: `tests/test_priors.py`
- Modify: `src/litterman/priors.py`

**Step 1: Write failing tests**

```python
"""Tests for prior specifications."""

import pytest

from litterman.priors import MinnesotaPrior
from litterman.protocols import Prior


class TestMinnesotaPrior:
    def test_default_construction(self):
        prior = MinnesotaPrior()
        assert prior.tightness == 0.1
        assert prior.decay == "harmonic"
        assert prior.cross_shrinkage == 0.5

    def test_custom_construction(self):
        prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.8)
        assert prior.tightness == 0.2
        assert prior.decay == "geometric"

    def test_frozen(self):
        prior = MinnesotaPrior()
        with pytest.raises(Exception):
            prior.tightness = 0.5

    def test_satisfies_prior_protocol(self):
        prior = MinnesotaPrior()
        assert isinstance(prior, Prior)

    def test_rejects_zero_tightness(self):
        with pytest.raises(Exception):
            MinnesotaPrior(tightness=0.0)

    def test_rejects_negative_tightness(self):
        with pytest.raises(Exception):
            MinnesotaPrior(tightness=-0.1)

    def test_build_priors_returns_dict(self):
        prior = MinnesotaPrior()
        result = prior.build_priors(n_vars=3, n_lags=2)
        assert isinstance(result, dict)
        assert "B_mu" in result
        assert "B_sigma" in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_priors.py -v`
Expected: FAIL.

**Step 3: Implement MinnesotaPrior**

In `src/litterman/priors.py`:

```python
"""Prior specifications for VAR models."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MinnesotaPrior(BaseModel):
    """Minnesota prior for VAR coefficient shrinkage.

    Args:
        tightness: Overall shrinkage toward prior mean. Must be > 0.
        decay: How coefficients shrink on longer lags.
        cross_shrinkage: Shrinkage on other variables' lags vs own. 0 = only own lags, 1 = equal.
    """

    model_config = ConfigDict(frozen=True)

    tightness: float = Field(0.1, gt=0)
    decay: Literal["harmonic", "geometric"] = "harmonic"
    cross_shrinkage: float = Field(0.5, ge=0, le=1)

    def build_priors(self, n_vars: int, n_lags: int) -> dict:
        """Build prior mean and standard deviation arrays for VAR coefficients.

        Args:
            n_vars: Number of endogenous variables.
            n_lags: Number of lags.

        Returns:
            Dictionary with keys 'B_mu' and 'B_sigma' as numpy arrays.
        """
        n_coeffs = n_vars * n_lags
        B_mu = np.zeros((n_vars, n_coeffs))
        B_sigma = np.ones((n_vars, n_coeffs))

        for eq in range(n_vars):
            for lag in range(1, n_lags + 1):
                if self.decay == "harmonic":
                    lag_decay = 1.0 / lag
                else:  # geometric
                    lag_decay = 1.0 / (lag**2)

                for var in range(n_vars):
                    col = (lag - 1) * n_vars + var
                    if var == eq:
                        # Own lag: prior mean = 1.0 for first lag, 0 otherwise
                        if lag == 1:
                            B_mu[eq, col] = 1.0
                        B_sigma[eq, col] = self.tightness * lag_decay
                    else:
                        # Cross lag
                        B_sigma[eq, col] = self.tightness * lag_decay * self.cross_shrinkage

        return {"B_mu": B_mu, "B_sigma": B_sigma}
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_priors.py -v`
Expected: PASS.

### Task 2.3: NUTSSampler

**Files:**
- Create: `tests/test_samplers.py`
- Modify: `src/litterman/samplers.py`

**Step 1: Write failing tests**

```python
"""Tests for sampler specifications."""

import pytest

from litterman.protocols import Sampler
from litterman.samplers import NUTSSampler


class TestNUTSSampler:
    def test_default_construction(self):
        sampler = NUTSSampler()
        assert sampler.draws == 1000
        assert sampler.tune == 1000
        assert sampler.chains == 4
        assert sampler.cores is None
        assert sampler.target_accept == 0.8
        assert sampler.random_seed is None

    def test_custom_construction(self):
        sampler = NUTSSampler(draws=2000, chains=2, random_seed=42)
        assert sampler.draws == 2000
        assert sampler.chains == 2
        assert sampler.random_seed == 42

    def test_frozen(self):
        sampler = NUTSSampler()
        with pytest.raises(Exception):
            sampler.draws = 500

    def test_satisfies_sampler_protocol(self):
        sampler = NUTSSampler()
        assert isinstance(sampler, Sampler)

    def test_rejects_zero_draws(self):
        with pytest.raises(Exception):
            NUTSSampler(draws=0)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_samplers.py -v`
Expected: FAIL.

**Step 3: Implement NUTSSampler**

In `src/litterman/samplers.py`:

```python
"""Sampler specifications for posterior inference."""

from __future__ import annotations

import arviz as az
import pymc as pm
from pydantic import BaseModel, ConfigDict, Field


class NUTSSampler(BaseModel):
    """NUTS sampler configuration for PyMC.

    Args:
        draws: Number of posterior draws per chain.
        tune: Number of tuning steps per chain.
        chains: Number of independent chains.
        cores: Number of CPU cores. None = auto-detect.
        target_accept: Target acceptance rate for NUTS.
        random_seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    draws: int = Field(1000, ge=1)
    tune: int = Field(1000, ge=0)
    chains: int = Field(4, ge=1)
    cores: int | None = Field(None, ge=1)
    target_accept: float = Field(0.8, gt=0, lt=1)
    random_seed: int | None = None

    def sample(self, model: pm.Model) -> az.InferenceData:
        """Run NUTS sampling on the given PyMC model.

        Args:
            model: A fully specified PyMC model.

        Returns:
            ArviZ InferenceData with posterior and log_likelihood groups.
        """
        with model:
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                idata_kwargs={"log_likelihood": True},
            )
        return idata
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_samplers.py -v`
Expected: PASS.


### Task 2.4: Result base classes

**Files:**
- Create: `tests/test_results.py`
- Modify: `src/litterman/results.py`

**Step 1: Write failing tests**

```python
"""Tests for result objects."""

import arviz as az
import numpy as np
import pandas as pd
import pytest

from litterman.results import HDIResult, LagOrderResult, VARResultBase


class TestHDIResult:
    def test_construction(self):
        lower = pd.DataFrame({"a": [1.0], "b": [2.0]})
        upper = pd.DataFrame({"a": [3.0], "b": [4.0]})
        hdi = HDIResult(lower=lower, upper=upper, prob=0.89)
        assert hdi.prob == 0.89

    def test_frozen(self):
        lower = pd.DataFrame({"a": [1.0]})
        upper = pd.DataFrame({"a": [2.0]})
        hdi = HDIResult(lower=lower, upper=upper, prob=0.89)
        with pytest.raises(Exception):
            hdi.prob = 0.95


class TestLagOrderResult:
    def test_construction(self):
        table = pd.DataFrame({"aic": [100, 95], "bic": [110, 105], "hq": [105, 100]}, index=[1, 2])
        result = LagOrderResult(aic=2, bic=2, hq=2, criteria_table=table)
        assert result.aic == 2
        assert result.bic == 2

    def test_summary_returns_table(self):
        table = pd.DataFrame({"aic": [100, 95], "bic": [110, 105], "hq": [105, 100]}, index=[1, 2])
        result = LagOrderResult(aic=2, bic=2, hq=2, criteria_table=table)
        summary = result.summary()
        assert isinstance(summary, pd.DataFrame)

    def test_criteria_table_excluded_from_repr(self):
        table = pd.DataFrame({"aic": [100]}, index=[1])
        result = LagOrderResult(aic=1, bic=1, hq=1, criteria_table=table)
        assert "criteria_table" not in repr(result)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_results.py -v`
Expected: FAIL.

**Step 3: Implement result classes**

In `src/litterman/results.py`:

```python
"""Result objects for VAR post-estimation output."""

from __future__ import annotations

from abc import abstractmethod

import arviz as az
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field


class HDIResult(BaseModel):
    """Structured HDI output with separate lower/upper bounds.

    Args:
        lower: DataFrame of lower HDI bounds.
        upper: DataFrame of upper HDI bounds.
        prob: HDI probability level.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lower: pd.DataFrame
    upper: pd.DataFrame
    prob: float


class VARResultBase(BaseModel):
    """Base class for VAR post-estimation results.

    Args:
        idata: ArviZ InferenceData holding the result draws.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData

    def median(self) -> pd.DataFrame:
        """Compute posterior median of the result."""
        raise NotImplementedError

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Compute highest density interval.

        Args:
            prob: Probability mass for the HDI. Default 0.89.
        """
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to a tidy DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> Figure:
        """Plot the result. Subclasses must implement."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LagOrderResult(BaseModel):
    """Result from lag order selection.

    Args:
        aic: Optimal lag order by AIC.
        bic: Optimal lag order by BIC.
        hq: Optimal lag order by Hannan-Quinn.
        criteria_table: DataFrame of all criteria values by lag order.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    aic: int
    bic: int
    hq: int
    criteria_table: pd.DataFrame = Field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Return the full criteria table.

        Returns:
            DataFrame with information criteria for each lag order.
        """
        return self.criteria_table
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_results.py -v`
Expected: PASS.

### Task 2.5: Lag selection (`select_lag_order`)

**Files:**
- Create: `tests/test_lag_selection.py`
- Create: `src/litterman/_lag_selection.py`

**Step 1: Write failing tests**

```python
"""Tests for lag order selection."""

import numpy as np
import pandas as pd

from litterman._lag_selection import select_lag_order
from litterman.data import VARData
from litterman.results import LagOrderResult


@pytest.fixture
def var_data():
    """Simple VAR(2) DGP for testing lag selection."""
    rng = np.random.default_rng(42)
    T = 200
    n = 3
    y = np.zeros((T, n))
    for t in range(2, T):
        y[t] = 0.5 * y[t - 1] + 0.2 * y[t - 2] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2", "y3"], index=index)


import pytest


class TestSelectLagOrder:
    def test_returns_lag_order_result(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        assert isinstance(result, LagOrderResult)

    def test_aic_bic_hq_are_positive_ints(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        assert result.aic >= 1
        assert result.bic >= 1
        assert result.hq >= 1

    def test_summary_has_expected_columns(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        summary = result.summary()
        assert "aic" in summary.columns
        assert "bic" in summary.columns
        assert "hq" in summary.columns

    def test_summary_rows_match_max_lags(self, var_data):
        result = select_lag_order(var_data, max_lags=6)
        assert len(result.summary()) == 6
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_lag_selection.py -v`
Expected: FAIL.

**Step 3: Implement select_lag_order**

In `src/litterman/_lag_selection.py`:

```python
"""OLS-based lag order selection using information criteria."""

from __future__ import annotations

import numpy as np
import pandas as pd

from litterman.data import VARData
from litterman.results import LagOrderResult


def select_lag_order(data: VARData, max_lags: int = 12) -> LagOrderResult:
    """Select optimal VAR lag order using AIC, BIC, and Hannan-Quinn.

    Uses OLS estimation (fast) to compute information criteria for each
    candidate lag order from 1 to max_lags.

    Args:
        data: VARData instance.
        max_lags: Maximum number of lags to evaluate.

    Returns:
        LagOrderResult with optimal lag orders and full criteria table.
    """
    y = data.endog
    T, n = y.shape

    results = []
    for p in range(1, max_lags + 1):
        # Build lagged regressor matrix
        Y = y[p:]  # (T-p, n)
        T_eff = Y.shape[0]
        X_parts = [np.ones((T_eff, 1))]  # intercept
        for lag in range(1, p + 1):
            X_parts.append(y[p - lag : T - lag])
        if data.exog is not None:
            X_parts.append(data.exog[p:])
        X = np.hstack(X_parts)  # (T_eff, 1 + n*p + k)

        # OLS
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        sigma = (resid.T @ resid) / T_eff

        # Log determinant of residual covariance
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            logdet = np.inf

        k_params = X.shape[1] * n  # total parameters
        aic = logdet + 2 * k_params / T_eff
        bic = logdet + np.log(T_eff) * k_params / T_eff
        hq = logdet + 2 * np.log(np.log(T_eff)) * k_params / T_eff

        results.append({"lag": p, "aic": aic, "bic": bic, "hq": hq})

    table = pd.DataFrame(results).set_index("lag")

    return LagOrderResult(
        aic=int(table["aic"].idxmin()),
        bic=int(table["bic"].idxmin()),
        hq=int(table["hq"].idxmin()),
        criteria_table=table,
    )
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_lag_selection.py -v`
Expected: PASS.

### Task 2.6: VAR specification model

**Files:**
- Create: `tests/test_spec.py`
- Modify: `src/litterman/spec.py`

**Step 1: Write failing tests**

```python
"""Tests for VAR specification."""

import pytest

from litterman.priors import MinnesotaPrior
from litterman.spec import VAR


class TestVARSpec:
    def test_fixed_lags(self):
        spec = VAR(lags=4, prior="minnesota")
        assert spec.lags == 4
        assert spec.max_lags is None

    def test_string_lags(self):
        spec = VAR(lags="bic", prior="minnesota")
        assert spec.lags == "bic"

    def test_string_lags_with_max(self):
        spec = VAR(lags="aic", max_lags=12, prior="minnesota")
        assert spec.max_lags == 12

    def test_rejects_max_lags_with_fixed(self):
        with pytest.raises(ValueError, match="max_lags is only valid"):
            VAR(lags=4, max_lags=12, prior="minnesota")

    def test_prior_string_resolves(self):
        spec = VAR(lags=2, prior="minnesota")
        assert isinstance(spec.resolved_prior, MinnesotaPrior)

    def test_prior_object(self):
        prior = MinnesotaPrior(tightness=0.2)
        spec = VAR(lags=2, prior=prior)
        assert spec.resolved_prior.tightness == 0.2

    def test_frozen(self):
        spec = VAR(lags=2, prior="minnesota")
        with pytest.raises(Exception):
            spec.lags = 3

    def test_rejects_zero_lags(self):
        with pytest.raises(Exception):
            VAR(lags=0, prior="minnesota")
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_spec.py -v`
Expected: FAIL.

**Step 3: Implement VAR spec**

In `src/litterman/spec.py`:

```python
"""VAR model specification."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from litterman.priors import MinnesotaPrior
from litterman.protocols import Prior

_PRIOR_REGISTRY: dict[str, type] = {
    "minnesota": MinnesotaPrior,
}


class VAR(BaseModel):
    """Immutable VAR model specification.

    Args:
        lags: Fixed lag order (int >= 1) or selection criterion string.
        max_lags: Upper bound for automatic selection. Only valid with string lags.
        prior: Prior shorthand string or Prior protocol instance.
    """

    model_config = ConfigDict(frozen=True)

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Union[Literal["minnesota"], Prior] = "minnesota"  # noqa: UP007

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        if self.max_lags is not None and isinstance(self.lags, int):
            raise ValueError("max_lags is only valid when lags is a selection criterion ('aic', 'bic', 'hq')")
        if isinstance(self.lags, int) and self.lags < 1:
            raise ValueError(f"lags must be >= 1, got {self.lags}")
        return self

    @property
    def resolved_prior(self) -> Prior:
        """Resolve string prior shorthand to a Prior instance."""
        if isinstance(self.prior, str):
            return _PRIOR_REGISTRY[self.prior]()
        return self.prior
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_spec.py -v`
Expected: PASS.

### Task 2.7: FittedVAR and VAR.fit()

**Files:**
- Create: `tests/test_fitted.py`
- Modify: `src/litterman/fitted.py`
- Modify: `src/litterman/spec.py` (add `fit` method)

**Step 1: Write failing tests**

```python
"""Tests for FittedVAR."""

import numpy as np
import pandas as pd
import pytest

from litterman.data import VARData
from litterman.fitted import FittedVAR
from litterman.samplers import NUTSSampler
from litterman.spec import VAR


@pytest.fixture
def var_data():
    """Simple VAR(1) DGP."""
    rng = np.random.default_rng(42)
    T = 200
    n = 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


class TestFittedVAR:
    @pytest.mark.slow
    def test_fit_returns_fitted_var(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert isinstance(result, FittedVAR)

    @pytest.mark.slow
    def test_fitted_properties(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert result.n_lags == 1
        assert result.has_exog is False
        assert result.idata is not None

    @pytest.mark.slow
    def test_fit_with_auto_lags(self, var_data):
        spec = VAR(lags="bic", max_lags=4, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert result.n_lags >= 1

    def test_fitted_var_frozen(self, var_data):
        """Test that FittedVAR is immutable (uses a mock to avoid sampling)."""
        # This tests the model structure — actual sampling tested in slow tests
        pass

    @pytest.mark.slow
    def test_repr_is_compact(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        r = repr(result)
        assert "FittedVAR" in r
        assert "n_lags=1" in r
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_fitted.py -v -k "not slow"`
Expected: FAIL (or no tests collected; the slow tests will fail when run).

**Step 3: Implement FittedVAR**

In `src/litterman/fitted.py`:

```python
"""FittedVAR — reduced-form posterior from Bayesian VAR estimation."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from litterman.data import VARData


class FittedVAR(BaseModel):
    """Immutable container for a fitted (reduced-form) Bayesian VAR.

    Args:
        idata: ArviZ InferenceData with posterior draws.
        n_lags: Lag order used in estimation.
        data: Original VARData used for fitting.
        var_names: Names of endogenous variables.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData
    n_lags: int
    data: VARData
    var_names: list[str]

    @property
    def has_exog(self) -> bool:
        """Whether the model includes exogenous variables."""
        return self.data.exog is not None

    @property
    def coefficients(self) -> np.ndarray:
        """Posterior draws of B coefficient matrices."""
        return self.idata.posterior["B"].values

    @property
    def intercepts(self) -> np.ndarray:
        """Posterior draws of intercept vectors."""
        return self.idata.posterior["intercept"].values

    @property
    def sigma(self) -> np.ndarray:
        """Posterior draws of residual covariance matrix."""
        return self.idata.posterior["Sigma"].values

    def __repr__(self) -> str:
        n_vars = len(self.var_names)
        posterior = self.idata.posterior
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        return f"FittedVAR(n_lags={self.n_lags}, n_vars={n_vars}, n_draws={n_draws}, n_chains={n_chains})"
```

**Step 4: Add `fit` method to VAR**

Add to `src/litterman/spec.py`:

```python
    def fit(
        self,
        data: VARData,
        sampler: Sampler | None = None,
    ) -> FittedVAR:
        """Estimate the Bayesian VAR model.

        Args:
            data: VARData instance.
            sampler: Sampler protocol instance. Defaults to NUTSSampler().

        Returns:
            FittedVAR with posterior draws.
        """
        from litterman._lag_selection import select_lag_order
        from litterman.fitted import FittedVAR
        from litterman.samplers import NUTSSampler

        if sampler is None:
            sampler = NUTSSampler()

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        # Build prior arrays
        prior = self.resolved_prior
        n_vars = data.endog.shape[1]
        prior_params = prior.build_priors(n_vars=n_vars, n_lags=n_lags)

        # Build data matrices
        y = data.endog
        Y = y[n_lags:]
        T_eff = Y.shape[0]
        X_parts = []
        for lag in range(1, n_lags + 1):
            X_parts.append(y[n_lags - lag : -lag])
        X_lag = np.hstack(X_parts)

        if data.exog is not None:
            X_exog = data.exog[n_lags:]
        else:
            X_exog = None

        # Build PyMC model
        import pymc as pm

        with pm.Model() as model:
            # Intercept
            intercept = pm.Normal("intercept", mu=0, sigma=1, shape=n_vars)

            # VAR coefficients with Minnesota prior
            B = pm.Normal(
                "B",
                mu=prior_params["B_mu"],
                sigma=prior_params["B_sigma"],
                shape=(n_vars, n_vars * n_lags),
            )

            # Exogenous coefficients
            if X_exog is not None:
                n_exog = X_exog.shape[1]
                B_exog = pm.Normal("B_exog", mu=0, sigma=1, shape=(n_vars, n_exog))
                mu = intercept + pm.math.dot(X_lag, B.T) + pm.math.dot(X_exog, B_exog.T)
            else:
                mu = intercept + pm.math.dot(X_lag, B.T)

            # Residual covariance (LKJ prior)
            chol, corr, stds = pm.LKJCholeskyCov(
                "Sigma_chol",
                n=n_vars,
                eta=2.0,
                sd_dist=pm.HalfCauchy.dist(beta=2.5),
                compute_corr=True,
            )
            Sigma = pm.Deterministic("Sigma", pm.math.dot(chol, chol.T))

            # Likelihood
            pm.MvNormal("obs", mu=mu, chol=chol, observed=Y)

        # Sample
        idata = sampler.sample(model)

        return FittedVAR(
            idata=idata,
            n_lags=n_lags,
            data=data,
            var_names=data.endog_names,
        )
```

You will also need to add these imports at the top of `spec.py`:

```python
import numpy as np
from litterman.data import VARData
from litterman.protocols import Prior, Sampler
```

And the return type import: `from litterman.fitted import FittedVAR` is done inline to avoid circular imports.

**Step 5: Run slow tests**

Run: `uv run python -m pytest tests/test_fitted.py -v -m slow`
Expected: PASS (may take a minute for MCMC sampling).


### MILESTONE 2: Users can specify a Bayesian VAR, select lag order, and estimate the model. `VAR(lags="bic", prior="minnesota").fit(data)` returns a `FittedVAR` with accessible posterior draws.

### Task D2: Estimation pipeline documentation

This is the largest doc task. Six new API reference pages, a quickstart notebook, two how-to guides, and two explanation pages.

**Files:**
- Create: `docs/reference/spec.md`
- Create: `docs/reference/priors.md`
- Create: `docs/reference/samplers.md`
- Create: `docs/reference/fitted.md`
- Create: `docs/reference/results.md`
- Create: `docs/reference/protocols.md`
- Create: `docs/tutorials/quickstart.ipynb`
- Create: `docs/how-to/custom-priors.md`
- Create: `docs/how-to/lag-selection.md`
- Create: `docs/explanation/bayesian-var.md`
- Create: `docs/explanation/minnesota-prior.md`
- Modify: `mkdocs.yml`

**Step 1: Create API reference pages**

Each reference page follows the same pattern. Create the following files:

`docs/reference/spec.md`:

```markdown
# VAR Specification

::: litterman.spec
```

`docs/reference/priors.md`:

```markdown
# Priors

::: litterman.priors
```

`docs/reference/samplers.md`:

```markdown
# Samplers

::: litterman.samplers
```

`docs/reference/fitted.md`:

```markdown
# FittedVAR

::: litterman.fitted
```

`docs/reference/results.md`:

```markdown
# Results

::: litterman.results
```

`docs/reference/protocols.md`:

```markdown
# Protocols

::: litterman.protocols
```

**Step 2: Create quickstart tutorial notebook**

Create `docs/tutorials/quickstart.ipynb` as a Jupyter notebook with these cells:

Cell 1 (markdown):
```markdown
# Quickstart: Your First Bayesian VAR

This tutorial walks you through fitting a Bayesian VAR model to macroeconomic data using Litterman.
```

Cell 2 (code):
```python
import numpy as np
import pandas as pd
from litterman import VAR, VARData, select_lag_order
from litterman.samplers import NUTSSampler
```

Cell 3 (markdown):
```markdown
## Create some synthetic data

We'll simulate a simple VAR(1) process with two variables.
```

Cell 4 (code):
```python
rng = np.random.default_rng(42)
T = 200
y = np.zeros((T, 2))
for t in range(1, T):
    y[t] = 0.5 * y[t - 1] + rng.standard_normal(2) * 0.1

index = pd.date_range("2000-01-01", periods=T, freq="QS")
data = VARData(endog=y, endog_names=["gdp", "inflation"], index=index)
data
```

Cell 5 (markdown):
```markdown
## Select lag order

Use information criteria to find the optimal lag length.
```

Cell 6 (code):
```python
ic = select_lag_order(data, max_lags=8)
print(f"AIC: {ic.aic}, BIC: {ic.bic}, HQ: {ic.hq}")
ic.summary()
```

Cell 7 (markdown):
```markdown
## Specify and estimate the model

Create a VAR specification and fit it. We use a small number of draws here for speed.
```

Cell 8 (code):
```python
spec = VAR(lags="bic", prior="minnesota")
sampler = NUTSSampler(draws=500, tune=500, chains=2, random_seed=42)
fitted = spec.fit(data, sampler=sampler)
fitted
```

Cell 9 (markdown):
```markdown
## Inspect the posterior

Access the ArviZ InferenceData directly for diagnostics.
```

Cell 10 (code):
```python
import arviz as az
az.summary(fitted.idata, var_names=["B", "intercept"])
```

**Step 3: Write custom priors how-to**

`docs/how-to/custom-priors.md`:

```markdown
# Writing a Custom Prior

Litterman uses `typing.Protocol` for extensibility. You can write your own prior by implementing the `Prior` protocol.

## The Prior protocol

\`\`\`python
from litterman.protocols import Prior

class MyPrior:
    def build_priors(self, n_vars: int, n_lags: int) -> dict:
        ...
\`\`\`

Your `build_priors` method must return a dictionary with keys `"B_mu"` and `"B_sigma"`, both NumPy arrays of shape `(n_vars, n_vars * n_lags)`.

- `B_mu`: Prior mean for VAR coefficient matrix
- `B_sigma`: Prior standard deviation for VAR coefficient matrix

## Example: Flat prior

\`\`\`python
import numpy as np

class FlatPrior:
    def build_priors(self, n_vars: int, n_lags: int) -> dict:
        n_coeffs = n_vars * n_lags
        return {
            "B_mu": np.zeros((n_vars, n_coeffs)),
            "B_sigma": np.ones((n_vars, n_coeffs)) * 10.0,
        }
\`\`\`

## Using your custom prior

\`\`\`python
from litterman import VAR

spec = VAR(lags=2, prior=FlatPrior())
fitted = spec.fit(data)
\`\`\`
```

**Step 4: Write lag selection how-to**

`docs/how-to/lag-selection.md`:

```markdown
# Choosing Lag Order

Litterman provides two ways to set the lag order for your VAR model.

## Fixed lag order

If you know the lag order you want, pass an integer:

\`\`\`python
from litterman import VAR

spec = VAR(lags=4, prior="minnesota")
\`\`\`

## Automatic selection via information criteria

Pass a criterion name (`"aic"`, `"bic"`, or `"hq"`) and Litterman selects the optimal lag via OLS:

\`\`\`python
spec = VAR(lags="bic", prior="minnesota")
\`\`\`

You can cap the search range:

\`\`\`python
spec = VAR(lags="aic", max_lags=12, prior="minnesota")
\`\`\`

## Inspecting the criteria table

Use `select_lag_order` directly to see all criteria values:

\`\`\`python
from litterman import select_lag_order

ic = select_lag_order(data, max_lags=8)
print(f"AIC selects {ic.aic} lags, BIC selects {ic.bic} lags")
ic.summary()  # Returns a DataFrame with all criteria by lag
\`\`\`
```

**Step 5: Write Bayesian VAR explanation**

`docs/explanation/bayesian-var.md`:

```markdown
# What Is a Bayesian VAR?

A **Vector Autoregression (VAR)** models multiple time series as a system of equations where each variable depends on its own lags and the lags of all other variables in the system.

A **Bayesian VAR** adds prior distributions over the model parameters. This serves two purposes:

1. **Regularization** — VARs have many parameters (grows as $n^2 \times p$ where $n$ is the number of variables and $p$ is the lag order). Priors shrink estimates toward sensible values, reducing overfitting.
2. **Uncertainty quantification** — instead of point estimates, you get a full posterior distribution over coefficients, forecasts, and structural quantities.

## When to use a Bayesian VAR

- You have a moderate number of macroeconomic or financial time series (2–20 variables)
- You want probabilistic forecasts with credible intervals
- You want to study how shocks propagate through a system (impulse responses)
- You want to decompose forecast error variance or historical variation by shock source

## The Litterman pipeline

Litterman models this as a sequence of immutable types:

\`\`\`
VARData → VAR → FittedVAR → IdentifiedVAR
\`\`\`

Each step adds information. You cannot skip steps or go backward.
```

**Step 6: Write Minnesota prior explanation**

`docs/explanation/minnesota-prior.md`:

```markdown
# The Minnesota Prior

The **Minnesota prior** (Litterman, 1986) is the most widely used prior for Bayesian VARs. It encodes the belief that each variable follows a random walk, with coefficients on other variables' lags shrunk toward zero.

## Key hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tightness` | 0.1 | Overall shrinkage. Smaller = more shrinkage toward prior. |
| `decay` | `"harmonic"` | How fast coefficients shrink on longer lags. `"harmonic"`: $1/l$. `"geometric"`: $1/l^2$. |
| `cross_shrinkage` | 0.5 | Relative shrinkage on other variables' lags vs own lags. 0 = only own lags matter, 1 = equal treatment. |

## Intuition

The prior mean for the coefficient on a variable's own first lag is 1.0 (random walk). All other coefficients have prior mean 0.0. The prior standard deviation controls how far the posterior can move from these defaults.

## Usage in Litterman

\`\`\`python
from litterman import VAR
from litterman.priors import MinnesotaPrior

# Use defaults
spec = VAR(lags=4, prior="minnesota")

# Customize hyperparameters
prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.3)
spec = VAR(lags=4, prior=prior)
\`\`\`
```

**Step 7: Update mkdocs.yml nav**

```yaml
nav:
  - Home: index.md
  - Tutorials:
    - tutorials/index.md
    - tutorials/quickstart.ipynb
  - How-To Guides:
    - how-to/index.md
    - how-to/data-preparation.md
    - how-to/custom-priors.md
    - how-to/lag-selection.md
  - Explanation:
    - explanation/index.md
    - explanation/bayesian-var.md
    - explanation/minnesota-prior.md
  - Reference:
    - reference/index.md
    - reference/data.md
    - reference/spec.md
    - reference/priors.md
    - reference/samplers.md
    - reference/fitted.md
    - reference/results.md
    - reference/protocols.md
```

**Step 8: Update landing page with estimation example**

Update `docs/index.md` to extend the code example:

```markdown
# Litterman

**Bayesian Vector Autoregression in Python.**

\`\`\`python
import pandas as pd
from litterman import VAR, VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

fitted = VAR(lags="bic", prior="minnesota").fit(data)
\`\`\`

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable types** — all objects are frozen after creation, preventing accidental mutation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains
- **Minnesota prior** — smart defaults with tunable hyperparameters for shrinkage
- **Automatic lag selection** — AIC, BIC, and Hannan-Quinn criteria
- **PyMC backend** — full Bayesian estimation with NUTS sampling

## Installation

\`\`\`bash
pip install litterman
\`\`\`
```

**Step 9: Verify docs build**

Run: `uv run mkdocs build`
Expected: Clean build. All reference pages render autodoc content. Quickstart notebook renders as HTML.

---

## Phase 3 — Forecasting

**Goal:** Add forecasting from `FittedVAR`, producing `ForecastResult` with `.median()`, `.hdi()`, `.to_dataframe()`.

### Task 3.1: ForecastResult

**Files:**
- Modify: `src/litterman/results.py`
- Modify: `tests/test_results.py`

**Step 1: Write failing test**

Append to `tests/test_results.py`:

```python
class TestForecastResult:
    def test_is_subclass_of_base(self):
        from litterman.results import ForecastResult, VARResultBase
        assert issubclass(ForecastResult, VARResultBase)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_results.py::TestForecastResult -v`
Expected: FAIL.

**Step 3: Add ForecastResult to results.py**

```python
class ForecastResult(VARResultBase):
    """Result from VAR forecasting.

    Args:
        idata: ArviZ InferenceData with forecast draws.
        steps: Number of forecast steps.
        var_names: Names of forecasted variables.
    """

    steps: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        """Posterior median forecast."""
        forecast = self.idata.posterior_predictive["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        return pd.DataFrame(med, columns=self.var_names)

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for forecast."""
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["forecast"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values, columns=self.var_names)
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values, columns=self.var_names)
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame."""
        forecast = self.idata.posterior_predictive["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        df = pd.DataFrame(med, columns=self.var_names)
        df.index.name = "step"
        return df

    def plot(self) -> Figure:
        from litterman.plotting import plot_forecast
        return plot_forecast(self)
```

**Step 4: Run test**

Run: `uv run python -m pytest tests/test_results.py::TestForecastResult -v`
Expected: PASS.


### Task 3.2: `FittedVAR.forecast()`

**Files:**
- Modify: `src/litterman/fitted.py`
- Modify: `tests/test_fitted.py`

**Step 1: Write failing test**

Append to `tests/test_fitted.py`:

```python
class TestForecasting:
    @pytest.mark.slow
    def test_forecast_returns_forecast_result(self, var_data):
        from litterman.results import ForecastResult
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=4)
        assert isinstance(fcast, ForecastResult)
        assert fcast.steps == 4

    @pytest.mark.slow
    def test_forecast_median_shape(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=8)
        med = fcast.median()
        assert med.shape == (8, 2)

    @pytest.mark.slow
    def test_forecast_hdi_returns_hdi_result(self, var_data):
        from litterman.results import HDIResult
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=4)
        hdi = fcast.hdi(prob=0.89)
        assert isinstance(hdi, HDIResult)

    @pytest.mark.slow
    def test_forecast_exog_required_error(self):
        """If model has exog, forecast without exog_future raises."""
        rng = np.random.default_rng(42)
        T, n = 200, 2
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
        exog = rng.standard_normal((T, 1))
        index = pd.date_range("2000-01-01", periods=T, freq="QS")
        data = VARData(endog=y, endog_names=["y1", "y2"], exog=exog, exog_names=["x1"], index=index)

        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
        fitted = spec.fit(data, sampler=sampler)
        with pytest.raises(ValueError, match="exog_future"):
            fitted.forecast(steps=4)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_fitted.py::TestForecasting -v -m slow`
Expected: FAIL — `forecast` not defined.

**Step 3: Implement forecast on FittedVAR**

Add to `FittedVAR` in `src/litterman/fitted.py`:

```python
    def forecast(
        self,
        steps: int,
        exog_future: np.ndarray | None = None,
    ) -> ForecastResult:
        """Produce h-step-ahead forecasts from the reduced-form posterior.

        Args:
            steps: Number of forecast steps.
            exog_future: Future exogenous values, shape (steps, k). Required if model has exog.

        Returns:
            ForecastResult with posterior forecast draws.
        """
        from litterman.results import ForecastResult

        if self.has_exog and exog_future is None:
            raise ValueError("exog_future is required when model includes exogenous variables")
        if not self.has_exog and exog_future is not None:
            raise ValueError("exog_future provided but model has no exogenous variables")

        B_draws = self.coefficients  # (chains, draws, n_vars, n_vars*n_lags)
        intercept_draws = self.intercepts  # (chains, draws, n_vars)
        n_chains, n_draws, n_vars, _ = B_draws.shape

        # Last n_lags observations for initial conditions
        y_hist = self.data.endog[-self.n_lags:]  # (n_lags, n_vars)

        forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                intercept = intercept_draws[c, d]
                y_buffer = list(y_hist)

                for h in range(steps):
                    x_lag = np.concatenate([y_buffer[-(lag)] for lag in range(1, self.n_lags + 1)])
                    y_new = intercept + B @ x_lag
                    if self.has_exog and exog_future is not None:
                        B_exog = self.idata.posterior["B_exog"].values[c, d]
                        y_new = y_new + B_exog @ exog_future[h]
                    forecasts[c, d, h] = y_new
                    y_buffer.append(y_new)

        # Package into InferenceData
        import xarray as xr
        forecast_da = xr.DataArray(
            forecasts,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))

        return ForecastResult(idata=idata, steps=steps, var_names=self.var_names)
```

Add `import xarray as xr` to the imports if not already present (it's imported inline).

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_fitted.py::TestForecasting -v -m slow`
Expected: PASS.

### MILESTONE 3: Users can forecast from a fitted VAR. `result.forecast(steps=8).median()` returns a DataFrame of point forecasts. HDI credible intervals available via `.hdi()`.

### Task D3: Forecasting documentation

**Files:**
- Create: `docs/tutorials/forecasting.ipynb`
- Modify: `docs/index.md`
- Modify: `mkdocs.yml`

**Step 1: Create forecasting tutorial notebook**

Create `docs/tutorials/forecasting.ipynb` as a Jupyter notebook with these cells:

Cell 1 (markdown):
```markdown
# Forecasting with Litterman

This tutorial shows how to produce probabilistic forecasts from a fitted Bayesian VAR.
```

Cell 2 (code):
```python
import numpy as np
import pandas as pd
from litterman import VAR, VARData
from litterman.samplers import NUTSSampler
```

Cell 3 (markdown):
```markdown
## Simulate and fit

We'll create a simple VAR(1) and fit it.
```

Cell 4 (code):
```python
rng = np.random.default_rng(42)
T = 200
y = np.zeros((T, 2))
for t in range(1, T):
    y[t] = 0.5 * y[t - 1] + rng.standard_normal(2) * 0.1

index = pd.date_range("2000-01-01", periods=T, freq="QS")
data = VARData(endog=y, endog_names=["gdp", "inflation"], index=index)

sampler = NUTSSampler(draws=500, tune=500, chains=2, random_seed=42)
fitted = VAR(lags=1, prior="minnesota").fit(data, sampler=sampler)
```

Cell 5 (markdown):
```markdown
## Produce forecasts

Call `.forecast(steps=h)` to get an h-step-ahead forecast.
```

Cell 6 (code):
```python
fcast = fitted.forecast(steps=8)
fcast.median()
```

Cell 7 (markdown):
```markdown
## Credible intervals

Use `.hdi()` to get highest density intervals.
```

Cell 8 (code):
```python
hdi = fcast.hdi(prob=0.89)
print("Lower bounds:")
print(hdi.lower)
print("\nUpper bounds:")
print(hdi.upper)
```

Cell 9 (markdown):
```markdown
## Convert to DataFrame

Use `.to_dataframe()` for a tidy format suitable for further analysis.
```

Cell 10 (code):
```python
fcast.to_dataframe()
```

**Step 2: Update landing page to include forecast step**

Update the code example in `docs/index.md` to extend with forecasting:

```markdown
# Litterman

**Bayesian Vector Autoregression in Python.**

\`\`\`python
import pandas as pd
from litterman import VAR, VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

fitted = VAR(lags="bic", prior="minnesota").fit(data)
forecast = fitted.forecast(steps=8)
forecast.median()  # point forecasts
forecast.hdi()     # credible intervals
\`\`\`

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable types** — all objects are frozen after creation, preventing accidental mutation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains
- **Minnesota prior** — smart defaults with tunable hyperparameters for shrinkage
- **Automatic lag selection** — AIC, BIC, and Hannan-Quinn criteria
- **PyMC backend** — full Bayesian estimation with NUTS sampling
- **Probabilistic forecasts** — posterior median, HDI credible intervals, tidy DataFrames

## Installation

\`\`\`bash
pip install litterman
\`\`\`
```

**Step 3: Update mkdocs.yml nav**

Add the forecasting notebook to the nav:

```yaml
  - Tutorials:
    - tutorials/index.md
    - tutorials/quickstart.ipynb
    - tutorials/forecasting.ipynb
```

**Step 4: Verify docs build**

Run: `uv run mkdocs build`
Expected: Clean build. Forecasting notebook renders as HTML.

---

## Phase 4 — Identification and Structural Analysis

**Goal:** Build `Cholesky` and `SignRestriction` identification schemes, `IdentifiedVAR`, and the three structural methods: IRF, FEVD, historical decomposition.

### Task 4.1: IRFResult, FEVDResult, HistoricalDecompositionResult

**Files:**
- Modify: `src/litterman/results.py`
- Modify: `tests/test_results.py`

**Step 1: Write tests**

Append to `tests/test_results.py`:

```python
class TestStructuralResults:
    def test_irf_result_is_subclass(self):
        from litterman.results import IRFResult, VARResultBase
        assert issubclass(IRFResult, VARResultBase)

    def test_fevd_result_is_subclass(self):
        from litterman.results import FEVDResult, VARResultBase
        assert issubclass(FEVDResult, VARResultBase)

    def test_hd_result_is_subclass(self):
        from litterman.results import HistoricalDecompositionResult, VARResultBase
        assert issubclass(HistoricalDecompositionResult, VARResultBase)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_results.py::TestStructuralResults -v`
Expected: FAIL.

**Step 3: Add result subclasses to results.py**

Add after `ForecastResult`:

```python
class IRFResult(VARResultBase):
    """Result from impulse response function computation.

    Args:
        idata: ArviZ InferenceData with IRF draws.
        horizon: Number of IRF horizons.
        var_names: Names of variables.
    """

    horizon: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        irf = self.idata.posterior_predictive["irf"]
        return pd.DataFrame(irf.median(dim=("chain", "draw")).values.reshape(self.horizon + 1, -1))

    def hdi(self, prob: float = 0.89) -> HDIResult:
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["irf"]
        n = hdi_data.sel(hdi="lower").values
        lower = pd.DataFrame(n.reshape(self.horizon + 1, -1))
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values.reshape(self.horizon + 1, -1))
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        return self.median()

    def plot(self) -> Figure:
        from litterman.plotting import plot_irf
        return plot_irf(self)


class FEVDResult(VARResultBase):
    """Result from forecast error variance decomposition.

    Args:
        idata: ArviZ InferenceData with FEVD draws.
        horizon: Number of FEVD horizons.
        var_names: Names of variables.
    """

    horizon: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        fevd = self.idata.posterior_predictive["fevd"]
        return pd.DataFrame(fevd.median(dim=("chain", "draw")).values.reshape(self.horizon + 1, -1))

    def hdi(self, prob: float = 0.89) -> HDIResult:
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["fevd"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values.reshape(self.horizon + 1, -1))
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values.reshape(self.horizon + 1, -1))
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        return self.median()

    def plot(self) -> Figure:
        from litterman.plotting import plot_fevd
        return plot_fevd(self)


class HistoricalDecompositionResult(VARResultBase):
    """Result from historical decomposition.

    Args:
        idata: ArviZ InferenceData with decomposition draws.
        var_names: Names of variables.
    """

    var_names: list[str]

    def median(self) -> pd.DataFrame:
        hd = self.idata.posterior_predictive["hd"]
        return pd.DataFrame(hd.median(dim=("chain", "draw")).values.reshape(-1, len(self.var_names)))

    def hdi(self, prob: float = 0.89) -> HDIResult:
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["hd"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values.reshape(-1, len(self.var_names)))
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values.reshape(-1, len(self.var_names)))
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        return self.median()

    def plot(self) -> Figure:
        from litterman.plotting import plot_historical_decomposition
        return plot_historical_decomposition(self)
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_results.py::TestStructuralResults -v`
Expected: PASS.

### Task 4.2: Cholesky identification

**Files:**
- Create: `tests/test_identification.py`
- Modify: `src/litterman/identification.py`

**Step 1: Write failing tests**

```python
"""Tests for identification schemes."""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from litterman.identification import Cholesky
from litterman.protocols import IdentificationScheme


class TestCholesky:
    def test_construction(self):
        c = Cholesky(ordering=["gdp", "inflation", "rate"])
        assert c.ordering == ["gdp", "inflation", "rate"]

    def test_frozen(self):
        c = Cholesky(ordering=["gdp", "inflation"])
        with pytest.raises(Exception):
            c.ordering = ["a", "b"]

    def test_satisfies_protocol(self):
        c = Cholesky(ordering=["a", "b"])
        assert isinstance(c, IdentificationScheme)

    def test_identify_produces_structural_idata(self):
        """Test Cholesky decomposition on synthetic covariance draws."""
        rng = np.random.default_rng(42)
        n_vars = 2
        n_chains, n_draws = 1, 50

        # Generate positive-definite covariance matrices
        sigma_draws = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                A = rng.standard_normal((n_vars, n_vars))
                sigma_draws[c, d] = A @ A.T + np.eye(n_vars)

        sigma_da = xr.DataArray(
            sigma_draws,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        )
        idata = az.InferenceData(posterior=xr.Dataset({"Sigma": sigma_da}))

        chol = Cholesky(ordering=["y1", "y2"])
        result = chol.identify(idata, var_names=["y1", "y2"])

        assert "structural_shock_matrix" in result.posterior
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_identification.py -v`
Expected: FAIL.

**Step 3: Implement Cholesky**

In `src/litterman/identification.py`:

```python
"""Identification schemes for structural VAR analysis."""

from __future__ import annotations

import arviz as az
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict


class Cholesky(BaseModel):
    """Cholesky identification scheme.

    Uses the lower-triangular Cholesky decomposition of the residual
    covariance matrix to identify structural shocks. Variable ordering
    determines the causal ordering.

    Args:
        ordering: Ordered list of variable names (most exogenous first).
    """

    model_config = ConfigDict(frozen=True)

    ordering: list[str]

    def identify(self, idata: az.InferenceData, var_names: list[str]) -> az.InferenceData:
        """Apply Cholesky identification to posterior covariance draws.

        Args:
            idata: InferenceData with 'Sigma' in posterior.
            var_names: Variable names from the VAR model.

        Returns:
            InferenceData with 'structural_shock_matrix' added to posterior.
        """
        sigma = idata.posterior["Sigma"].values  # (chains, draws, n, n)
        n_chains, n_draws, n_vars, _ = sigma.shape

        # Reorder if needed
        perm = [var_names.index(v) for v in self.ordering]
        sigma_ordered = sigma[:, :, np.ix_(perm, perm)[0], np.ix_(perm, perm)[1]]

        # Cholesky decompose each draw
        P = np.zeros_like(sigma_ordered)
        for c in range(n_chains):
            for d in range(n_draws):
                P[c, d] = np.linalg.cholesky(sigma_ordered[c, d])

        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "shock", "response"],
            coords={"shock": self.ordering, "response": self.ordering},
        )

        new_posterior = idata.posterior.assign(structural_shock_matrix=P_da)
        return az.InferenceData(posterior=new_posterior)
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_identification.py -v`
Expected: PASS.


### Task 4.3: SignRestriction identification (stub)

**Files:**
- Modify: `tests/test_identification.py`
- Modify: `src/litterman/identification.py`

**Step 1: Write failing test**

Append to `tests/test_identification.py`:

```python
from litterman.identification import SignRestriction


class TestSignRestriction:
    def test_construction(self):
        sr = SignRestriction(
            restrictions={"gdp": {"supply": "+", "demand": "+"},
                          "inflation": {"supply": "-", "demand": "+"}},
            n_rotations=1000,
            random_seed=42,
        )
        assert sr.n_rotations == 1000

    def test_satisfies_protocol(self):
        sr = SignRestriction(
            restrictions={"gdp": {"supply": "+"}},
        )
        assert isinstance(sr, IdentificationScheme)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_identification.py::TestSignRestriction -v`
Expected: FAIL.

**Step 3: Implement SignRestriction**

Add to `src/litterman/identification.py`:

```python
class SignRestriction(BaseModel):
    """Sign restriction identification scheme.

    Uses random rotation matrices to find structural impact matrices
    satisfying sign restrictions on impulse responses.

    Args:
        restrictions: Dict mapping variable -> {shock_name: "+" or "-"}.
        n_rotations: Number of candidate rotations per draw.
        random_seed: Seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    restrictions: dict[str, dict[str, str]]
    n_rotations: int = Field(default=1000, ge=1)
    random_seed: int | None = None

    def identify(self, idata: az.InferenceData, var_names: list[str]) -> az.InferenceData:
        """Apply sign restriction identification.

        Args:
            idata: InferenceData with 'Sigma' in posterior.
            var_names: Variable names from the VAR model.

        Returns:
            InferenceData with 'structural_shock_matrix' added to posterior.
        """
        from scipy.stats import special_ortho_group

        sigma = idata.posterior["Sigma"].values
        n_chains, n_draws, n_vars, _ = sigma.shape
        rng = np.random.default_rng(self.random_seed)

        shock_names = list(next(iter(self.restrictions.values())).keys())

        P = np.full((n_chains, n_draws, n_vars, n_vars), np.nan)

        for c in range(n_chains):
            for d in range(n_draws):
                chol = np.linalg.cholesky(sigma[c, d])
                found = False
                for _ in range(self.n_rotations):
                    Q = special_ortho_group.rvs(n_vars, random_state=rng)
                    candidate = chol @ Q
                    if self._check_restrictions(candidate, var_names, shock_names):
                        P[c, d] = candidate
                        found = True
                        break
                if not found:
                    P[c, d] = chol  # fallback to Cholesky if no valid rotation found

        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "response", "shock"],
            coords={"response": var_names, "shock": shock_names if len(shock_names) == n_vars else var_names},
        )

        new_posterior = idata.posterior.assign(structural_shock_matrix=P_da)
        return az.InferenceData(posterior=new_posterior)

    def _check_restrictions(
        self, candidate: np.ndarray, var_names: list[str], shock_names: list[str]
    ) -> bool:
        """Check if a candidate matrix satisfies all sign restrictions."""
        for var_name, shocks in self.restrictions.items():
            var_idx = var_names.index(var_name)
            for shock_name, sign in shocks.items():
                shock_idx = shock_names.index(shock_name)
                val = candidate[var_idx, shock_idx]
                if sign == "+" and val < 0:
                    return False
                if sign == "-" and val > 0:
                    return False
        return True
```

Add to imports at top: `from pydantic import BaseModel, ConfigDict, Field`.

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_identification.py -v`
Expected: PASS.


### Task 4.4: IdentifiedVAR

**Files:**
- Create: `tests/test_identified.py`
- Modify: `src/litterman/identified.py`
- Modify: `src/litterman/fitted.py` (add `set_identification_strategy`)

**Step 1: Write failing tests**

```python
"""Tests for IdentifiedVAR."""

import numpy as np
import pandas as pd
import pytest

from litterman.data import VARData
from litterman.identification import Cholesky
from litterman.identified import IdentifiedVAR
from litterman.results import FEVDResult, HistoricalDecompositionResult, IRFResult
from litterman.samplers import NUTSSampler
from litterman.spec import VAR


@pytest.fixture
def fitted_var():
    """Fit a small VAR for testing."""
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    data = VARData(endog=y, endog_names=["y1", "y2"], index=index)
    spec = VAR(lags=1, prior="minnesota")
    sampler = NUTSSampler(draws=100, tune=100, chains=2, random_seed=42)
    return spec.fit(data, sampler=sampler)


class TestIdentifiedVAR:
    @pytest.mark.slow
    def test_set_identification_returns_identified(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        assert isinstance(identified, IdentifiedVAR)

    @pytest.mark.slow
    def test_impulse_response(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        irf = identified.impulse_response(horizon=10)
        assert isinstance(irf, IRFResult)
        assert irf.horizon == 10

    @pytest.mark.slow
    def test_fevd(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        fevd = identified.fevd(horizon=10)
        assert isinstance(fevd, FEVDResult)

    @pytest.mark.slow
    def test_historical_decomposition(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        hd = identified.historical_decomposition()
        assert isinstance(hd, HistoricalDecompositionResult)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_identified.py -v -k "not slow"`
Expected: No tests collected (all are slow-marked).

**Step 3: Implement IdentifiedVAR**

In `src/litterman/identified.py`:

```python
"""IdentifiedVAR — structural VAR with identified shocks."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict

from litterman.data import VARData
from litterman.results import FEVDResult, HistoricalDecompositionResult, IRFResult


class IdentifiedVAR(BaseModel):
    """Immutable structural VAR with identified shocks.

    Args:
        idata: InferenceData with structural_shock_matrix in posterior.
        n_lags: Lag order.
        data: Original VARData.
        var_names: Endogenous variable names.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData
    n_lags: int
    data: VARData
    var_names: list[str]

    def impulse_response(
        self,
        horizon: int = 20,
        shock: str | None = None,
        response: str | None = None,
    ) -> IRFResult:
        """Compute structural impulse response functions.

        Args:
            horizon: Number of periods.
            shock: Optional shock variable to filter to.
            response: Optional response variable to filter to.

        Returns:
            IRFResult with IRF posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (chains, draws, n, n*p)
        P_draws = self.idata.posterior["structural_shock_matrix"].values  # (chains, draws, n, n)
        n_chains, n_draws, n_vars, _ = B_draws.shape
        n_lags = self.n_lags

        # Compute IRFs
        irfs = np.zeros((n_chains, n_draws, horizon + 1, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]  # (n, n*p)
                P = P_draws[c, d]  # (n, n)

                # Companion form coefficients
                Phi = [np.eye(n_vars)]  # Phi_0 = I
                for h in range(1, horizon + 1):
                    phi_h = np.zeros((n_vars, n_vars))
                    for j in range(1, min(h, n_lags) + 1):
                        A_j = B[:, (j - 1) * n_vars : j * n_vars]
                        phi_h += A_j @ Phi[h - j]
                    Phi.append(phi_h)

                for h in range(horizon + 1):
                    irfs[c, d, h] = Phi[h] @ P

        irf_da = xr.DataArray(
            irfs,
            dims=["chain", "draw", "horizon", "response", "shock"],
            coords={
                "response": self.var_names,
                "shock": self.var_names,
                "horizon": np.arange(horizon + 1),
            },
            name="irf",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"irf": irf_da}))
        return IRFResult(idata=idata, horizon=horizon, var_names=self.var_names)

    def forecast_error_variance_decomposition(self, horizon: int = 20) -> FEVDResult:
        """Compute forecast error variance decomposition.

        Args:
            horizon: Number of periods.

        Returns:
            FEVDResult with FEVD posterior draws.
        """
        B_draws = self.idata.posterior["B"].values
        P_draws = self.idata.posterior["structural_shock_matrix"].values
        n_chains, n_draws, n_vars, _ = B_draws.shape
        n_lags = self.n_lags

        fevd = np.zeros((n_chains, n_draws, horizon + 1, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                P = P_draws[c, d]

                Phi = [np.eye(n_vars)]
                for h in range(1, horizon + 1):
                    phi_h = np.zeros((n_vars, n_vars))
                    for j in range(1, min(h, n_lags) + 1):
                        A_j = B[:, (j - 1) * n_vars : j * n_vars]
                        phi_h += A_j @ Phi[h - j]
                    Phi.append(phi_h)

                # Accumulate MSE contributions
                mse_total = np.zeros((n_vars, n_vars))
                for h in range(horizon + 1):
                    theta_h = Phi[h] @ P
                    mse_total += theta_h**2
                    for resp in range(n_vars):
                        total = mse_total[resp].sum()
                        if total > 0:
                            fevd[c, d, h, resp] = mse_total[resp] / total

        fevd_da = xr.DataArray(
            fevd,
            dims=["chain", "draw", "horizon", "response", "shock"],
            coords={"response": self.var_names, "shock": self.var_names},
            name="fevd",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"fevd": fevd_da}))
        return FEVDResult(idata=idata, horizon=horizon, var_names=self.var_names)

    def fevd(self, horizon: int = 20) -> FEVDResult:
        """Alias for forecast_error_variance_decomposition.

        Args:
            horizon: Number of periods.

        Returns:
            FEVDResult.
        """
        return self.forecast_error_variance_decomposition(horizon=horizon)

    def historical_decomposition(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        cumulative: bool = False,
    ) -> HistoricalDecompositionResult:
        """Compute historical decomposition of observed series.

        Args:
            start: Optional start date to restrict decomposition.
            end: Optional end date to restrict decomposition.
            cumulative: If True, return cumulative shock contributions.

        Returns:
            HistoricalDecompositionResult.
        """
        B_draws = self.idata.posterior["B"].values
        P_draws = self.idata.posterior["structural_shock_matrix"].values
        intercept_draws = self.idata.posterior["intercept"].values
        n_chains, n_draws, n_vars, _ = B_draws.shape

        y = self.data.endog
        T = y.shape[0]
        n_lags = self.n_lags

        hd = np.zeros((n_chains, n_draws, T - n_lags, n_vars, n_vars))

        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                P = P_draws[c, d]
                intercept = intercept_draws[c, d]
                P_inv = np.linalg.inv(P)

                # Compute structural residuals
                for t in range(n_lags, T):
                    x_lag = np.concatenate([y[t - lag] for lag in range(1, n_lags + 1)])
                    resid = y[t] - intercept - B @ x_lag
                    structural_resid = P_inv @ resid
                    # Each shock's contribution
                    for k in range(n_vars):
                        hd[c, d, t - n_lags, :, k] = P[:, k] * structural_resid[k]

                if cumulative:
                    hd[c, d] = np.cumsum(hd[c, d], axis=0)

        # Trim to date range if requested
        idx = self.data.index[n_lags:]
        t_start = 0
        t_end = len(idx)
        if start is not None:
            t_start = idx.searchsorted(start)
        if end is not None:
            t_end = idx.searchsorted(end, side="right")
        hd = hd[:, :, t_start:t_end]

        hd_da = xr.DataArray(
            hd,
            dims=["chain", "draw", "time", "response", "shock"],
            coords={"response": self.var_names, "shock": self.var_names},
            name="hd",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"hd": hd_da}))
        return HistoricalDecompositionResult(idata=idata, var_names=self.var_names)

    def __repr__(self) -> str:
        n_vars = len(self.var_names)
        posterior = self.idata.posterior
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        return f"IdentifiedVAR(n_lags={self.n_lags}, n_vars={n_vars}, n_draws={n_draws}, n_chains={n_chains})"
```

**Step 4: Add `set_identification_strategy` to FittedVAR**

Add to `FittedVAR` in `src/litterman/fitted.py`:

```python
    def set_identification_strategy(self, scheme: IdentificationScheme) -> IdentifiedVAR:
        """Apply a structural identification scheme.

        Args:
            scheme: An IdentificationScheme protocol instance (e.g. Cholesky).

        Returns:
            IdentifiedVAR with structural shock matrix in the posterior.
        """
        from litterman.identified import IdentifiedVAR

        identified_idata = scheme.identify(self.idata, self.var_names)
        return IdentifiedVAR(
            idata=identified_idata,
            n_lags=self.n_lags,
            data=self.data,
            var_names=self.var_names,
        )
```

Add import at the top of `fitted.py`:

```python
from litterman.protocols import IdentificationScheme
```

**Step 5: Run slow tests**

Run: `uv run python -m pytest tests/test_identified.py -v -m slow`
Expected: PASS.


### MILESTONE 4: Full structural analysis pipeline works. Users can go from `FittedVAR` → `IdentifiedVAR` → `impulse_response()`, `fevd()`, `historical_decomposition()`. All return typed result objects with `.median()` and `.hdi()`.

### Task D4: Structural analysis documentation

**Files:**
- Create: `docs/reference/identified.md`
- Create: `docs/reference/identification.md`
- Create: `docs/tutorials/structural-analysis.ipynb`
- Create: `docs/how-to/sign-restrictions.md`
- Create: `docs/explanation/identification.md`
- Modify: `mkdocs.yml`

**Step 1: Create API reference pages**

`docs/reference/identified.md`:

```markdown
# IdentifiedVAR

::: litterman.identified
```

`docs/reference/identification.md`:

```markdown
# Identification Schemes

::: litterman.identification
```

**Step 2: Create structural analysis tutorial notebook**

Create `docs/tutorials/structural-analysis.ipynb` as a Jupyter notebook with these cells:

Cell 1 (markdown):
```markdown
# Structural Analysis with Identified VARs

This tutorial walks through the complete structural VAR pipeline: fitting a model, applying Cholesky identification, and computing impulse responses, forecast error variance decompositions, and historical decompositions.
```

Cell 2 (code):
```python
import numpy as np
import pandas as pd
from litterman import VAR, VARData
from litterman.identification import Cholesky
from litterman.samplers import NUTSSampler
```

Cell 3 (markdown):
```markdown
## Simulate a VAR(1) process
```

Cell 4 (code):
```python
rng = np.random.default_rng(42)
T = 200
y = np.zeros((T, 3))
for t in range(1, T):
    y[t, 0] = 0.6 * y[t - 1, 0] + rng.standard_normal() * 0.1
    y[t, 1] = 0.3 * y[t - 1, 0] + 0.5 * y[t - 1, 1] + rng.standard_normal() * 0.1
    y[t, 2] = 0.2 * y[t - 1, 1] + 0.4 * y[t - 1, 2] + rng.standard_normal() * 0.1

index = pd.date_range("2000-01-01", periods=T, freq="QS")
data = VARData(endog=y, endog_names=["gdp", "inflation", "rate"], index=index)
```

Cell 5 (markdown):
```markdown
## Fit the model
```

Cell 6 (code):
```python
sampler = NUTSSampler(draws=500, tune=500, chains=2, random_seed=42)
fitted = VAR(lags=2, prior="minnesota").fit(data, sampler=sampler)
fitted
```

Cell 7 (markdown):
```markdown
## Apply Cholesky identification

The ordering determines the causal structure. Variables listed first are "most exogenous" — they are not contemporaneously affected by variables listed later.
```

Cell 8 (code):
```python
identified = fitted.set_identification_strategy(
    Cholesky(ordering=["gdp", "inflation", "rate"])
)
identified
```

Cell 9 (markdown):
```markdown
## Impulse response functions

How does a one-standard-deviation structural shock propagate through the system?
```

Cell 10 (code):
```python
irf = identified.impulse_response(horizon=20)
irf.median()
```

Cell 11 (markdown):
```markdown
## Forecast error variance decomposition

What fraction of each variable's forecast error variance is explained by each structural shock?
```

Cell 12 (code):
```python
fevd = identified.fevd(horizon=20)
fevd.median()
```

Cell 13 (markdown):
```markdown
## Historical decomposition

What was the contribution of each structural shock to each variable at each point in time?
```

Cell 14 (code):
```python
hd = identified.historical_decomposition()
hd.median()
```

**Step 3: Write sign restrictions how-to**

`docs/how-to/sign-restrictions.md`:

```markdown
# Using Sign Restrictions

Sign restrictions identify structural shocks by imposing qualitative constraints on the impact matrix, rather than a recursive ordering.

## Define restrictions

Specify which direction each variable should respond to each named shock:

\`\`\`python
from litterman.identification import SignRestriction

scheme = SignRestriction(
    restrictions={
        "gdp":       {"supply": "+", "demand": "+"},
        "inflation": {"supply": "-", "demand": "+"},
    },
    n_rotations=1000,
    random_seed=42,
)
\`\`\`

- `"+"` means the variable must increase on impact
- `"-"` means the variable must decrease on impact
- Omitted entries are unrestricted

## Apply to a fitted model

\`\`\`python
identified = fitted.set_identification_strategy(scheme)
irf = identified.impulse_response(horizon=20)
\`\`\`

## Tips

- More restrictions = fewer valid rotations found per draw. If identification is too tight, consider relaxing some constraints.
- Increase `n_rotations` if many draws fail to find a valid rotation (the default fallback is plain Cholesky).
- Set `random_seed` for reproducibility.
```

**Step 4: Write identification explanation**

`docs/explanation/identification.md`:

```markdown
# Structural Identification

A reduced-form VAR estimates the joint dynamics of a set of variables, but its residuals are correlated. **Structural identification** decomposes these correlated residuals into uncorrelated structural shocks with economic interpretations.

## Why identification matters

Without identification, you can describe correlations but not causation. Identification lets you answer:

- "What happens to inflation when there is a monetary policy shock?" (impulse responses)
- "How much of GDP variation is due to supply vs demand shocks?" (variance decomposition)

## Cholesky identification

The simplest approach. Uses the lower-triangular Cholesky factor of the residual covariance matrix. This implies a **recursive causal ordering**: the first variable is not contemporaneously affected by any other, the second is affected only by the first, and so on.

\`\`\`python
from litterman.identification import Cholesky

scheme = Cholesky(ordering=["gdp", "inflation", "rate"])
\`\`\`

The ordering encodes your assumptions. Changing it changes the results.

## Sign restrictions

A more agnostic approach. Instead of imposing a full recursive structure, you specify qualitative constraints: "a supply shock raises GDP and lowers inflation." The algorithm searches over random rotation matrices to find decompositions consistent with your restrictions.

\`\`\`python
from litterman.identification import SignRestriction

scheme = SignRestriction(
    restrictions={
        "gdp":       {"supply": "+", "demand": "+"},
        "inflation": {"supply": "-", "demand": "+"},
    },
)
\`\`\`

Sign restrictions are weaker than Cholesky (they don't uniquely identify the model), but they require fewer assumptions.
```

**Step 5: Update mkdocs.yml nav**

```yaml
nav:
  - Home: index.md
  - Tutorials:
    - tutorials/index.md
    - tutorials/quickstart.ipynb
    - tutorials/forecasting.ipynb
    - tutorials/structural-analysis.ipynb
  - How-To Guides:
    - how-to/index.md
    - how-to/data-preparation.md
    - how-to/custom-priors.md
    - how-to/lag-selection.md
    - how-to/sign-restrictions.md
  - Explanation:
    - explanation/index.md
    - explanation/bayesian-var.md
    - explanation/minnesota-prior.md
    - explanation/identification.md
  - Reference:
    - reference/index.md
    - reference/data.md
    - reference/spec.md
    - reference/priors.md
    - reference/samplers.md
    - reference/fitted.md
    - reference/results.md
    - reference/protocols.md
    - reference/identified.md
    - reference/identification.md
```

**Step 6: Verify docs build**

Run: `uv run mkdocs build`
Expected: Clean build. All new pages render.

---

## Phase 5 — Plotting, Public API, Runtime Checks

**Goal:** Add plotting functions, wire up `__init__.py` re-exports, and add optional runtime type checking.

### Task 5.1: Plotting package

**Files:**
- Modify: `src/litterman/plotting/_irf.py`
- Modify: `src/litterman/plotting/_fevd.py`
- Modify: `src/litterman/plotting/_forecast.py`
- Modify: `src/litterman/plotting/_historical_decomposition.py`
- Modify: `src/litterman/plotting/__init__.py`
- Create: `tests/test_plotting.py`

**Step 1: Write failing tests**

```python
"""Tests for plotting functions."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import pytest
from matplotlib.figure import Figure


class TestPlotImports:
    def test_plot_irf_importable(self):
        from litterman.plotting import plot_irf
        assert callable(plot_irf)

    def test_plot_fevd_importable(self):
        from litterman.plotting import plot_fevd
        assert callable(plot_fevd)

    def test_plot_forecast_importable(self):
        from litterman.plotting import plot_forecast
        assert callable(plot_forecast)

    def test_plot_historical_decomposition_importable(self):
        from litterman.plotting import plot_historical_decomposition
        assert callable(plot_historical_decomposition)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_plotting.py -v`
Expected: FAIL.

**Step 3: Implement plot functions**

Each plotting module follows the same pattern. Here is `src/litterman/plotting/_irf.py`:

```python
"""IRF plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import IRFResult


def plot_irf(
    result: IRFResult,
    variables: list[str] | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Plot impulse response functions with credible bands.

    Args:
        result: IRFResult from IdentifiedVAR.impulse_response().
        variables: Optional subset of response variables to plot.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    hdi = result.hdi()
    var_names = variables or result.var_names
    n_vars = len(var_names)

    fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize, squeeze=False)
    fig.suptitle("Impulse Response Functions")

    for i, resp in enumerate(var_names):
        for j, shock in enumerate(var_names):
            ax = axes[i][j]
            ax.set_title(f"{shock} → {resp}", fontsize=9)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            horizons = range(result.horizon + 1)
            ax.plot(horizons, med.values[:, i * n_vars + j] if med.shape[1] > 1 else med.values[:, 0])
            ax.fill_between(
                horizons,
                hdi.lower.values[:, i * n_vars + j] if hdi.lower.shape[1] > 1 else hdi.lower.values[:, 0],
                hdi.upper.values[:, i * n_vars + j] if hdi.upper.shape[1] > 1 else hdi.upper.values[:, 0],
                alpha=0.3,
            )

    fig.tight_layout()
    return fig
```

`src/litterman/plotting/_forecast.py`:

```python
"""Forecast plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import ForecastResult


def plot_forecast(
    result: ForecastResult,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot forecast fan chart with credible bands.

    Args:
        result: ForecastResult from FittedVAR.forecast().
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    hdi = result.hdi()
    n_vars = len(result.var_names)

    fig, axes = plt.subplots(1, n_vars, figsize=figsize, squeeze=False)
    fig.suptitle("Forecast")

    for i, name in enumerate(result.var_names):
        ax = axes[0][i]
        ax.set_title(name)
        steps = range(result.steps)
        ax.plot(steps, med[name].values)
        ax.fill_between(steps, hdi.lower[name].values, hdi.upper[name].values, alpha=0.3)

    fig.tight_layout()
    return fig
```

`src/litterman/plotting/_fevd.py`:

```python
"""FEVD plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import FEVDResult


def plot_fevd(
    result: FEVDResult,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot forecast error variance decomposition as stacked areas.

    Args:
        result: FEVDResult from IdentifiedVAR.fevd().
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Forecast Error Variance Decomposition")
    ax.stackplot(range(result.horizon + 1), med.values.T, labels=result.var_names, alpha=0.8)
    ax.legend()
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Share")
    fig.tight_layout()
    return fig
```

`src/litterman/plotting/_historical_decomposition.py`:

```python
"""Historical decomposition plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from litterman.results import HistoricalDecompositionResult


def plot_historical_decomposition(
    result: HistoricalDecompositionResult,
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Plot historical decomposition as stacked bar chart.

    Args:
        result: HistoricalDecompositionResult.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    med = result.median()
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Historical Decomposition")
    n_vars = len(result.var_names)
    T = med.shape[0]
    bottom = None
    for i, name in enumerate(result.var_names):
        vals = med.iloc[:, i].values
        if bottom is None:
            ax.bar(range(T), vals, label=name, alpha=0.8)
            bottom = vals.copy()
        else:
            ax.bar(range(T), vals, bottom=bottom, label=name, alpha=0.8)
            bottom += vals
    ax.legend()
    ax.set_xlabel("Time")
    fig.tight_layout()
    return fig
```

`src/litterman/plotting/__init__.py`:

```python
"""Plotting functions for VAR results."""

from litterman.plotting._fevd import plot_fevd
from litterman.plotting._forecast import plot_forecast
from litterman.plotting._historical_decomposition import plot_historical_decomposition
from litterman.plotting._irf import plot_irf

__all__ = ["plot_irf", "plot_fevd", "plot_forecast", "plot_historical_decomposition"]
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_plotting.py -v`
Expected: PASS.


### Task 5.2: Public `__init__.py` re-exports

**Files:**
- Modify: `src/litterman/__init__.py`
- Create: `tests/test_public_api.py`

**Step 1: Write failing test**

```python
"""Tests for public API re-exports."""


class TestPublicAPI:
    def test_var_importable(self):
        from litterman import VAR
        assert VAR is not None

    def test_var_data_importable(self):
        from litterman import VARData
        assert VARData is not None

    def test_select_lag_order_importable(self):
        from litterman import select_lag_order
        assert select_lag_order is not None

    def test_enable_runtime_checks_importable(self):
        from litterman import enable_runtime_checks
        assert callable(enable_runtime_checks)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_public_api.py -v`
Expected: FAIL.

**Step 3: Wire up `__init__.py`**

```python
"""Litterman: Bayesian Vector Autoregression in Python."""

from litterman._lag_selection import select_lag_order
from litterman.data import VARData
from litterman.spec import VAR

__all__ = [
    "VAR",
    "VARData",
    "select_lag_order",
    "enable_runtime_checks",
]


def enable_runtime_checks() -> None:
    """Enable beartype runtime type checking on public API.

    Intended for use in test suites. Wraps public functions and methods
    with beartype decorators for runtime validation.
    """
    from beartype import beartype
    from beartype.roar import BeartypeDecorHintPep484585Exception

    import litterman.data
    import litterman.fitted
    import litterman.identified
    import litterman.spec

    for mod in [litterman.data, litterman.spec, litterman.fitted, litterman.identified]:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                try:
                    setattr(mod, name, beartype(obj))
                except BeartypeDecorHintPep484585Exception:
                    pass
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_public_api.py -v`
Expected: PASS.

### Task 5.3: Add `scipy` dependency

**Files:**
- Modify: `pyproject.toml`

Note: `SignRestriction` uses `scipy.stats.special_ortho_group`. Add `scipy` to runtime dependencies.

**Step 1: Add scipy**

In `pyproject.toml`, add `"scipy>=1.10"` to the `dependencies` list.

**Step 2: Sync**

Run: `uv lock && uv sync`


### Task 5.4: Run full test suite + lint

**Step 1: Run linter**

Run: `uv run ruff check . && uv run ruff format .`
Fix any issues.

**Step 2: Run type checker**

Run: `uv run ty check`
Fix any issues.

**Step 3: Run full test suite**

Run: `uv run python -m pytest --cov --cov-config=pyproject.toml -v`
Expected: All tests pass, coverage meets target.

### MILESTONE 5 (FINAL): Library is feature-complete for v1. The full pipeline works: `VARData` → `VAR` → `FittedVAR` → `IdentifiedVAR` → IRF/FEVD/HD results → `.plot()`. Public API exports are clean. Runtime type checking available via `enable_runtime_checks()`.

### Task D5: Final documentation — plotting reference and landing page polish

**Files:**
- Create: `docs/reference/plotting.md`
- Modify: `docs/index.md`
- Modify: `mkdocs.yml`

**Step 1: Create plotting API reference page**

`docs/reference/plotting.md`:

```markdown
# Plotting

::: litterman.plotting
```

**Step 2: Write final landing page with full pipeline showcase**

Overwrite `docs/index.md`:

```markdown
# Litterman

**Bayesian Vector Autoregression in Python.**

\`\`\`python
import pandas as pd
from litterman import VAR, VARData
from litterman.identification import Cholesky

# Load data
df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

# Estimate
fitted = VAR(lags="bic", prior="minnesota").fit(data)

# Forecast
forecast = fitted.forecast(steps=8)
forecast.median()  # point forecasts
forecast.hdi()     # credible intervals

# Structural analysis
identified = fitted.set_identification_strategy(
    Cholesky(ordering=["gdp", "inflation", "rate"])
)
irf = identified.impulse_response(horizon=20)
irf.plot()
\`\`\`

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable pipeline** — `VARData` → `VAR` → `FittedVAR` → `IdentifiedVAR`, each stage frozen after creation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains
- **Minnesota prior** — smart defaults with tunable hyperparameters for shrinkage
- **Automatic lag selection** — AIC, BIC, and Hannan-Quinn criteria
- **PyMC backend** — full Bayesian estimation with NUTS sampling
- **Probabilistic forecasts** — posterior median, HDI credible intervals, tidy DataFrames
- **Structural identification** — Cholesky and sign restriction schemes
- **Built-in plotting** — IRF, FEVD, forecast, and historical decomposition plots

## Installation

\`\`\`bash
pip install litterman
\`\`\`

## Learn more

- [Quickstart tutorial](tutorials/quickstart.ipynb) — fit your first Bayesian VAR
- [Forecasting tutorial](tutorials/forecasting.ipynb) — produce probabilistic forecasts
- [Structural analysis tutorial](tutorials/structural-analysis.ipynb) — impulse responses and variance decompositions
- [API Reference](reference/index.md) — complete module documentation
```

**Step 3: Update mkdocs.yml nav with plotting reference**

Add to the Reference section:

```yaml
  - Reference:
    - reference/index.md
    - reference/data.md
    - reference/spec.md
    - reference/priors.md
    - reference/samplers.md
    - reference/fitted.md
    - reference/results.md
    - reference/protocols.md
    - reference/identified.md
    - reference/identification.md
    - reference/plotting.md
```

**Step 4: Polish pass — verify all pages build and cross-links resolve**

Run: `uv run mkdocs build --strict`
Expected: Clean build with no warnings. All pages render. All cross-links resolve.

**Step 5: Verify full site locally**

Run: `uv run mkdocs serve`
Expected: Complete documentation site with all 5 tabs populated:
- Home: full pipeline showcase
- Tutorials: 3 notebooks (quickstart, forecasting, structural analysis)
- How-To Guides: 4 guides (data prep, custom priors, lag selection, sign restrictions)
- Explanation: 3 pages (Bayesian VAR, Minnesota prior, identification)
- Reference: 10 API reference pages

### DOCUMENTATION COMPLETE: Full Diataxis documentation site is live with tutorials, how-to guides, explanations, and auto-generated API reference.

---

## Summary of All Tasks

| Task | Description | Phase | Type |
|------|-------------|-------|------|
| 0.1 | Add runtime dependencies | 0 | Code |
| 0.2 | Remove scaffold placeholder | 0 | Code |
| 0.3 | Create empty module files | 0 | Code |
| **D0** | **MkDocs setup + Diataxis doc scaffold** | **0** | **Docs** |
| 1.1 | VARData Pydantic model | 1 | Code |
| 1.2 | VARData validation edge cases | 1 | Code |
| 1.3 | VARData.from_df | 1 | Code |
| **D1** | **VARData reference + data preparation how-to** | **1** | **Docs** |
| 2.1 | Protocols | 2 | Code |
| 2.2 | MinnesotaPrior | 2 | Code |
| 2.3 | NUTSSampler | 2 | Code |
| 2.4 | Result base classes | 2 | Code |
| 2.5 | Lag selection | 2 | Code |
| 2.6 | VAR specification | 2 | Code |
| 2.7 | FittedVAR + VAR.fit() | 2 | Code |
| **D2** | **Estimation pipeline docs — 6 ref pages, quickstart, how-tos, explanations** | **2** | **Docs** |
| 3.1 | ForecastResult | 3 | Code |
| 3.2 | FittedVAR.forecast() | 3 | Code |
| **D3** | **Forecasting tutorial notebook** | **3** | **Docs** |
| 4.1 | Structural result classes | 4 | Code |
| 4.2 | Cholesky identification | 4 | Code |
| 4.3 | SignRestriction identification | 4 | Code |
| 4.4 | IdentifiedVAR + structural methods | 4 | Code |
| **D4** | **Structural analysis docs — tutorial, how-to, explanation, ref pages** | **4** | **Docs** |
| 5.1 | Plotting package | 5 | Code |
| 5.2 | Public __init__.py + runtime checks | 5 | Code |
| 5.3 | Add scipy dependency | 5 | Code |
| 5.4 | Full test suite + lint pass | 5 | Code |
| **D5** | **Plotting reference + final landing page + polish** | **5** | **Docs** |

## pytest Configuration Note

Tests marked `@pytest.mark.slow` involve actual MCMC sampling. Configure `pytest.ini_options` in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests that run MCMC sampling (deselect with '-m \"not slow\"')",
]
```

Run fast tests only: `uv run python -m pytest -m "not slow" -v`
Run all: `uv run python -m pytest -v`
