# Litterman Documentation Design

## Overview

Documentation is built alongside the library, with a documentation task after each of the 6 milestones (D0-D5). This ensures the docs site is always inspectable and reflects the current state of the API.

## Tooling

- **Engine:** MkDocs Material (existing setup, extended)
- **API autodoc:** mkdocstrings[python] (already configured)
- **Notebooks:** mkdocs-jupyter for tutorial .ipynb files
- **Format:** Markdown (.md) for guides/explanation/reference, Jupyter (.ipynb) for tutorials

## Structure (Diataxis)

```
docs/
  index.md                                  # Code-first landing page
  tutorials/
    index.md                                # Section landing
    quickstart.ipynb                        # D2: data -> fit -> inspect
    forecasting.ipynb                       # D3: fit -> forecast -> plot
    structural-analysis.ipynb               # D4: full SVAR pipeline
  how-to/
    index.md                                # Section landing
    data-preparation.md                     # D1: VARData from DataFrames
    custom-priors.md                        # D2: writing a Prior protocol impl
    lag-selection.md                        # D2: select_lag_order + lags="bic"
    sign-restrictions.md                    # D4: sign restriction setup
  explanation/
    index.md                                # Section landing
    bayesian-var.md                         # D2: what is a Bayesian VAR
    minnesota-prior.md                      # D2: prior theory + hyperparameters
    identification.md                       # D4: Cholesky vs sign restrictions
  reference/
    index.md                                # Section landing
    data.md                                 # D1: ::: litterman.data
    spec.md                                 # D2: ::: litterman.spec
    priors.md                               # D2: ::: litterman.priors
    samplers.md                             # D2: ::: litterman.samplers
    fitted.md                               # D2: ::: litterman.fitted
    results.md                              # D2: ::: litterman.results
    protocols.md                            # D2: ::: litterman.protocols
    identified.md                           # D4: ::: litterman.identified
    identification.md                       # D4: ::: litterman.identification
    plotting.md                             # D5: ::: litterman.plotting
```

## Task Schedule

| Task | After | What gets written | Effort |
|------|-------|-------------------|--------|
| D0 | Milestone 0 | mkdocs.yml expansion, mkdocs-jupyter dep, 4 section index pages | Small |
| D1 | Milestone 1 | `reference/data.md`, `how-to/data-preparation.md`, update `index.md` | Small |
| D2 | Milestone 2 | 6 reference pages, `tutorials/quickstart.ipynb`, 2 how-tos, 2 explanations | Large |
| D3 | Milestone 3 | `tutorials/forecasting.ipynb`, update `index.md` | Small |
| D4 | Milestone 4 | 2 reference pages, `tutorials/structural-analysis.ipynb`, 1 how-to, 1 explanation | Medium |
| D5 | Milestone 5 | `reference/plotting.md`, final `index.md`, polish pass | Small |

## Landing Page Design

Code-first showcase. Lead with a complete pipeline example (VARData -> VAR -> fit -> identify -> forecast -> plot), followed by feature highlights and install instructions. The example grows as milestones complete:

- After D1: VARData construction
- After D2: VARData -> VAR -> fit
- After D3: + forecast
- After D4: + identification + IRF
- After D5: + plot (final version)

## MkDocs Configuration

Extend `mkdocs.yml` with:
- `mkdocs-jupyter` plugin
- Full Diataxis nav tree
- Material features: `navigation.tabs`, `navigation.sections`, `navigation.indexes`, `content.code.copy`, `content.tabs.link`
- Markdown extensions: `admonition`, `pymdownx.details`, `pymdownx.superfences`, `pymdownx.tabbed`

## API Reference Convention

Each reference page is a single autodoc directive:

```markdown
# ModuleName

::: litterman.module_name
    options:
      show_source: false
      heading_level: 2
```

mkdocstrings renders Google-style docstrings from the source code. No manual API docs to maintain.
