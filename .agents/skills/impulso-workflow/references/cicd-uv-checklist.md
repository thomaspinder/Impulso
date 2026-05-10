# CI/CD and uv Checklist

## uv Configuration

- [ ] **`uv.lock` committed**: Lock file is in version control for reproducible installs
- [ ] **Dependency groups**: Dev, test, docs dependencies separated via `[dependency-groups]` in pyproject.toml (uv's native mechanism, preferred over `[project.optional-dependencies]` for dev tooling)
- [ ] **Python version pinned**: `requires-python` set appropriately; uv respects this
- [ ] **`uv run`**: Scripts/tasks invokable via `uv run` (e.g., `uv run pytest`, `uv run ruff check`)
- [ ] **`uv sync --frozen`**: CI uses `--frozen` to install from lock file without updating it
- [ ] **No `pip install` in CI**: All installs go through `uv sync` or `uv pip install`
- [ ] **`uv tool run`**: One-off tools (e.g., `uv tool run ruff`) used instead of installing globally
- [ ] **Build backend**: pyproject.toml uses a modern build backend (hatchling, setuptools with pyproject.toml, maturin, etc.)
- [ ] **Source distribution**: `uv build` produces correct sdist and wheel
- [ ] **Workspace features**: If monorepo, uses `[tool.uv.workspace]` for multi-package management

## GitHub Actions

- [ ] **`astral-sh/setup-uv`**: Official uv action used instead of manual curl installs
- [ ] **uv cache**: Action cache enabled via `enable-cache: true` on setup-uv
- [ ] **Matrix testing**: Tests run across multiple Python versions (at least min and max supported)
- [ ] **Separate workflows**: Lint, test, and release are separate workflows (or clearly separated jobs)
- [ ] **Concurrency control**: `concurrency` key set to cancel superseded runs on same branch
- [ ] **Timeout**: Jobs have `timeout-minutes` set to prevent hung runs
- [ ] **Permissions**: `permissions` block is minimal (not using default `write-all`)
- [ ] **Dependabot / Renovate**: Automated dependency update PRs configured
- [ ] **Release automation**: Version bumping and PyPI publish automated (trusted publishing preferred)
- [ ] **Branch protection**: Main branch requires CI pass before merge

## Linting and Formatting

- [ ] **Ruff**: Used for both linting and formatting (replaces flake8, isort, black, pyflakes, etc.)
- [ ] **Ruff rules**: Comprehensive rule selection — at minimum: `E`, `F`, `I`, `UP`, `B`, `SIM`, `RUF`
  - `UP` (pyupgrade): Catches outdated Python patterns
  - `B` (bugbear): Catches common bugs
  - `SIM` (simplify): Suggests simplifications
  - `RUF` (ruff-specific): Ruff's own rules
  - Consider also: `ANN` (annotations), `D` (docstrings), `PT` (pytest style), `TCH` (type-checking imports)
- [ ] **Ruff in CI**: Linting runs in CI and blocks merge on failure
- [ ] **Pre-commit**: Ruff (and optionally mypy) configured as pre-commit hooks
- [ ] **mypy or pyright**: Static type checking enabled and running in CI
- [ ] **Type checker strictness**: At least basic strictness; ideally `strict = true` with targeted overrides

## Release and Versioning

- [ ] **Single source of version**: Version defined in ONE place (pyproject.toml or `__version__`)
- [ ] **Semantic versioning**: Version follows semver
- [ ] **Changelog**: CHANGELOG.md maintained (manually or via automated tooling like `git-cliff`)
- [ ] **Trusted publishing**: PyPI publish uses OIDC trusted publishing (no API tokens in secrets)
- [ ] **Test before release**: Release workflow depends on test workflow passing
