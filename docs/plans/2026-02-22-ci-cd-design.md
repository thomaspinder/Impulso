# CI/CD Pipeline Design

## Goal

Tighten the CI/CD pipeline for Litterman: deploy docs on every push to main, simplify the release flow, automate dependency updates, and protect the main branch.

## Changes

### 1. Docs Deployment

Add a `deploy-docs` job to `.github/workflows/main.yml` that runs only on push to main (not on PRs), after all checks pass. Uses `mkdocs gh-deploy --force` to push to the `gh-pages` branch. Requires `contents: write` permission.

Remove the `deploy-docs` job from `on-release-main.yml` since docs now deploy on every main push.

Enable GitHub Pages via `gh api` targeting the `gh-pages` branch.

### 2. Release Script

Add `scripts/release.py` following the cbspy pattern:

- Accepts `patch`, `minor`, or `major` as argument
- Checks for clean working tree
- Runs `uv version --bump <type>`
- Commits `pyproject.toml` and `uv.lock`
- Tags with `v<version>`
- Pushes commit and tag
- Creates GitHub release with `gh release create --generate-notes`

### 3. Simplified Release Workflow

Rewrite `.github/workflows/on-release-main.yml`:

- Remove the `set-version` job and artifact passing (version already correct in committed pyproject.toml)
- Single `publish` job: checkout, setup env, build with `uv build`, publish with `uv publish`

### 4. Dependabot

Add `.github/dependabot.yml` with weekly updates for:

- `pip` (Python dependencies)
- `github-actions` (action version bumps)

### 5. Branch Protection

Configure via `gh api`:

- Require status checks to pass: `quality`, `tests-and-type-check`, `check-docs`
- No review requirement (solo dev)
- No force pushes to main
- Allow branch deletion after merge

## Implementation

All changes go on a `ci-improvements` branch off `main`, merged via PR.
