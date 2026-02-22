# CI/CD Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tighten the CI/CD pipeline with docs deployment on every push to main, a local release script, dependabot, and branch protection.

**Architecture:** All changes on a `ci-improvements` branch off `main`. No test code to write — this is pure infrastructure. Each task is a self-contained commit.

**Tech Stack:** GitHub Actions, GitHub Pages, GitHub CLI (`gh`), MkDocs, uv

---

### Task 1: Create branch

**Step 1: Create and switch to ci-improvements branch**

Run:
```bash
git checkout main
git pull origin main
git checkout -b ci-improvements
```

Expected: On `ci-improvements` branch, up to date with `main`.

---

### Task 2: Add docs deployment to main workflow

**Files:**
- Modify: `.github/workflows/main.yml`

**Step 1: Add deploy-docs job to main.yml**

Append this job after the existing `check-docs` job in `.github/workflows/main.yml`:

```yaml
  deploy-docs:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: [quality, tests-and-type-check, check-docs]
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Check out
        uses: actions/checkout@v4

      - name: Set up the environment
        uses: ./.github/actions/setup-python-env

      - name: Deploy documentation
        run: uv run mkdocs gh-deploy --force
```

**Step 2: Validate YAML syntax**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/main.yml'))"
```

Expected: No output (valid YAML).

**Step 3: Commit**

```bash
git add .github/workflows/main.yml
git commit -m "ci: deploy docs to GitHub Pages on push to main"
```

---

### Task 3: Simplify release workflow

**Files:**
- Modify: `.github/workflows/on-release-main.yml`

**Step 1: Replace the entire file with the simplified workflow**

Write this content to `.github/workflows/on-release-main.yml`:

```yaml
name: release-main

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Check out
        uses: actions/checkout@v4

      - name: Set up the environment
        uses: ./.github/actions/setup-python-env

      - name: Build package
        run: uv build

      - name: Publish package
        run: uv publish
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}
```

This removes the `set-version` job (no more sed version bump), the artifact upload/download dance, and the `deploy-docs` job (now in `main.yml`).

**Step 2: Validate YAML syntax**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/on-release-main.yml'))"
```

Expected: No output (valid YAML).

**Step 3: Commit**

```bash
git add .github/workflows/on-release-main.yml
git commit -m "ci: simplify release workflow to build+publish only"
```

---

### Task 4: Add release script

**Files:**
- Create: `scripts/release.py`

**Step 1: Create the scripts directory and release.py**

Write this content to `scripts/release.py`:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Bump version, commit, tag, push, and create a GitHub release."""

import subprocess
import sys


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main() -> None:
    bump = sys.argv[1] if len(sys.argv) > 1 else "patch"
    if bump not in ("major", "minor", "patch"):
        print(f"Invalid bump type: {bump}. Use major, minor, or patch.", file=sys.stderr)
        sys.exit(1)

    # Check for clean working tree (aside from what we're about to change)
    status = run("git status --porcelain -uno")
    if status:
        print(f"Working tree is not clean:\n{status}", file=sys.stderr)
        sys.exit(1)

    run(f"uv version --bump {bump}")
    version = run("uv version").split()[-1]
    tag = f"v{version}"

    print(f"Releasing {tag}")

    run("git add pyproject.toml uv.lock")
    run(f'git commit -m "Bump version to {version}"')
    run(f"git tag {tag}")
    run("git push && git push --tags")
    run(f"gh release create {tag} --generate-notes")

    print(f"Released {tag}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify the script parses**

Run:
```bash
python -c "import ast; ast.parse(open('scripts/release.py').read()); print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/release.py
git commit -m "ci: add local release script (uv version --bump + gh release)"
```

---

### Task 5: Add Dependabot configuration

**Files:**
- Create: `.github/dependabot.yml`

**Step 1: Create dependabot.yml**

Write this content to `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

**Step 2: Validate YAML syntax**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))"
```

Expected: No output (valid YAML).

**Step 3: Commit**

```bash
git add .github/dependabot.yml
git commit -m "ci: add dependabot for pip and github-actions updates"
```

---

### Task 6: Run pre-commit and push

**Step 1: Run pre-commit on all files**

Run:
```bash
uv run pre-commit run --all-files
```

Expected: All checks pass. If any auto-fix, stage the changes and amend the last commit.

**Step 2: Push the branch**

Run:
```bash
git push -u origin ci-improvements
```

---

### Task 7: Create PR and wait for CI

**Step 1: Create the PR**

Run:
```bash
gh pr create --title "ci: tighten CI/CD pipeline" --body "$(cat <<'EOF'
## Summary

- Deploy docs to GitHub Pages on every push to main (not just on release)
- Simplify release workflow to build+publish only (version bump handled by local `scripts/release.py`)
- Add local release script following cbspy pattern (`uv run scripts/release.py patch|minor|major`)
- Add Dependabot for weekly pip and github-actions updates

## Test plan

- [ ] CI passes on this PR
- [ ] After merge, docs deploy to https://thomaspinder.github.io/litterman/
- [ ] `scripts/release.py` parses without errors

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 2: Wait for CI to pass**

Run:
```bash
gh pr checks <PR_NUMBER> --watch
```

Expected: All checks pass.

---

### Task 8: Enable GitHub Pages and branch protection

These are CLI commands that take effect immediately (not committed to the repo).

**Step 1: Enable GitHub Pages from gh-pages branch**

Run:
```bash
gh api repos/thomaspinder/litterman/pages -X POST -f build_type=legacy -f source='{"branch":"gh-pages","path":"/"}'
```

Note: This may fail if the `gh-pages` branch doesn't exist yet (it gets created on first docs deploy after merge). If it fails, skip this step and run it after the PR is merged and the first docs deploy completes.

**Step 2: Configure branch protection on main**

Run:
```bash
gh api repos/thomaspinder/litterman/branches/main/protection -X PUT \
  -F required_status_checks='{"strict":false,"contexts":["quality","check-docs","tests-and-type-check (3.10)","tests-and-type-check (3.11)","tests-and-type-check (3.12)","tests-and-type-check (3.13)","tests-and-type-check (3.14)"]}' \
  -F enforce_admins=false \
  -F required_pull_request_reviews=null \
  -F restrictions=null \
  -F allow_force_pushes=false \
  -F allow_deletions=true
```

Expected: JSON response with the protection rules.

---

### Task 9: Merge and verify docs deployment

**Step 1: Merge the PR**

Run:
```bash
gh pr merge <PR_NUMBER> --squash --delete-branch
```

**Step 2: Wait for docs deployment**

After merge, the `deploy-docs` job in `main.yml` will run. Watch for it:

Run:
```bash
gh run list --branch main --limit 1 --json status,conclusion,name
```

Expected: The Main workflow runs and all jobs (including `deploy-docs`) succeed.

**Step 3: Enable GitHub Pages (if not done in Task 8)**

If the `gh api pages` command failed earlier because `gh-pages` didn't exist, run it now:

```bash
gh api repos/thomaspinder/litterman/pages -X POST -f build_type=legacy -f source='{"branch":"gh-pages","path":"/"}'
```

**Step 4: Verify docs are live**

Visit https://thomaspinder.github.io/litterman/ — should show the MkDocs site.
