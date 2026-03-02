# /pr — Create a PR summarising changes vs main

Analyse the diff between the current branch and `main`, then open a pull request using the GitHub CLI (`gh`).

## Steps

1. Run `git diff main...HEAD` to get the full diff.
2. Run `git log main..HEAD --oneline` to get the commit list.
3. Analyse the diff and commits to understand the nature of the changes.
4. Classify the change into exactly one primary category and assign the corresponding label:

| Category          | PR title prefix | GitHub label      |
|-------------------|-----------------|-------------------|
| New feature       | `feat:`         | `enhancement`     |
| Bug fix           | `fix:`          | `bug`             |
| Breaking change   | `feat!:` / `fix!:` | `breaking-change` |
| Documentation     | `docs:`         | `documentation`   |
| Refactor          | `refactor:`     | `internal`        |
| Testing           | `test:`         | `internal`        |
| Chore / CI        | `chore:`        | `internal`        |
| Performance       | `perf:`         | `performance`     |

5. Write the PR title in imperative mood with the appropriate prefix.

6. Write the PR body using this structure:

### Body
```
## Summary
2–3 sentences: what this PR does and why.

## Changes
- Key changes grouped logically (not one per file).

## Testing
How changes were tested, or note if tests are included.

## Notes
Breaking changes, migration steps, deprecations, or follow-ups.
```
7. Request feedback from me around the PR title and body before creating the PR. I may ask for revisions to ensure clarity and completeness.

7. Only once you have my approval, create the PR:
```bash
gh pr create \
  --title "<prefix>: <title>" \
  --body "<body>" \
  --label "<label>"
```

If the label doesn't exist yet, create it first with `gh label create`.

## Rules
- Do NOT fabricate changes — only describe what is in the diff.
- Always assign exactly one category label.
- If the change is breaking, mention it explicitly in the Summary and Notes sections.
- If the diff is empty, stop and inform me.
- If `gh` is not authenticated, tell me to run `gh auth login`.
