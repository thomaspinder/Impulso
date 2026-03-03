# PR Review — In-Scope / Out-of-Scope Audit

You are reviewing a pull request. Your job is to audit the changes and produce a
structured review with two clearly separated sections: **In-Scope** items (things
that should be fixed in this PR before merge) and **Out-of-Scope** items (things
worth addressing but belonging in separate issues).

## 1. Gather context

- Read the project's `CLAUDE.md` (root and any directory-level files that share a
  path with changed files) to understand coding standards and review policy.
- Run `git diff origin/$BASE_REF...HEAD --name-only` to list changed files.
- Run `git diff origin/$BASE_REF...HEAD` to inspect the full diff.
- Read the PR title, description, and any linked issues for author intent.

## 2. Classify findings

### In-Scope (must be resolved in this PR)

A finding is **in-scope** if it meets ANY of the following:

- It is a bug, logic error, or correctness issue introduced or exposed by this diff.
- It violates a rule stated in `CLAUDE.md` and the violation is in a changed file/region.
- It is a security concern in changed code (injection, auth, data exposure, etc.).
- It is a missing or broken test for new/changed behaviour.
- It is a naming, typing, or API-contract inconsistency introduced by this diff.
- It would block CI or break downstream consumers.

### Out-of-Scope (separate follow-up work)

A finding is **out-of-scope** if it meets ANY of the following:

- It is pre-existing technical debt not introduced by this diff.
- It is a refactoring opportunity in unchanged code.
- It is a performance improvement unrelated to the PR's purpose.
- It is a feature idea or enhancement inspired by the changes.
- It is a documentation gap outside the scope of the changed files.
- It is a broader architectural concern that requires its own design discussion.

When in doubt, classify as **out-of-scope** — do not block PRs for tangential work.

## 3. Output format

Structure your review comment as follows. Use exactly this format so it can be
parsed by automation:

```
## PR Review

### In-Scope — Checklist

> These items should be resolved before this PR is merged.

- [ ] **[severity]** Short description of the issue
  File: `path/to/file.ext#L10-L15`
  Details: Explanation of the problem and a suggested fix.

- [ ] **[severity]** ...

### Out-of-Scope — Issue Drafts

> These items should be tracked as separate issues.

#### 1. Title for the issue
- **Labels:** `tech-debt`, `refactor` (suggest appropriate labels)
- **Context:** Why this was noticed and how it relates to the PR.
- **Problem:** Description of the problem or opportunity.
- **Suggested approach:** Concrete next steps or acceptance criteria.
- **Affected files:** `path/to/file.ext`, `path/to/other.ext`

#### 2. Title for another issue
- **Labels:** ...
- **Context:** ...
- **Problem:** ...
- **Suggested approach:** ...
- **Affected files:** ...
```

### Severity levels for in-scope items

- **P0 — Blocker**: Will cause incorrect behaviour, data loss, or security vulnerability.
- **P1 — Must-fix**: Violates project standards or will cause problems soon.
- **P2 — Should-fix**: Improves clarity, maintainability, or robustness.

### Rules

- Only flag issues you are confident about (≥ 80% confidence).
- For each in-scope item, link to the exact lines using full commit SHA URLs.
- For out-of-scope items, write enough detail that an issue can be created directly
  from the content without further context.
- If there are no in-scope findings, say so explicitly.
- If there are no out-of-scope findings, omit that section entirely.
- Do not comment on formatting or style unless it violates an explicit `CLAUDE.md` rule.

## 4. Post the review

If `--comment` argument is provided, post the review as a PR comment using:
```
gh pr comment $PR_NUMBER --body "<review>"
```

Otherwise, output the review to the terminal.
