# Issue tracker

This repo tracks work on **GitHub Issues** (`gilad-rubin/hypster`), operated via the `gh` CLI. Skills that say "the issue tracker should have been provided to you" mean this file.

## Triage labels

- `ready-for-agent` — the issue is self-contained: an agent can pick it up without access to the conversation that produced it.
- `spec` — a published spec/PRD; the source of truth for the tickets sliced from it.
- `wayfinder:map`, `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, `wayfinder:task` — see Wayfinding operations below.

## Conventions

- One issue per ticket; acceptance criteria as checkboxes; every ticket names what blocks it.
- `master` is protected: changes land via PR (0 approvals required, CI must pass). Merged PRs close issues via `Closes #N` in the PR body.
- Do not close or edit a parent issue when publishing its child tickets.
- Milestones group release waves (e.g. "0.9.0 — experimental interactive architecture").

## Wayfinding operations

How this tracker expresses the wayfinder skill's primitives (all verified against this repo):

- **The map** is an issue labeled `wayfinder:map`. Its tickets are **native sub-issues** of the map.
- **Create a sub-issue link** (needs the child's numeric *id*, not its number):

  ```bash
  CHILD_ID=$(gh api repos/gilad-rubin/hypster/issues/<child_number> --jq .id)
  gh api -X POST repos/gilad-rubin/hypster/issues/<map_number>/sub_issues -F sub_issue_id="$CHILD_ID"
  ```

- **Blocking edges** use GitHub's native issue dependencies:

  ```bash
  BLOCKER_ID=$(gh api repos/gilad-rubin/hypster/issues/<blocker_number> --jq .id)
  gh api -X POST repos/gilad-rubin/hypster/issues/<blocked_number>/dependencies/blocked_by -F issue_id="$BLOCKER_ID"
  # read back:
  gh api repos/gilad-rubin/hypster/issues/<number>/dependencies/blocked_by --jq '.[].number'
  ```

- **The frontier** = open sub-issues of the map that are unassigned and whose `blocked_by` list contains no open issues.
- **Claiming** a ticket = assigning it to yourself (`gh issue edit <n> --add-assignee @me`) before any work.
- **Resolving** = post the answer as a comment, close the issue, append a one-line gist + link to the map's "Decisions so far".

## Specs

`/to-spec` publishes specs as issues labeled `spec` + `ready-for-agent`. Tickets sliced from a spec (`/to-tickets`) reference the spec issue in their Parent section and carry `ready-for-agent`.

## PR review bots

CodeRabbit, Greptile, and CodSpeed comment on PRs. Their findings are triaged like any review: confirm against the code, fix or rebut with evidence, and reply on the thread. If a repo requires resolved conversations to merge, resolve threads only after the finding is genuinely addressed (GraphQL `resolveReviewThread` mutation).
