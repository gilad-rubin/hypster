# Feedback from the Superposition stack review (2026-06-10)

Findings from using hypster as the config layer for Panda/Subtext/Superposition, reviewed from an agent-builder perspective. Companions: `hypergraph/dev/feedback/`, `hypercache/dev/feedback/` (same date).

## Issues / gaps

### 1. No params migration story (the biggest real-world gap)

When a config space evolves (param renamed, branch removed, nesting moved), saved params/presets go stale. `on_unknown="raise"` is correct but blunt for this case, and `"ignore"` silently drops. Superposition had to hand-roll `migrate_retrieval_preset()` (filter stale `retriever.*`/`reranker.*` keys after a schema change) — every serious user of the replay story will hit this.

Suggested: a small `hypster.migrate` helper — declarative rename/drop maps applied to a params dict, plus an `on_unknown="report"` mode that returns which keys were dropped/unreachable instead of raising or silently ignoring. Even just a documented pattern page ("evolving config spaces without breaking replays") would help.

**Update (2026-06-10, after design review — supersedes the above suggestion):** drop `hypster.migrate`; migration maps are the wrong shape (Gilad's call: no migration machinery). The refined ask is one additive primitive: `on_invalid: "raise" | "default"` mirroring `on_unknown`, on `instantiate`/`instantiate_with_params`/`explore`. When a *provided* value fails validation (removed option, out-of-bounds), fall back to that call's default, keep executing (top-to-bottom execution means later values are unaffected), and return an `issues` list (`{path, provided, reason}`) on the output — with unknown keys reported there as data too, reusing the existing typo hints. One execution applies every valid value, defaults only the stale ones, and reports honestly; hosts (Superposition presets) build warning UX on top. Related footgun found while verifying: with `options_only=False` (the default), a stale option key on a dict-backed select passes through **silently** as a raw string into downstream code — `on_invalid` should emit at least a note-level issue for that case.

### 2. Option-level maturity metadata for tiered spaces

Real projects keep "experimental" options alive for HPO exploration (try Gemini next month when the new model lands) alongside "supported" options used in production. Today there's no way to tag a select option's status, so downstream UIs/HPO can't distinguish tiers, and the only alternatives are deleting options (losing the space) or letting dead options look production-ready.

Suggested: optional per-option metadata on `hp.select`/`hp.multi_select` (e.g. `option_meta={"gemini": {"status": "experimental"}}` or richer option objects), surfaced in `ConfigSchema` so hosts can render/filter tiers and optimizers can opt in/out of experimental branches.

**Update (2026-06-10): resolved by release.** PR #58 (`2d899c3`, on master) added `metadata={...}` to every hp call, surfaced in `ConfigSchema`. Per-option tiers can ride on it when actually needed (e.g. `metadata={"tiers": {"gemini-embed": "exploration"}}` on the select). No further library work; whether sp ever consumes it is deferred until the optimizer needs to filter options — production code pins values explicitly anyway, so the tier question only exists in Studio/HPO contexts.

### 3. IDE/typing surface

No `.pyi` stubs for `HP`; `hp.int/.float/...` return types and overloads aren't IDE-visible. For a library whose whole pitch is "plain Python," stub files are cheap leverage. (The dynamic-discoverability weakness mostly vanishes for agents — we run `explore()` constantly — but stubs help the human-in-IDE half.)

### 4. Values ergonomics in non-Python hosts

Flat dotted params are perfect in Python/JSON, but terrible through shell `--set key=value` once values are long Hebrew prompt strings (quoting, RTL, newlines). Not hypster's bug, but worth a doc note recommending hosts accept a values *file* (JSON) as the canonical override channel, with `--set` reserved for short scalars. (Superposition's CLI is adopting this.)

### 5. Document the async boundary explicitly

Config functions are sync-only by design (instantiation should be cheap and side-effect-free). Worth one explicit docs paragraph, since AI-stack users will try `await` inside a config within their first hour.

## What's working notably well

- Dict-backed selects (log the key, return the object) — quietly the best idea in the library; keep it front and center in positioning.
- `instantiate_with_params` with defaults baked in — exactly the right experiment-record unit; Superposition's Candidate/Trial records are built directly on it.
- Strict unknown values — saved a real preset-drift bug from silently corrupting eval comparisons.
- `interact()` snapshot model — the JSON-friendly session snapshot turns out to be agent-readable too, not just widget plumbing.
