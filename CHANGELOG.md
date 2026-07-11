# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- HPO: configs using `hp.bool`, `hp.text`, or `multi_*` parameters no longer crash under `suggest_values()` with a raw `AttributeError`. Kinds without an Optuna mapping keep their (validated) defaults and are omitted from the returned values dict.
- HPO: trial-suggested values now pass the same validation as explicit values (bounds, types); previously suggestions were trusted unvalidated.
- Calling a parameter without a default (e.g. `hp.int(name="x")`) now raises a friendly `Parameter 'x': requires a default value ...` error instead of a `TypeError` naming a private method.
- The interactive widget's notebook background shim is now scoped with `:has(.hypster-widget)` so it only affects output cells that contain a Hypster widget, instead of repainting every cell in the notebook (ADR-0002).
- Setting an unreachable parameter in an interactive session now surfaces the backend's `Unknown or unreachable parameters` error (with typo suggestions) instead of a bespoke controller message; under `on_unknown="warn"`/`"ignore"` it is a policy-consistent no-op.
- `Leaf.from_dict` validates its keys and raises a clear `ValueError` on malformed rule JSON instead of a raw `KeyError`.
- `hp.rules()` and `hp.schema()` now validate user-supplied `metadata` for JSON compatibility like every other parameter kind.

### Added
- `hypster.HPCallError` is exported: parameter validation errors keep their type (still a `ValueError` subclass) instead of being re-raised as bare `ValueError`, so callers can catch Hypster errors specifically.
- `instantiate_with_params(..., tracker=)` is documented: a tracker observes `record_parameter`/`record_nest` events for every `hp.*` call. `tracker` is now a reserved execution-argument name.
- `SchemaField.required` flag (from 0.7.0 follow-up work) is documented; it round-trips via `to_dict`/`from_dict` and is intentionally not emitted by `to_json_schema()`.
- `HP` accepts an internal `value_provider` seam (the mechanism behind the Optuna adapter); `hypster.hpo` now exports `suggest_values` and `TrialValueProvider` directly.
- CI: Python 3.13 across the matrix plus macOS and Windows jobs; 3.13 classifier.

### Changed
- **Breaking:** the `hypster.integrations` namespace is removed. It was a wildcard re-export that accidentally published internals (`FloatValidator`, `normalize_values`, `typing.Mapping`, ...) as public API. Import from `hypster.hpo` / `hypster.hpo.optuna` instead.
- **Breaking:** `hp.schema()` explore metadata key renamed `field_specs` → `schema_fields`; `field_specs` remains the rules vocabulary key (CONTEXT.md reserves "Field Spec" for rule conditions).
- Internal restructure with no behavior change: `hp.py` public methods are defined directly (no `__getattr__` dispatch, ~2.8× lower per-call attribute overhead), the Spec/executor indirection layer is deleted, `explore()` builds schemas through a plain tracker, and shared execution policy lives in `_execution.py` (hp/core circular imports removed). `hp.py` shrank from 1209 to ~900 lines.

## [0.7.0] - 2026-06-21

### Added
- `hp.rules(when=fields, then=spec, name=...)` primitive for declarative WHEN/THEN rules as a configuration value.
  - `when` accepts a list of `FieldSpec` objects declaring the condition vocabulary.
  - `then` accepts a named `FieldSpec` (or list thereof) describing the payload widget — e.g. `field.text(name="prompt", multiline=True)`.
  - Multi-then support: pass a list of FieldSpecs to `then` for composite payloads (e.g. prompt + use_citations).
  - Rules are recorded as `kind="rules"` in `explore()` schema with `field_specs`, `then_specs`, and `combinators` metadata.
  - `output.value` returns typed `Rule` objects; `output.params` returns serializable dicts.
- Typed rule objects: `Rule`, `Leaf`, `Group`, `And`, `Or`, `Not` in `hypster.rules` — frozen dataclasses with `to_dict()`/`from_dict()` serialization.
- `FieldSpec` class and `field.*` constructors (`field.select`, `field.multi_select`, `field.bool`, `field.text`, `field.int`, `field.float`) in `hypster.field` — previously required `hyperrules`, now self-contained.
- `hp.text(multiline=True)` parameter — records `{"multiline": True}` in metadata for UI rendering hints.

### Changed
- `then` parameter in `hp.rules()` is required and must be an explicit named `FieldSpec` — no implicit defaults or bare strings.
- `hp.rules()` no longer requires `hyperrules` to be installed — types and field constructors are now built-in. `hyperrules` remains the reference evaluator for `matches(condition, context)`.
- Imports changed: `from hypster import Rule, Leaf, field` (previously `from hyperrules import ...`).
- Prepared the 0.7.0 release by syncing the package version constants and refreshing dependency locks/workflow actions.

## [0.6.0] - 2026-06-04

### Breaking Changes
- Replaced public `args=` and `kwargs=` forwarding with direct keyword execution arguments across `instantiate`, `instantiate_with_params`, `explore`, `interact`, `hp.nest`, and Optuna `suggest_values`.
- Renamed `explore(return_info=True)` to `explore(return_schema=True)`.
- Changed Optuna `suggest_values` to take the config function positionally: `suggest_values(trial, config, **kwargs)`.

### Added
- Documented **Execution Arguments** as keyword-only dependencies forwarded into configuration functions without being included in selected params.
- Added an ADR for the direct keyword execution-arguments API decision.

## [0.5.3] - 2026-05-24

### Changed
- Updated release guidance to include the `viz` extra in generated GitHub releases, workflow summaries, README installation examples, and GitBook installation docs.
- Corrected README and GitBook examples to emphasize the preferred runtime-object return pattern and dict-backed nested component selection.

### Fixed
- Updated lockfile-only vulnerable dependencies for Mako, pytest, and Pygments.

### Removed
- Removed production-readiness warning blocks from the README and docs.

## [0.5.2] - 2026-05-24

### Changed
- Overhauled the GitBook documentation with updated getting-started, examples, how-to, reference, integration, and reproducibility pages.
- Clarified that Hypster configs are ordinary Python functions rather than a DSL, including the define-by-run implications for `explore()`, `interact()`, HPO, and custom UIs.
- Documented the preferred typed-runtime-object return pattern across quickstart and best-practice examples.

## [0.5.1] - 2026-05-24

### Fixed
- Reject boolean values for numeric hyperparameters instead of treating `True`/`False` as integers.
- Apply non-strict float coercion consistently for top-level and nested parameter paths.
- Treat nested scope values such as `values={"child": ...}` as unknown unless they target a concrete child parameter.
- Raise on unknown explicit `hp.nest(..., values=...)` child overrides instead of silently ignoring them.
- Allow `hp.select([], allow_none=True)` to default to `None`, matching its error guidance.
- Compute `HpoInt(include_max=False)` ranges correctly when `step` does not evenly divide the interval.
- Reject unsupported Optuna HPO spec fields instead of silently ignoring them.
- Validate HPO nested override values before returning them from `suggest_values()`.
- Return helpful configuration signature errors for keyword-only `hp` parameters and callable objects.

## [0.5.0] - 2026-05-23

### Added
- `interact()` for live notebook instantiation through an `InteractiveResult` handle with `.value`, `.params`, `.snapshot`, and `.interact()`.
- A minimal anywidget renderer for interactive instantiation, using file-backed vanilla JavaScript and scoped CSS instead of React or inline HTML blobs.
- In-memory branch choice memory for interactive sessions, so branch-specific selections are restored when users switch away and back.
- `auto_apply=False` manual apply mode for staging widget edits before updating `result.value` and `result.params`.
- The `viz` extra for installing interactive notebook widget dependencies across Jupyter Notebook, JupyterLab, and VS Code notebooks.
- Optional `description=` metadata for hyperparameters and `hp.nest(...)`, surfaced in explore schemas and the notebook UI.
- Humanized display labels in explore schema metadata, such as rendering `top_k` as "Top K".

### Changed
- Interactive widget setup errors now tell users to install `hypster[viz]` when widget dependencies are missing.
- Interactive snapshots expose draft values, applied values, current status, and explicit exploration or instantiation errors for renderer-neutral UIs.

### Fixed
- The notebook renderer preserves non-string select values instead of letting HTML option values coerce everything to strings.
- Multi-select and empty multi-value widget edits now round-trip through the notebook transport correctly.
- Interactive errors no longer expose stale selected params in snapshots while `result.params` is raising.
- Branch choice memory skips incompatible remembered multi-value inputs instead of replaying invalid hidden state.
- Branch choice memory distinguishes reused parameter paths by reachable branch context and parameter metadata, so same-name branch parameters do not leak values into each other.
- VS Code notebook outputs now receive a pre-widget background shim so the anywidget renderer does not leave the host's white widget-output background visible in dark themes.
- Reset now reports exploration or instantiation errors in the snapshot instead of leaking stale values or raising out of the widget action.

## [0.4.0] - 2026-05-22

### Added
- `instantiate_with_params()` and `InstantiationOutput` for returning both the config value and replayable selected params.
- `allow_none=True` for scalar params and `select`/`multi_select` choices.
- Documentation for reproducible logging, replay, nullable params, and dict-backed select values.

### Breaking Changes
- `instantiate()`, `instantiate_with_params()`, and `explore()` now raise on unknown or unreachable `values` by default. Pass `on_unknown="warn"` or `on_unknown="ignore"` for softer behavior.
- `hp.*` and `hp.nest` names must be valid Python identifiers. Replace literal dots, spaces, and hyphens with underscores, and let Hypster compose dotted paths from nesting.
- Scalar parameters now require `allow_none=True` when `default=None` or a `None` override is valid.
- `select` and `multi_select` choices must be logging-safe scalar values. Use dictionary-backed select to map simple keys to complex runtime objects.
- Complex dictionary values in `values=` are interpreted as nested parameter paths, not as custom select values.

### Changed
- Nested dict `values` are normalized into dotted parameter paths for duplicate and unknown-value checks.
- Nested dict keys must be valid identifier segments; use top-level dotted keys to spell full parameter paths.

### Fixed
- Unknown/unreachable value errors now guide users to run `explore(config, values=...)` to inspect the active branch.
- Duplicate dotted and nested entries for the same parameter path now fail instead of silently choosing one value.
- Invalid `on_unknown` policies are rejected before user config code executes.

## [0.3.10] - 2026-03-14

### Fixed
- Preserve readable Unicode characters in `explore()` tree output across languages instead of ASCII-escaping them

## [0.3.9] - 2026-03-14

### Added
- `explore()` function for configuration introspection and schema extraction
  - Traces through a config function to capture all parameters without executing the full config
  - Returns a `ConfigSchema` with parameter types, defaults, options, and constraints
  - Supports nested configurations with dot-notation path tracking
  - Pretty-prints a tree view of the parameter structure
  - Use `return_info=True` to get a `ConfigSchema` object for programmatic access
  - `ConfigSchema.defaults()` extracts default values as a flat dict
  - `ConfigSchema.to_dict()` returns a JSON-serializable schema

### Changed
- Updated all dependencies to latest versions

## [0.3.8] - 2025-08-28

### Fixed
- Fixed linting issues in hp.py (removed unused import of `Any` from typing)
- Cleaned up import statements to comply with linting standards

### Changed
- Updated changelog to reflect actual release history and remove unreleased version entries

## [0.3.7] - 2025-08-27

Hypster Revamp: explicit, debuggable, and ready for HPO.

### Breaking Changes
- Removed `@config` decorator and AST execution model. Configs are now plain Python functions with full access to surrounding scope.
- `return` is mandatory. Every configuration function must explicitly return its outputs.
- Parameter names are no longer inferred. All `hp.*` calls intended to be overrideable must include `name="..."`.
- `hp.nest(...)` now accepts only a callable (the child configuration function).
- Removed `hp.number`. Use `hp.float(...)` for floating-point values and `hp.int(...)` for integers.
- Removed built-in `save()` and `load()` functions; manage configuration modules directly.
- Removed `final_vars` and `exclude_vars`; return only what you want to expose.
- Removed the global registry in favor of explicit composition.
- Removed `pydantic` dependency.

### Added
- Single execution entry point: `instantiate(config_func, values=..., on_unknown=...)`.
  - Controls unknown/unreachable parameters via `on_unknown={"warn" (default), "raise", "ignore"}`.
  - Supports passing extra `*args`/`**kwargs` through to your config function.
- `hp.float(...)` for floating-point parameters (see also `hp.int(...)` for integers).
- `hp.collect(mapping, include=[...])` helper to gather and filter locals for returning.
- First-class HPO integration (Optuna): location `hypster.hpo.optuna` with `suggest_values(trial, config, ...)`.
  - Typed specs in `hypster.hpo.types`: `HpoInt`, `HpoFloat`, `HpoCategorical` via `hpo_spec=`.
  - Optional extra install: `uv add 'hypster[optuna]'`.
- CI workflow (ruff, mypy, pytest with coverage)
- Docs deploy workflow for GitHub Pages
- Release workflow (PyPI via Trusted Publisher)
- PEP 561 typing support (`py.typed`, `hp.pyi`) to improve IDE hints
- Coverage configuration and README badges

### Changed
- Clear override semantics with dotted keys and nested dictionaries; nested dictionary wins when both provided.
- Composition simplified to callable-only nesting via `hp.nest(child_config_func)`.
- Codebase adheres to production standards for linting (ruff, black), typing (mypy), and docstrings.
- Packaging configuration for Hatch (wheel/sdist includes for typing files)

### Removed
- `hp.number` (replaced by `hp.float`/`hp.int`).
- `@config`, AST execution, global registry.
- `save()`/`load()`, `final_vars`, `exclude_vars`.
- External `pydantic` dependency.
- UI and Jupyter integration (temporarily removed).

### Migration
- Replace usages of `hp.number(...)` with `hp.float(...)` (for floats) or `hp.int(...)` (for integers):

```python
# Before
lr = hp.number(0.01, min=0.001, max=0.1, name="lr")

# After (float)
lr = hp.float(0.01, min=0.001, max=0.1, name="lr")
```

- Remove `@config`; convert configs to regular functions and ensure they `return` outputs.
- Add `name="..."` to every `hp.*` call that should be overrideable.
- Update `hp.nest(...)` calls to pass a callable (no paths).
- Replace old execution with `instantiate(config_func, values=..., on_unknown=...)`.
- Drop uses of `final_vars`/`exclude_vars` and built-in `save()`/`load()`.
- If using HPO, switch to `hypster.hpo.optuna.suggest_values(...)` and add typed specs via `hpo_spec=` as needed.
