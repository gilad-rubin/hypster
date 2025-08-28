# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5] - 2025-01-13

### Fixed
- Fixed Hatch build configuration for proper package metadata generation
- Resolved CI test failures due to import errors
- Enhanced release workflow with better error handling and package verification

### Improved
- Updated GitHub Actions workflows for more reliable CI/CD
- Added optuna dependency to CI for comprehensive HPO testing
- Enhanced codecov badge configuration and removed deprecated GitBook badge

## [0.3.4] - 2025-08-27

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
