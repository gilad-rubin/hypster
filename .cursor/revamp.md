### PRD: Hypster execution, return semantics, naming, locals collection, and registry

### Feature 1: Config return semantics (return statements and return types)
- Before
  - No `return` statement; Hypster executed a function body via AST manipulation and returned filtered locals as a dict.
  - Return type was not declared; IDE type suggestions for outputs were limited.

- Motivation
  - Enable typed returns for better IDE completions.
  - Allow returning a single object or a typed container when appropriate.
  - Reduce magic and make outputs explicit and predictable.

- After
  - Configs are plain functions: the only requirement is the first parameter is `hp: HP`.
  - Configs must include an explicit `return`.
  - Allowed return forms:
    - Dict[Any, Any]: returned as-is (no runtime filtering).
    - Single object (e.g., `RandomForestClassifier`): returned as-is; full IDE completions based on annotation.
    - Typed containers (dataclass, Pydantic model, NamedTuple): returned as-is; full IDE completions.
  - Return type annotations are recommended (not strictly enforced) to improve IDE assistance and readability.

  - Runtime filtering removed:
    - What the function returns is exactly what callers receive.
  - Recommended patterns for IDE DX:
    - Single object returns for simple use-cases (best autocompletion).
    - Typed containers (dataclass, Pydantic, NamedTuple) for multiple outputs with dot access.
    - Protocol-based return types to enforce shared APIs across registry variants.

- How to test it
  - Unit tests:
    - Returning dict: verify `values` overrides are applied via explicit HP names; no runtime filtering occurs.
    - Returning single object: assert type and methods; HP calls still recorded in history.
    - Returning dataclass/Pydantic/NamedTuple: assert typed access.
    - Absence of return (legacy): raises clear error with guidance to add `return`.
  - Integration:
    - `hp.nest` returns the nested object/dict pass-through.

---

### Core API: instantiate and hp.nest (args/kwargs)
- instantiate
  - Signature: `instantiate(func, *, values: dict | None = None, args: tuple = (), kwargs: dict | None = None) -> Any`
  - Behavior: builds `HP` with provided `values`, then calls `func(hp, *args, **kwargs)`; returns result as-is.
- hp.nest
  - Signature: `hp.nest(child: ConfigFunc, *, name: str, values: dict | None = None, args: tuple = (), kwargs: dict | None = None) -> Any`
  - Behavior: internally forwards to `instantiate(child, values=..., args=..., kwargs=...)` within the parent, using `name` as the dot prefix for overrides.
- Notes
  - Only `hp`-named parameters are overrideable via `values`. Extras come via args/kwargs and are not part of overrides.

---

### Engineering quality: typing, docs, linting, CI
- Goal
  - Ship production-grade code that passes strict linting and typing with comprehensive docs.

- Requirements
  - Typing
    - Add precise type annotations across the codebase; consider `from __future__ import annotations`.
  - Docstrings
    - Google-style or NumPy-style docstrings for all public functions/classes/methods, with `Args`, `Returns`, `Raises`.
  - Linting/formatting
    - Tools: ruff (lint + isort), black (format), mypy (strict), pydocstyle (docstrings), codespell (optional).
    - Configure pyproject.toml for strict rules; enable ruff’s flake8-bugbear, flake8-annotations.
  - CI
    - Workflow: uv install; ruff check; black --check; mypy; pydocstyle; pytest.
  - Validation & error handling
    - Replace Pydantic with manual validators for HP calls (ints/floats/bools/options/bounds) and raise `HPCallError` with friendly messages.
    - Provide intelligent errors: e.g. unknown keys in `values` trigger an error that suggests close matches above a similarity threshold.
    - human-readable Logging with default being silent, and enabling verbosity at different levels.

- How to test it
  - Run linters/formatters locally; CI must pass on PRs.

---

### Feature 2: Locals collection helper
- Before
  - Internals captured locals implicitly (via exec). No first-class helper existed to collect variables explicitly.

- Motivation
  - Provide a first-class, ergonomic way to collect variables without noise.
  - Support selective collection when only a subset of variables is desired.

- After
  - Add `hp.collect(locals_dict, include=None, exclude=None) -> dict`.
    - Sanitizes: removes `hp`, dunder/private names, modules, functions, classes.
    - Optional `include=[...]` and `exclude=[...]` to control scope.

- How to test it
  - Unit tests:
    - Full collection: `hp.collect(locals())` excludes disallowed entries; includes plain variables and results of HP calls.
    - Include-only: only named variables returned.
    - Exclude-only: everything except listed names returned.
    - Interop: mixing HP-named values and plain Python variables works as expected.
  - Docs tests:
    - Minimal examples using `hp.collect` in place of `export_locals`.

---

### Feature 3: Remove automatic naming; run original function; allow external imports
- Before
  - AST-based automatic naming inferred names from assignment, dict keys, and kwargs and injected `name=...` into HP calls.
  - Function body was executed via `exec` after AST manipulation.
  - Imports were recommended inside the function to ensure portability.

- Motivation
  - Reduce magic and potential surprises from AST mutation.
  - Make configurations explicit and easier to reason about.
  - Allow normal Python module patterns (imports/objects defined outside).
  - Enable standard debugging of the function itself (set breakpoints, step through code, inspect variables).

- After
  - Execution
    - Run the original function directly; no AST modification.
    - Imports and objects can live outside the config function (normal Python behavior).
  - Naming
    - Automatic naming is removed.
    - All HP calls that should be addressable via `values={...}` must specify `name="..."` explicitly (dotted names allowed).
    - Missing `name` on HP calls: raises a validation error when the call is executed.
  - `values` overrides
    - Only apply to explicitly named HP calls.
    - Dotted keys remain supported for hierarchical names.

- How to test it
  - Unit tests:
    - Without `name=...`, HP calls fail with a clear error.
    - With explicit names, `values` overrides work as expected.
    - External imports and global objects are accessible from configs.
  - Regression:
    - Remove reliance on auto-injection tests; update docs accordingly.

---

### Feature 4: Callable-only nesting (with args/kwargs)
- Before
  - Dynamic selection relied on a string references for a path to a file/module; nesting accepted strings (aliases, paths) and handled resolution internally.

- Motivation
  - Reduce magic by removing the registry and string resolution.

- After
  - `hp.nest(child: ConfigFunc, *, name: str, values: dict | None = None, args: tuple = (), kwargs: dict | None = None) -> Any`:
    - `ConfigFunc` contract: the first parameter must be `hp: HP`. Additional args/kwargs are allowed and forwarded.
    - `name` is required and acts as the stable prefix for values/snapshots (e.g., `retriever.dim`).
    - Returns the child’s result pass-through.
  - Overrides
    - Support dotted keys (canonical) and nested dict overrides. Precedence: nested dict entries override dotted keys on conflict.
    - Example: `{"retriever": {"dim": 128}}` and `{"retriever.dim": 256}` → effective `dim = 128`.

- How to test it
  - Unit tests:
    - Dict-based selection chooses the right callable and `hp.nest` executes it.
    - Dotted overrides using the `name` prefix affect the child configuration.
    - Passing a non-callable to `hp.nest` raises a clear error.
    - Missing `name` in `hp.nest` raises a clear error.
    - Forwarded args/kwargs reach the child callable correctly.

---

### Feature 6: No built-in save/load; users handle code organization explicitly
- Before
  - Built-in save/load APIs and file-path resolution added complexity and AST handling.

- Motivation
  - Reduce surface area and magic; keep Hypster focused on parameterization and execution.

- After
  - Remove save/load from the public API and docs.
  - Users organize code with normal Python modules/files and import functions directly as needed.
  - `hp.nest` does not accept strings/paths; callers pass callables (resolved via imports or dicts).

- How to test it
  - Documentation examples import functions normally and pass them to `hp.nest`.
  - Ensure no references to save/load or file-path resolution remain in tests or docs.

---


### Feature 7: Float-only HP call (hp.float)
- Motivation
  - Provide a strictly float-only parameter type where `number` is too permissive.

- After
  - Add `hp.float(default: float, *, name: Optional[str] = None, min: Optional[float] = None, max: Optional[float] = None) -> float`.
  - Behavior mirrors `hp.number` but rejects ints (type must be float), enforces optional bounds, and integrates with history/snapshots.

- How to test it
  - Defaults: accepts float defaults; raises on int defaults.
  - Overrides: `values` must provide floats; ints raise with a clear error.
  - Bounds: min/max enforced with friendly messages.

---

### Backward compatibility and migration
- Breaking changes
  - Configs without a `return` must be updated to return a dict/object.
  - `final_vars` and `exclude_vars` parameters are removed from the public API.
  - Auto-naming removed; all HP calls must declare `name="..."` if they need to be overridden via `values`.
  - Docs requiring imports inside function are relaxed; imports can move to module scope.

### Acceptance criteria
- Config functions execute directly (no AST rewriting) and must return a value.
- No runtime filtering: returned values are passed through as-is (dicts, objects, or typed containers).
- `hp.collect` sanitizes `locals()` and supports include/exclude.
- Auto-naming is removed; missing `name` on HP calls surfaces a clear error.
- `hp.nest` accepts only callables and uses `name` as the stable prefix.
- Dynamic selection via dicts works and supports consistent return types across variants.
- `hp.float` exists and enforces float-only semantics and bounds.
- Documentation updated to reflect new return semantics, explicit naming, callable-only nesting, and hp.float.

---

### Test plan (high level)
- Return semantics
  - Dict/Single/Dataclass/Pydantic/NamedTuple returns.
  - No runtime filtering; verify pass-through behavior.
- Naming/values
  - Missing `name` errors.
  - `values` overrides apply to explicitly named calls; dotted names honored.
- Nesting
  - Callable-only `hp.nest` with required `name`; dynamic selection with dicts; args/kwargs forwarded to child callables.
  - Error handling for non-callable child and missing `name`.
- Float-only call
  - `hp.float` default/override/bounds behavior and error messages.
- External imports/objects
  - Global imports/objects referenced in configs are supported.

---

### Describe + Ask/Tell + Interactive UI
- Describe
  - `describe(func, *, args=(), kwargs=None) -> dict` executes `func` with a recording HP to build a schema of parameters (name, type, default, options, bounds). This schema can be used to drive HPO (e.g., Optuna distributions) or UIs.
- Ask/Tell
  - Convert `describe` schema to distributions (e.g., Optuna) and use ask/tell to sample values, then call `instantiate(func, values, args, kwargs)`; report objective back.
- Interactive UI
  - Use `describe` to build UI components (select/int/float/text/bool) and update values on user interaction; re-run `instantiate` with updated values to reflect changes.
- Notes
  - Core remains stateless; history and UI state live externally.
