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
    - Ensure run history records HP calls regardless of return type.
    - `hp.nest` returns the nested object/dict pass-through.

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

### Feature 4: Registry and dynamic nesting (manual registration)
- Before
  - No first-class registry; nesting supported files and direct `Hypster` objects. Automatic naming influenced nested key mapping.

- Motivation
  - Enable dynamic selection of interchangeable components (models, retrievers, LLMs) with easy alias-based references.
  - Support notebook/ad-hoc workflows with minimal setup.

- After
  - Registry API (minimal)
    - `registry.register(obj, name: str)`: registers a `Hypster` (or compatible callable) under an alias (e.g., `"retriever/tfidf"`).
    - `registry.get(name: str) -> Hypster`: retrieves a previously registered object.
    - `registry.list() -> dict[str, str]`: inspection.
  - Manual registration (chosen default for notebooks/ad-hoc)
    - Users import configs and register them explicitly:
      - registry.register(tfidf_config, name="retriever/tfidf")
      - registry.register(m2v_config, name="retriever/model2vec")
  - Nesting
    - `hp.nest(ref, ...)` resolves:
      - Hypster object: used as-is.
      - Alias string: looked up in `registry`.
      - File path: load a Python module that defines and binds a Hypster object (via @config) and import it by name; no AST parsing.
    - Return shape pass-through: parent receives exactly what the nested config returns (dict or object).
  - Consistency for dynamic choices
    - Recommended contract: all alternatives registered under a namespace return the same type (e.g., a `Retriever` Protocol, or a model object).

- How to test it
  - Unit tests:
    - Manual registration and retrieval of two variants; `hp.nest` resolves alias to either variant.
    - Dynamic selection: `rtype = hp.select([...], name="retriever.type")`; `hp.nest(f"retriever/{rtype}", ...)` picks the correct config.
    - Consistency: both variants return the same type; downstream code uses shared methods (e.g., `.retrieve()`/`.fit()`).
    - Error handling: requesting an unknown alias; mixing return shapes leads to clear errors at use-site.
  - Integration:
    - Notebook-like flow: register in a separate context and use in a parent config; confirm correct resolution.

---

### Feature 6: Simplified save/load (and file path nesting)
- Before
  - save/load relied on extracting a function’s source and AST parsing to find the `hp` function body and inject/import scaffolding.
  - Loading by file path executed the entire file in a bespoke namespace and searched for a config function by signature.

- Motivation
  - Reduce complexity and remove AST/source surgery.
  - Make saved modules importable and debuggable as normal Python modules.
  - Align file path nesting with import mechanics (no custom exec).

- After
  - save(hypster, path): writes a normal Python module that contains the original `@config`-decorated function so importing that module binds a `Hypster` object.
    - No AST rewriting; no function-body extraction. Keep the original function and decorator usage.
  - load(path): simplified to import the module by path (or read + exec minimal wrapper), then retrieve a bound `Hypster` object by attribute name. Prefer explicit import path usage in docs.
  - hp.nest(file_path): import the module, access the bound `Hypster` object by name; return pass-through result. No AST parsing.

- How to test it
  - Saving writes an importable module; `importlib.import_module` can import it; the expected `Hypster` object exists as a module attribute.
  - Loading by path returns the same `Hypster` object; calling it returns the expected result.
  - hp.nest with a file path resolves and executes the saved config correctly.

---


### Backward compatibility and migration
- Breaking changes
  - Configs without a `return` must be updated to return a dict/object.
  - `final_vars` and `exclude_vars` parameters are removed from the public API.
  - Auto-naming removed; all HP calls must declare `name="..."` if they need to be overridden via `values`.
  - Docs requiring imports inside function are relaxed; imports can move to module scope.

- Migration guidance
  - Add `return` at the end of each config; choose:
    - Return single object (for best IDE DX) when only one output is needed.
    - Return `hp.collect(locals())` or an explicit dict otherwise.
  - Add explicit `name=...` to HP calls you need to override via `values`.
  - For dynamic selection, register configs manually in notebooks and use alias strings with `hp.nest`.

---

### Acceptance criteria
- Config functions execute directly (no AST rewriting) and must return a value.
- No runtime filtering: returned values are passed through as-is (dicts, objects, or typed containers).
- `hp.collect` sanitizes `locals()` and supports include/exclude.
- Auto-naming is removed; missing `name` on HP calls surfaces a clear error.
- Manual registry allows alias-based retrieval and works with `hp.nest`.
- Dynamic selection via aliases works and supports consistent return types across variants.
- Documentation updated to reflect new return semantics, explicit naming, and registry usage.

---

### Test plan (high level)
- Return semantics
  - Dict/Single/Dataclass/Pydantic/NamedTuple returns.
  - No runtime filtering; verify pass-through behavior.
- Naming/values
  - Missing `name` errors.
  - `values` overrides apply to explicitly named calls; dotted names honored.
- Registry and nesting
  - Manual register/get/list; nesting via aliases; dynamic selection with hp.select; consistent API across variants.
  - Error handling for unknown aliases and mixed return shapes.
- External imports/objects
  - Global imports/objects referenced in configs are supported.

This PRD focuses on Option 2 (manual registration) for registry population, optimized for notebooks and ad‑hoc workflows. Optional future work can add AST discovery, entry points, or watch mode if desired.
