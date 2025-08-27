### Revamp Implementation Plan

Scope: Update Hypster to execute original config functions, require explicit `return`, remove automatic naming and runtime filtering, introduce `hp.collect`, drop the registry and save/load, add `instantiate(...)`, and make nesting accept only callables resolved by the user (dicts/imports).

This plan maps PRD features to concrete codebase edits (no code yet), including file/function targets, API changes, tests, and migration.

---

## 1) Core execution & return semantics

- Goal
  - Run the original config function directly; whatever it returns is passed through as-is. No AST mutation, no runtime filtering.

- Changes
  - File: `src/hypster/core.py`
    - Replace `Hypster` to wrap a callable `func(hp: HP) -> Any` instead of storing `source_code`.
    - Remove dependencies on `ast_analyzer`, `inject_names_to_source_code`, and `remove_function_signature`.
    - `Hypster.__call__(..., values: Dict[str, Any], explore_mode: bool=False) -> Any`:
      - Construct `HP` with minimal init signature (see Section 3).
      - Call `self.func(hp)` and return the result directly (dict/object/typed container).
      - Drop support for `final_vars` and `exclude_vars` in public API (see Migration for deprecation shim, if desired).
    - Remove `_execute_function` and `_process_results` logic that executed/filtered locals via `exec`.
    - Keep `run_history` behavior: HP calls must still record to history, independent of returned object type.
  - File: `src/hypster/config.py`
    - Remove decorator usage entirely in favor of plain functions + factories.
    - Provide `instantiate(func, values, explore_mode=False, history=None)` at module level.

- Public API updates
  - Remove `Hypster` public usage; use `instantiate(func, ...)` to execute.
- Return type: pass-through as returned by the user function.

- Tests
  - New tests for:
    - Returning dict: verify values overrides from explicitly named HP calls affect computation; result equals the dict returned by the function.
    - Returning single object: type and methods accessible; no filtering performed.
    - Returning dataclass/Pydantic/NamedTuple: dot access works; no filtering performed.
    - instantiate passes args/kwargs through to the target function.

---

## 2) Remove automatic naming & AST

- Goal
  - Explicit naming only. Remove AST parsing/injection and related docs.

- Changes
  - File: `src/hypster/core.py`
    - Remove imports: `collect_hp_calls`, `inject_names_to_source_code`.
  - File: `src/hypster/ast_analyzer.py`
    - Deprecate/remove module from runtime (keep file but unused, or delete in a major bump). If kept, mark as deprecated in docstring and ensure no imports reference it.
  - File: `src/hypster/tests/test_naming.py`
    - Rewrite tests to use explicit `name="..."` in HP calls.
    - Remove tests asserting automatic naming behavior.

- Public API updates
  - `HP` calls require explicit `name` if they should be overridden via `values`.
  - Providing an HP call without `name` is allowed but not overridable; optionally raise a validation error if `values` references a missing name.

- Tests
  - Missing `name` on a call combined with a `values` override for that name should raise a clear error.

---

## 3) HP API: simplify init + add hp.collect + hp.float

- Goal
  - Simplify `HP` to the essentials; add a first-class helper to collect locals.

- Changes
  - File: `src/hypster/hp.py`
    - Update `HP.__init__` signature to:
      - `def __init__(self, values: Dict[str, Any], run_history: HistoryDatabase, run_id: UUID, explore_mode: bool = False):`
      - Remove `final_vars` and `exclude_vars` from state.
    - Add method `def collect(self, vars_dict: Dict[str, Any], include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> Dict[str, Any]` that:
      - Removes `hp`, dunder/private names, modules, functions, classes.
      - Applies include/exclude filters if provided.
    - Add `def float(self, default: float, *, name: Optional[str] = None, min: Optional[float] = None, max: Optional[float] = None) -> float`:
      - Strict float-only; ints rejected; bounds validated.
    - Keep all HP call methods (`select`, `number`, `int`, etc.) the same regarding parameters, but they must work without auto-injected names.
  - File: `src/hypster/hp_calls.py`
    - Drop Pydantic; re-implement calls with manual validation and clear error messages (HPCallError).
    - Implement `FloatInputCall` and share bounds logic with numeric calls.

- Public API updates
  - New: `hp.collect(...)` to return a sanitized dict from `locals()`.
  - `final_vars`/`exclude_vars` removed from `HP` and all call chains.

- Tests
  - Unit tests for `hp.collect` (full, include-only, exclude-only; ignores hp/dunders/callables/modules/classes).

---

## 4) Dynamic selection via dicts (no registry)

- Goal
  - Replace registry usage with plain Python dictionaries selected via HP.

- Changes
  - Remove `src/hypster/registry.py` and any exports.
  - Update docs and examples to use local dicts (e.g., `choices = {"tfidf": tfidf_conf, ...}`).

- Tests
  - Dict-based selection drives `hp.nest` with callable-only input.

---

## 5) Nesting: callable-only (no strings)

- Goal
  - Make `hp.nest` accept only config callables `(hp: HP) -> Any` and require `name` as the prefix.

- Changes
  - File: `src/hypster/hp.py`
    - `def nest(self, child: ConfigFunc, *, name: str, values: Dict[str, Any] = {}) -> Any`
    - Remove support for strings (paths/aliases) and ignore `final_vars`/`exclude_vars` entirely.
    - Use `name` as the stable prefix for nested values/snapshots.

- Tests
  - hp.nest with a callable works; with a non-callable raises TypeError; missing `name` raises ValueError.

---

## 6) Remove save/load from public API

- Goal
  - Drop built-in save/load; users manage modules/files themselves.

- Changes
  - Remove save/load exports and references from code and docs.
  - Ensure no code path depends on file-path resolution.

- Tests
  - Remove save/load tests; update docs examples to use normal imports.

---

## 7) Utils clean-up

- Goal
  - Remove utils that supported AST-based execution.

- Changes
  - File: `src/hypster/utils.py`
    - Deprecate/remove `find_hp_function_body_and_name` and `remove_function_signature` usage.
    - Keep `query_combinations` untouched.

---

## 8) Public API & __init__

- Goal
  - Export the right symbols and remove deprecated parameters.

- Changes
  - File: `src/hypster/__init__.py`
    - Expose: `config`, `HP`, `Hypster`, `save`, `load` (simplified), `registry`.
    - Remove mentions of `inject_names` anywhere in docstrings.

---

## 9) Tests update & additions

- Remove/replace tests that rely on automatic naming and runtime locals filtering.
- Add tests for:
  - New return semantics (dict/object/containers) and `values` overrides via explicit names.
  - `hp.collect` behavior.
  - Registry manual registration & listing; nesting with alias/import path/file path.
  - Simplified save/load round-trips.

---

## 10) Docs & examples

- Update `docs/` and top-level `README.md` to reflect:
  - Explicit `return` and typed returns (single object or container) with IDE DX guidance.
  - Explicit naming in HP calls; no auto-naming.
  - Using `hp.collect` for ergonomics.
  - Manual registry usage with `registry.register`/`registry.get`/`registry.list`.
  - `hp.nest` resolution rules.
  - Simplified save/load and preference for import paths.

---

## 11) Migration & deprecation strategy

- Remove public `final_vars`/`exclude_vars` from all APIs.
- Remove `@config`, `Hypster`, registry, and save/load from public exports.
- If needed, provide a temporary shim layer with clear migration errors.
- Raise clear errors on legacy patterns (no return; expecting auto-naming; using `values` for non-named parameters).

---

## 12) Incremental delivery plan

1. Introduce registry module and exports; add tests.
2. Refactor `Hypster` to wrap original function; update `config` decorator; keep `save/load` as-is temporarily.
3. Add `hp.collect`; adjust `HP` init; propagate signature changes across the codebase and tests.
4. Remove AST analyzer usages; delete/dep-create file; clean utils.
5. Simplify save/load; update docs and tests.
6. Update `hp.nest` to resolve alias/import/file paths.
7. Final docs and examples pass.

---

## 13) Acceptance (from PRD)

- Configs execute directly and must return a value.
- No runtime filtering: pass-through returns.
- `hp.collect` exists and works.
- Auto-naming removed; explicit names required for `values` overrides.
- Registry with manual registration; `registry.list()` available.
- `hp.nest` resolves alias/import/file path and returns pass-through results.
- Simplified save/load produce/import standard modules.
- Tests and docs updated accordingly.
