### Tests Revamp Suite

This document defines the new test suite aligned with the PRD revamp. It replaces tests that relied on automatic naming, locals filtering, and `final_vars`/`exclude_vars`, and adds coverage for explicit naming, return semantics, registry, nesting resolution, `hp.collect`, and simplified save/load.

---

## Directory structure (proposal)

- `tests_revamp/`
  - `test_return_semantics.py`
  - `test_hp_collect.py`
  - `test_hp_calls/`
    - `test_select.py`
    - `test_multi_select.py`
    - `test_number_input.py`
    - `test_int_input.py`
    - `test_bool_input.py`
    - `test_text_input.py`
    - `test_propagation.py`
  - `test_naming_explicit.py`
  - `test_registry_and_nesting.py`
  - `test_save_load.py`
  - `test_run_history_snapshots.py`
  - `test_docs_examples.py`

Note: We can keep existing per-call tests with minor edits (explicit names), but most of `test_naming.py`, `test_exclude_vars.py`, and parts of `test_snapshots.py` will be replaced.

---

## 1) test_return_semantics.py

- Purpose: Ensure pass-through returns and typed-return patterns work.

- Cases
  - Return dict (basic)
    - Given a config that returns `hp.collect(locals())` with explicit HP names
    - When called with/without `values`
    - Then the returned value equals the dict that the function returns; overrides applied
  - Return single object
    - Given a config that returns a model object
    - Then the result is that object; methods exist (e.g., `fit`/`predict` if sklearn available, else a mock class)
  - Return dataclass / NamedTuple / Pydantic
    - Given typed containers with multiple fields
    - Then dot-access works and values match
  - No return
    - Given a config without a return
    - Then calling it raises a clear error guiding to add `return`

---

## 2) test_hp_collect.py

- Purpose: Validate the locals collection helper.

- Cases
  - Full collection
    - Given locals with hp, private, callables, modules, classes
    - Then `hp.collect(locals())` excludes noise and keeps data objects
  - Include-only
    - Given include=["a", "b"]
    - Then only those keys returned
  - Exclude-only
    - Given exclude=["tmp", "_temp"]
    - Then those keys are removed

---

## 3) tests in test_hp_calls/ (minimal edits)

- Purpose: Keep strong coverage of call types and validation.

- Edits
  - Add explicit `name="..."` to all HP calls inside configs
  - Ensure `values` overrides reference those names (including dotted forms for hierarchical use)

- Additional Cases
  - Validation errors (options_only; numeric bounds)
  - Multi-select reproducibility flags preserved in run history

---

## 4) test_naming_explicit.py

- Purpose: Replace auto-naming tests with explicit naming behavior and error pathways.

- Cases
  - Explicit names at assignment, dict keys, and kwargs
    - e.g., `hp.number(..., name="cfg.learning_rate")`, `hp.select(..., name="model_type")`
  - Dotted names resolution
    - Override `values={"cfg.learning_rate": 0.01}`
  - Missing `name` raises clear error
    - A config with `hp.select([...])` (no name) should raise at call time

---

## 5) test_registry_and_nesting.py

- Purpose: Manual registry workflow and enhanced `hp.nest` resolution.

- Cases
  - Manual registration
    - Register two configs: `retriever/tfidf`, `retriever/model2vec`
    - `registry.list()` reflects both; `registry.get()` returns objects
  - Nest by alias
    - `rtype = hp.select(["tfidf", "model2vec"], name="retriever.type")`
    - `child = hp.nest(f"retriever/{rtype}", name="retriever")`
    - Both variants return the same type/shape (recommended contract), and are usable downstream
  - Nest by import path
    - `hp.nest("my_pkg.mod:child", name="child")` resolves and executes
  - Nest by file path
    - Save a config to file, then `hp.nest(path, name="child")` imports and executes
  - Error cases
    - Unknown alias
    - File path with multiple bound `Hypster` objects without specifying which one

---

## 6) test_save_load.py

- Purpose: Simplified save/load round-trip via importable modules.

- Cases
  - Save a config and re-import
    - `save(conf, path)` writes a module that, when imported, exposes a bound `Hypster`
    - `load(path)` returns that `Hypster`
    - Calling it returns expected result
  - Prefer import paths in docs, but ensure file path load works

---

## 7) test_run_history_snapshots.py

- Purpose: Ensure run history still captures HP parameter values and nested history works.

- Cases
  - Basic snapshot
    - After running, `get_last_snapshot()` returns flat map of HP names → values
  - Nested snapshot
    - Child config’s HP values appear as `child_name.param` keys (based on `hp.nest(..., name="child_name")`)
  - Multi-select snapshot
    - Stores list values and reproducibility flags (implicitly validated via acceptance of values on next run)
  - Conditional branches
    - Snapshot reflects parameters used in each branch
  - Snapshot replay
    - Using a previous snapshot as `values` reproduces the same result

---

## 8) test_docs_examples.py

- Purpose: Validate key snippets in the docs still execute under the new semantics (smoke tests).

- Cases
  - Minimal configs returning dict via `hp.collect`
  - Single-object return example
  - Registry + alias nesting example

---

## Mapping from existing tests

- Remove/replace
  - `tests/test_naming.py` → superseded by `test_naming_explicit.py`
  - `tests/test_exclude_vars.py` → removed; behavior now achieved by return shape or `hp.collect(include/exclude)`
  - Parts of `tests/test_snapshots.py` using `final_vars` → move to `test_run_history_snapshots.py` without final/exclude vars

- Keep with edits
  - `tests/test_hp_calls/*` → keep structure, add explicit names, update `values` accordingly
  - `tests/test_docs.py` → fold into `test_docs_examples.py` (or keep if notebook tests are required)

---

## Test utilities

- Helper classes for single-object returns (if sklearn not installed) to simulate `fit`/`predict` API
- Temporary registry setup/teardown utilities: `registry.clear()` if we add it; else register unique aliases per test
- Temp directories for save/load tests; clean up after tests

---

## Execution guidance

- Use `uv run -m pytest tests_revamp -q`
- For notebook or docs tests, gate with markers if needed (optional dependency)

---

## Acceptance

- All new tests reflect the updated semantics (explicit names, pass-through returns, no final/exclude vars)
- Registry, nesting, and save/load work end-to-end
- HP call validations remain intact (bounds, options-only, type checks)
- Run history snapshots remain usable for replay
