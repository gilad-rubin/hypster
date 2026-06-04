# Direct Keyword Execution Arguments

Hypster execution APIs should forward ordinary config dependencies as direct keyword **Execution Arguments** instead of teaching separate `args=` and `kwargs=` containers. This is a breaking greenfield cleanup: Hypster-owned controls such as `values`, `on_unknown`, `return_schema`, `auto_apply`, `name`, and `description` remain reserved at their execution boundary, while all other direct keywords are forwarded into the **Configuration Function** and are never included in **Selected Params**.

## Consequences

- `instantiate`, `instantiate_with_params`, `explore`, `interact`, `hp.nest`, and HPO `suggest_values` use the same direct-keyword forwarding model.
- Public `args=` and `kwargs=` forwarding are removed rather than kept as compatibility aliases.
- `explore(return_info=True)` is renamed to `explore(return_schema=True)` so the flag matches the **Explore Schema** domain term.
- This change should ship as a breaking pre-1.0 minor release.
