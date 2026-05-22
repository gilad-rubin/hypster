# Hypster

Hypster is a Python configuration context for defining parameterized config functions, instantiating them into arbitrary Python values, and recording the parameter choices that make a run reproducible.

## Language

**Configuration Function**:
A Python function that receives `hp` and returns the configured object or data for a run.
_Avoid_: config space when referring to the executable function itself

**Values**:
User-provided parameter overrides passed into a configuration function.
_Avoid_: params, selected params

**Selected Params**:
The complete, flat, dotted-path record of reachable `hp` parameter selections for one run, including defaults and using logging-friendly values that can be replayed through `values`.
_Avoid_: input values, return value, logged objects

**Select Choice**:
A logging-friendly scalar value chosen directly from list-backed select options or used as the key for dictionary-backed select options.
_Avoid_: mapped select value, returned option value

**Explicit None**:
`None` used intentionally as a selected parameter value.
_Avoid_: missing default, not selected

**Nullable Parameter**:
A parameter whose domain explicitly includes **Explicit None** through `allow_none=True`.
_Avoid_: optional when it means an omitted argument

**Logging-Safe Value**:
A value that can be logged and replayed directly: `None`, `bool`, `int`, `float`, `str`, or a list containing only those types.
_Avoid_: arbitrary object, tuple, nested dict

**Parameter Path**:
A dotted name formed by Hypster from nested scopes and Python-identifier parameter names.
_Avoid_: literal dots or path syntax inside parameter names

**Instantiation Value**:
The arbitrary Python value returned by a configuration function.
_Avoid_: values, params

**Instantiation Output**:
The result of an execution that exposes both the **Instantiation Value** and its **Selected Params**.
_Avoid_: replacing the raw return value of `instantiate`

**Explore Schema**:
Structured metadata describing the reachable parameters, defaults, bounds, options, and nesting for a configuration function branch.
_Avoid_: selected params

## Relationships

- A **Configuration Function** can be executed with **Values**.
- A **Configuration Function** produces one **Instantiation Value** per execution.
- `instantiate` returns only the **Instantiation Value**; `instantiate_with_params` returns an **Instantiation Output**.
- An **Instantiation Output** is a lightweight container with `value` and `params` attributes; `params` is a caller-owned plain dictionary copy.
- `instantiate_with_params` accepts the same execution arguments and unknown-parameter policy as `instantiate`.
- There is no params-only public API; selected params are exposed through `instantiate_with_params`.
- Flat dotted **Selected Params** are the canonical replay/logging shape; no nested params helper is included in the first API.
- **Selected Params** are derived from the reachable parameter calls in one execution, not from only the provided **Values**.
- **Selected Params** exclude parameters from branches that were not reached in that execution.
- **Selected Params** include only leaf `hp` parameter calls; `hp.nest` group names are namespaces, not selected parameters.
- **Selected Params** use the same dotted names as **Values** so they can be replayed.
- **Selected Params** are exposed as a plain flat dictionary.
- **Selected Params** are collected from the same parameter-recording event that powers the **Explore Schema**.
- Replaying an **Instantiation Output** through `instantiate(config, values=output.params)` must follow the same reachable parameter choices, but does not guarantee object identity or deterministic side effects outside the selected parameters.
- HPO-generated **Values** use the same **Parameter Path** and **Select Choice** rules as execution-generated **Selected Params**.
- Individual `hp` parameter names and `hp.nest` names must be valid Python identifiers and cannot be Python keywords.
- **Values** keys must be valid **Parameter Paths**, where each dotted segment or nested-dict segment is a valid Python identifier and not a Python keyword.
- `instantiate`, `instantiate_with_params`, and `explore` all use `on_unknown="raise"` by default.
- Nested-dict **Values** participate in unknown/unreachable checking as their equivalent dotted **Parameter Paths**.
- The `on_unknown` policy applies consistently after nested-dict **Values** are interpreted as dotted **Parameter Paths**.
- **Values** must not specify the same **Parameter Path** through both dotted keys and nested dictionaries, even when both entries have the same value; duplicates always raise as malformed input and are not controlled by `on_unknown`.
- Unknown/unreachable **Values** errors should guide users to `explore(config, values=...)` to inspect the active branch, but `instantiate` must not run extra branch exploration automatically.
- List-backed `select` and `multi_select` parameters record their **Select Choices** in **Selected Params**.
- Dictionary-backed `select` and `multi_select` parameters record **Select Choices** in **Selected Params**, even when the **Instantiation Value** uses mapped primitive or complex objects.
- **Select Choices** must be logging-safe scalar values at parameter execution time, regardless of whether **Selected Params** are requested.
- `options_only=False` permits custom **Select Choices** outside the declared options, but those choices must still be **Logging-Safe Values** and may only be `None` for a **Nullable Parameter**.
- Dictionary-backed `select` and `multi_select` should be used when a logging-safe **Select Choice** needs to produce a complex mapped object in the **Instantiation Value**.
- **Explicit None** is distinct from the absence of a default and requires an internal sentinel when an API parameter may omit `default`.
- **Explicit None** is valid only for a **Nullable Parameter**.
- `allow_none=True` is required when `default=None`, when a `None` override is accepted, and when `None` appears among `select` or `multi_select` choices.
- `allow_none=True` applies to scalar parameters and select choices, including `multi_select`; nullable elements for `multi_int`, `multi_float`, `multi_text`, and `multi_bool` are not supported yet and must fail with clear guidance.
- **Selected Params** contain only **Logging-Safe Values**.
- An **Explore Schema** describes parameter metadata; it is not the run's **Selected Params**.

## Documentation requirements

- Show `instantiate_with_params` for logging all selected params, including defaults.
- Show replaying `result.params` through `instantiate(..., values=result.params)`.
- Show dictionary-backed `select` with logging-safe choices mapped to complex values.
- Show `allow_none=True` examples for scalar params and select choices.
- Document that `allow_none=True` supports `multi_select` choices but not nullable elements for `multi_int`, `multi_float`, `multi_text`, or `multi_bool`; include the error guidance.
- Show that `instantiate_with_params` accepts the same execution arguments as `instantiate`, including `args`, `kwargs`, and `on_unknown`.
- Replace the old "nested dict wins" docs with guidance that dotted and nested forms are both supported, but duplicate **Parameter Paths** fail.
- Document that unknown/unreachable errors are based on the current execution path and that users should run `explore(config, values=...)` to inspect branches.

## Example dialogue

> **Dev:** "If the user only overrides `provider`, do we log only that value?"
> **Domain expert:** "No. **Selected Params** must include defaults too, so the run can be reproduced and logged to tools like MLflow."

## Flagged ambiguities

- "params" was used to mean both user-provided **Values** and run-derived **Selected Params**; resolved: **Values** are inputs, **Selected Params** are the replayable output sidecar.
- `select` was ambiguous between logging the returned mapped value and logging the user-facing choice; resolved: **Selected Params** record the **Select Choice** because it is replayable and logging-friendly.
- `None` was ambiguous between **Explicit None** and missing default; resolved: these must be separate states internally.
- Nullability was ambiguous when inferred from defaults or select options; resolved: `allow_none=True` explicitly marks a **Nullable Parameter**.
- The result shape was ambiguous between changing `instantiate` and adding a sibling API; resolved: `instantiate` keeps returning the **Instantiation Value**, while `instantiate_with_params` returns an **Instantiation Output**.
- A params-only shortcut was considered; resolved: skip it until there is a separate execution mode that can avoid constructing the **Instantiation Value**.
- A nested params helper was considered; resolved: skip it because flat dotted **Selected Params** are the canonical replay/logging shape.
- Literal dotted names and other path-like names were ambiguous with nested paths; resolved: individual names must be Python identifiers, while Hypster generates dotted **Parameter Paths**.
