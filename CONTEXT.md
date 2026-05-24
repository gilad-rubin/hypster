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

**Parameter Description**:
Optional human-written helper text attached to a parameter for lightweight display in interactive UIs and schema metadata.
_Avoid_: docstring, tooltip-only text

**Display Label**:
A human-readable label derived from a config, parameter, or group name unless explicitly provided by schema metadata.
_Avoid_: parameter path, description

**Interactive Snapshot**:
A JSON-friendly record of the current interactive exploration state, including reachable widget metadata, current override inputs, selected params, display state, and errors.
_Avoid_: UI schema, frontend state, selected params

**Interactive Action**:
A renderer-originated command that asks the interactive controller to change session state and produce a new interactive snapshot.
_Avoid_: frontend validation, JavaScript branch logic

**Interactive Result**:
The live object returned by interactive instantiation, exposing the latest instantiation value and selected params for one interactive session.
_Avoid_: result dict, widget, instantiation value

**Apply Mode**:
The interactive session behavior that decides whether widget changes are applied immediately or staged until the user explicitly applies them.
_Avoid_: save mode, persistence mode

**Interactive Baseline**:
The initial reachable widget state for an interactive session, derived from validated seed values or configuration defaults.
_Avoid_: reset defaults, saved state

**Draft Values**:
The current reachable override inputs represented by the interactive widgets, including unapplied edits in manual apply mode.
_Avoid_: selected params, pending params

**Applied Values**:
The reachable override inputs that produced the current interactive result value and selected params.
_Avoid_: saved values, committed params

**Branch Choice Memory**:
Session-local remembered widget choices used to restore compatible values when conditional parameters become reachable again.
_Avoid_: selected params, hidden values

**Exploration Error**:
An error that prevents draft values from producing a valid explore schema.
_Avoid_: instantiation error, result error

**Instantiation Error**:
An error that occurs while applying values to produce an instantiation value and selected params.
_Avoid_: exploration error, validation error

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
- An **Explore Schema** may include a **Parameter Description** for each reachable parameter.
- Interactive UI renderers should derive **Display Labels** by humanizing names, such as rendering `top_k` as "Top K".
- An **Interactive Snapshot** is produced by an interactive controller for one current widget state.
- An **Interactive Snapshot** includes the current **Explore Schema**, **Draft Values**, **Applied Values**, current **Selected Params**, and any current exploration or instantiation error.
- When an immediate interactive state cannot instantiate, an **Interactive Snapshot** uses `selected_params=None` so renderers do not confuse stale params with the current invalid state.
- An **Interactive Snapshot** may include notebook-friendly display metadata for an **Instantiation Value**, but the raw **Instantiation Value** remains available through the returned proxy.
- Remembered branch choices in an interactive session are UI input memory; they are keyed by reachable branch context and parameter metadata rather than by dotted path alone, and they are not **Selected Params** until their parameter paths are reachable in the current **Interactive Snapshot**.
- Notebook, Streamlit, and React renderers should consume the same **Interactive Snapshot** contract instead of each inventing a parameter metadata model.
- A renderer sends **Interactive Actions** to an interactive controller; the controller applies Hypster semantics and returns a new **Interactive Snapshot**.
- **Interactive Actions** must not implement branch reachability, parameter validation, or selected-param collection in the renderer.
- The interactive controller must validate **Values** through the same `explore` and `instantiate_with_params` code paths as non-interactive execution.
- The interactive controller must not soften or duplicate backend validation for **Values**, select choices, nullability, unknown paths, or unreachable paths.
- `interact(config)` returns an **Interactive Result**.
- An **Interactive Result** can be displayed more than once, with each view attached to the same live interactive session.
- `result.interact()` renders another live interactive view attached to the same interactive session.
- A fresh interactive session seeded from a prior run must be explicit through **Values**, such as `interact(config, values=result.params)`.
- `interact(config, values=...)` follows the same reachable **Values** rules as `explore` and `instantiate`; unreachable or unknown paths are not accepted as hidden branch memory.
- Reachable `values=` entries seed both the initial widget state and the remembered branch-choice state for that interactive session.
- **Branch Choice Memory** is created only from reachable `values=` entries and subsequent user **Interactive Actions** during the live session.
- **Branch Choice Memory** is in-memory state scoped to one **Interactive Result**; it is not persisted or shared across fresh interactive sessions.
- When a parameter becomes reachable again, the interactive controller chooses the most recent compatible value from **Branch Choice Memory**, then the parameter default, then the first available option when the parameter kind supports option fallback.
- Incompatible remembered values are skipped rather than treated as errors.
- **Branch Choice Memory** applies to nested parameters by **Parameter Path**.
- Reset restores the **Interactive Baseline**, clears later branch memory and pending edits, and applies that restored state immediately.
- An **Interactive Result** exposes the latest **Instantiation Value** separately from the latest **Interactive Snapshot**.
- An **Interactive Result** exposes the latest **Selected Params** through `params`.
- `result.params` returns a caller-owned plain dictionary copy.
- `result.value` returns the current **Instantiation Value** itself, not a defensive copy.
- An **Interactive Result** is a live handle, not the raw **Instantiation Value**.
- Notebook users read the current instantiated object through `result.value`.
- Because `interact(config)` returns a live handle instead of a raw **Instantiation Value**, there is no separate `interact_params` API.
- When the current interactive state cannot instantiate, `result.value` and `result.params` raise the current error instead of returning a stale previous value.
- `interact(config)` uses immediate **Apply Mode** by default, so valid widget actions update `result.value` and `result.params` live.
- Users can pass `auto_apply=False` to use manual **Apply Mode**, where widget changes are staged until an explicit Apply action updates `result.value` and `result.params`.
- Numeric controls may use interaction latency, such as applying on slider release or debounced text commit, so dragging and typing are not interrupted by repeated rerenders.
- In manual **Apply Mode**, **Draft Values** may differ from **Applied Values**.
- In manual **Apply Mode**, `explore(config, values=draft_values)` still runs after draft widget changes so reachable widgets and options stay current.
- In manual **Apply Mode**, `result.value` and `result.params` expose the last **Applied Values** state while unapplied **Draft Values** remain pending.
- In manual **Apply Mode**, invalid **Draft Values** keep Apply disabled and do not change `result.value` or `result.params`.
- In manual **Apply Mode**, `result.value` and `result.params` raise only when the last applied state itself has an error, not merely because pending **Draft Values** are invalid.
- Instantiation happens when **Draft Values** are applied, not lazily when `result.value` or `result.params` is read.
- In immediate **Apply Mode**, valid **Draft Values** become **Applied Values** after each applied widget action.
- In immediate **Apply Mode**, invalid widget actions put the session in an applied error state; `result.value` and `result.params` raise until the widget state is fixed.
- Interactive UI renderers must show current exploration or instantiation errors clearly enough that users do not mistake stale values or params for current ones.
- Interactive UI renderers must distinguish **Exploration Errors** from **Instantiation Errors**.
- In manual **Apply Mode**, **Exploration Errors** disable Apply and leave the current **Applied Values** result unchanged.
- In manual **Apply Mode**, **Instantiation Errors** occur when Apply runs and become the current applied error state.
- `result.params` represents the latest successful applied instantiation; it raises on an **Instantiation Error** instead of returning attempted params.
- UI renderers may display attempted params for diagnostics after an **Instantiation Error**, but must label them distinctly from public **Selected Params** and last successful params.
- The primary interactive UI should not duplicate all **Selected Params** in a separate params panel by default; users inspect the logging payload through `result.params`.
- The primary interactive UI should not render an **Instantiation Value** preview; users inspect instantiated values through `result.value`.
- Nested parameters should render as nested containers in the primary interactive UI.
- **Parameter Descriptions** should render lightly in the primary interactive UI without competing with labels or controls.
- In immediate **Apply Mode**, an invalid widget action makes the applied state invalid; `result.value` and `result.params` raise until the state is fixed.
- The UI may show an "applied" or "up to date" status for immediate **Apply Mode**, but this means applied to the live interactive session, not persisted outside the notebook.

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
- "intermediate representation" was ambiguous with **Explore Schema**; resolved: **Interactive Snapshot** is the UI-agnostic session state contract, while **Explore Schema** remains the reachable parameter metadata inside it.
- The old interactive UI exposed dict-like result access; resolved: modern interactive APIs expose `.value` and `.params` on an **Interactive Result**, instead of preserving dict-proxy behavior.
- `interactive_explore` was ambiguous because the UI instantiates the configuration; resolved: the public API is `interact`.
- The split between `interact` and `interact_params` was considered; resolved: skip it because `interact` already returns an **Interactive Result** handle rather than the raw **Instantiation Value**.
- A separate Clear-to-defaults action was considered; resolved: skip it because Reset restores the **Interactive Baseline**, including validated seed values.
