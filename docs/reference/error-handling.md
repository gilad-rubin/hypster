# Error Handling

Hypster validates names, values, paths, and branch reachability before it treats a run as replayable.

## Public Exception Surface

Public validation failures raise `ValueError`. Failures inside an `hp.*` call — missing default, type mismatch, bounds violation, `None` handling, select/rules/schema validation — raise `hypster.HPCallError`, a `ValueError` subclass whose message always starts with `Parameter '<path>':`. Catch `ValueError` at the boundary for full coverage, or `HPCallError` when you only want parameter-level failures:

{% code overflow="wrap" %}
```python
from hypster import HP, HPCallError, instantiate

def config(hp: HP):
    return hp.int(5, name="n", max=3)

try:
    instantiate(config)
except HPCallError as error:
    print(error)  # Parameter 'n': value 5 exceeds maximum bound 3
```
{% endcode %}

When `on_unknown="warn"`, unknown or unreachable values emit a `UserWarning` and the run continues.

## Unknown Or Unreachable Values

`instantiate()`, `instantiate_with_params()`, and `explore()` default to `on_unknown="raise"`.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate

def config(hp: HP):
    mode = hp.select(["a", "b"], name="mode", default="a")
    if mode == "a":
        return {"count": hp.int(1, name="count")}
    return {"mode": mode}

instantiate(config, values={"mode": "b", "count": 3})
```
{% endcode %}

`count` is unreachable on the `mode="b"` branch, so Hypster raises.

Example message:

{% code overflow="wrap" %}
```text
Unknown or unreachable parameters:
  - 'count': Unknown parameter

Run explore(config, values=...) to inspect the active branch.
Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects.
```
{% endcode %}

Unknown and unreachable values share the same public error family because both mean "this key was not consumed by the active run." To diagnose the difference, run `explore(config, values=...)` for the selected branch:

* If the path is absent from that branch but present in another branch, it is unreachable or stale.
* If the path is absent from all branches, it is likely a typo or removed parameter.

A typo can include a nearest-name suggestion:

{% code overflow="wrap" %}
```python
instantiate(config, values={"mode": "a", "coutn": 3})
```
{% endcode %}

{% code overflow="wrap" %}
```text
Unknown or unreachable parameters:
  - 'coutn': Did you mean 'count'? (similarity: 80%)

Run explore(config, values=...) to inspect the active branch.
Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects.
```
{% endcode %}

Use softer policies only when intentional:

{% code overflow="wrap" %}
```python
instantiate(config, values={"mode": "b", "count": 3}, on_unknown="warn")
instantiate(config, values={"mode": "b", "count": 3}, on_unknown="ignore")
```
{% endcode %}

With `on_unknown="warn"`, use Python warning controls if you need to capture the message in a UI or test:

{% code overflow="wrap" %}
```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    instantiate(config, values={"mode": "b", "count": 3}, on_unknown="warn")

assert caught
```
{% endcode %}

For user-facing tools, a common strategy is:

{% code overflow="wrap" %}
```python
try:
    value = instantiate(config, values=form_values)
except ValueError as exc:
    show_validation_error(str(exc))
else:
    run_workflow(value)
```
{% endcode %}

Interactive widget handles expose the same validation through a different boundary: direct `instantiate()` and `explore()` failures raise `ValueError`, while reading `InteractiveResult.value` or `InteractiveResult.params` during an invalid applied state raises `RuntimeError` with the underlying validation message. Show either message to the user and keep the last valid submitted params separate from draft UI state.

## Invalid Names

Parameter and nest names must be Python identifier-style strings.

{% code overflow="wrap" %}
```python
hp.int(32, name="batch_size")  # valid
hp.int(32, name="batch-size")  # invalid
```
{% endcode %}

Use nesting to create dotted paths:

{% code overflow="wrap" %}
```python
hp.nest(child_config, name="model")
# child_config's "learning_rate" parameter becomes "model.learning_rate"
```
{% endcode %}

## Duplicate Value Paths

These two entries spell the same final parameter path:

{% code overflow="wrap" %}
```python
values = {
    "model.learning_rate": 0.01,
    "model": {"learning_rate": 0.01},
}
```
{% endcode %}

Hypster raises even if both values are identical, because duplicate inputs make logs ambiguous.

## Missing Defaults

Every value-producing `hp.*` call requires a default as its first argument. Omitting it raises immediately:

{% code overflow="wrap" %}
```python
hp.int(name="depth")
```
{% endcode %}

Example message:

{% code overflow="wrap" %}
```text
Parameter 'depth': requires a default value as its first argument. How to fix: hp.int(<default>, name='depth')
```
{% endcode %}

## Type And Bounds Errors

Each `hp.*` call validates runtime values:

* `hp.int` accepts integral floats by default, but rejects non-integers, bool values, and precision-losing floats such as `64.5`.
* `hp.float` accepts integer values by default, but rejects non-numeric values and bool values.
* `strict=True` makes `hp.int` require real integers and `hp.float` require real floats.
* `hp.bool` requires actual `bool` values, not strings such as `"true"`.
* Numeric `min` and `max` bounds are enforced.
* `select` choices must be logging-safe scalars unless you use dictionary-backed selects.

## Complex Select Values

Do not put dictionaries or lists directly inside list-backed `select` choices:

{% code overflow="wrap" %}
```python
hp.select([{"layers": 2}, {"layers": 4}], name="model")
```
{% endcode %}

Use dictionary-backed select instead:

{% code overflow="wrap" %}
```python
hp.select(
    {
        "small": {"layers": 2},
        "large": {"layers": 4},
    },
    name="model",
)
```
{% endcode %}

The selected key is replayable, and the runtime value can still be complex.

## Reserved Execution Argument Names

`return_schema` (and the removed `return_info`) are actively guarded: passing them as execution arguments raises a `TypeError`. Other Hypster-owned keywords — `values`, `on_unknown`, `tracker` on `instantiate_with_params()`, `auto_apply` on `interact()`, `name`/`description` on `hp.nest()` — are ordinary parameters of the calling API. They bind to the API itself and are never forwarded, so a config that declares a required execution argument with one of these names fails with a plain `TypeError: config() missing 1 required keyword-only argument` and no mention of the collision. Rename the execution argument.

## Interactive Session Errors

Inside a live `interact()` session, a failed `set_value` on a **reachable** parameter is captured into `snapshot["error"]` so the widget can display it, and `result.value` / `result.params` raise `RuntimeError` while the applied state is invalid. Every action must carry `"protocol_version": 1`; a missing or mismatched version raises `ValueError` before state changes. Setting an **unreachable** path is different: `result.dispatch({"protocol_version": 1, "type": "set_value", "path": ..., "value": ...})` raises the backend's `Unknown or unreachable parameters` `ValueError` directly out of `dispatch()`.
