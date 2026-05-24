# Error Handling

Hypster validates names, values, paths, and branch reachability before it treats a run as replayable.

## Public Exception Surface

Public validation failures currently raise `ValueError`. When `on_unknown="warn"`, unknown or unreachable values emit a `UserWarning` and the run continues. Hypster does not expose a structured exception hierarchy today, so application and UI code should catch `ValueError` at the boundary where it can show a user-facing message.

## Unknown Or Unreachable Values

`instantiate()`, `instantiate_with_params()`, and `explore()` default to `on_unknown="raise"`.

```python
from hypster import HP, instantiate

def config(hp: HP):
    mode = hp.select(["a", "b"], name="mode", default="a")
    if mode == "a":
        return {"count": hp.int(1, name="count")}
    return {"mode": mode}

instantiate(config, values={"mode": "b", "count": 3})
```

`count` is unreachable on the `mode="b"` branch, so Hypster raises.

Example message:

```text
Unknown or unreachable parameters:
  - 'count': Unknown parameter

Run explore(config, values=...) to inspect the active branch.
Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects.
```

Unknown and unreachable values share the same public error family because both mean "this key was not consumed by the active run." To diagnose the difference, run `explore(config, values=...)` for the selected branch:

* If the path is absent from that branch but present in another branch, it is unreachable or stale.
* If the path is absent from all branches, it is likely a typo or removed parameter.

A typo can include a nearest-name suggestion:

```python
instantiate(config, values={"mode": "a", "coutn": 3})
```

```text
Unknown or unreachable parameters:
  - 'coutn': Did you mean 'count'? (similarity: 80%)

Run explore(config, values=...) to inspect the active branch.
Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects.
```

Use softer policies only when intentional:

```python
instantiate(config, values={"mode": "b", "count": 3}, on_unknown="warn")
instantiate(config, values={"mode": "b", "count": 3}, on_unknown="ignore")
```

With `on_unknown="warn"`, use Python warning controls if you need to capture the message in a UI or test:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    instantiate(config, values={"mode": "b", "count": 3}, on_unknown="warn")

assert caught
```

For user-facing tools, a common strategy is:

```python
try:
    value = instantiate(config, values=form_values)
except ValueError as exc:
    show_validation_error(str(exc))
else:
    run_workflow(value)
```

Interactive widget handles expose the same validation through a different boundary: direct `instantiate()` and `explore()` failures raise `ValueError`, while reading `InteractiveResult.value` or `InteractiveResult.params` during an invalid applied state raises `RuntimeError` with the underlying validation message. Show either message to the user and keep the last valid submitted params separate from draft UI state.

## Invalid Names

Parameter and nest names must be Python identifier-style strings.

```python
hp.int(32, name="batch_size")  # valid
hp.int(32, name="batch-size")  # invalid
```

Use nesting to create dotted paths:

```python
hp.nest(child_config, name="model")
# child_config's "learning_rate" parameter becomes "model.learning_rate"
```

## Duplicate Value Paths

These two entries spell the same final parameter path:

```python
values = {
    "model.learning_rate": 0.01,
    "model": {"learning_rate": 0.01},
}
```

Hypster raises even if both values are identical, because duplicate inputs make logs ambiguous.

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

```python
hp.select([{"layers": 2}, {"layers": 4}], name="model")
```

Use dictionary-backed select instead:

```python
hp.select(
    {
        "small": {"layers": 2},
        "large": {"layers": 4},
    },
    name="model",
)
```

The selected key is replayable, and the runtime value can still be complex.
