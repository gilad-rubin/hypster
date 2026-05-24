# Values & Overrides

`values=` is the dictionary you pass to `instantiate()`, `instantiate_with_params()`, or `explore()` to select concrete parameter values.

```python
from hypster import HP, instantiate

def child(hp: HP):
    return {
        "x": hp.int(10, name="x"),
        "y": hp.int(20, name="y"),
    }

def parent(hp: HP):
    return {"child": hp.nest(child, name="child")}
```

## Top-Level Values

```python
def config(hp: HP):
    return {"batch_size": hp.int(32, name="batch_size")}

instantiate(config, values={"batch_size": 64})
# => {"batch_size": 64}
```

## Dotted Keys

Use dotted keys for nested parameters:

```python
instantiate(parent, values={"child.x": 15})
# => {"child": {"x": 15, "y": 20}}
```

## Nested Dictionaries

Nested dictionaries are normalized to the same dotted paths:

```python
instantiate(parent, values={"child": {"x": 25}})
# => {"child": {"x": 25, "y": 20}}
```

You can mix dotted keys and nested dictionaries as long as each final parameter path appears once.

## Nested Scope Names Are Not Leaves

A nested scope name is a prefix for child parameters, not a parameter leaf by itself. These forms are valid because they target `child.x`:

```python
instantiate(parent, values={"child.x": 15})
instantiate(parent, values={"child": {"x": 15}})
```

This raises because `child` is a scope, not a selectable parameter:

```python
instantiate(parent, values={"child": 123})
# ValueError: Unknown or unreachable parameters
```

## Duplicate Paths

This raises because both entries target `child.x`:

```python
instantiate(
    parent,
    values={
        "child.x": 100,
        "child": {"x": 100},
    },
)
# ValueError: Duplicate value for 'child.x'
```

Hypster raises even when the duplicate values are identical. A single canonical path keeps experiment logs and replay payloads unambiguous.

## Conditional Reachability

Only parameters touched by the active branch may appear in `values=`.

```python
def model_config(hp: HP):
    family = hp.select(["linear", "forest"], name="family", default="linear", options_only=True)

    if family == "linear":
        return {"alpha": hp.float(0.1, name="alpha", min=0.0, max=10.0)}

    return {"n_estimators": hp.int(200, name="n_estimators", min=10)}

instantiate(model_config, values={"family": "linear", "n_estimators": 500})
# ValueError: Unknown or unreachable parameters
```

Use `explore(model_config, values={"family": "forest"})` to inspect the branch before instantiating it.

## Unknown Policies

`instantiate()`, `instantiate_with_params()`, and `explore()` accept the same `on_unknown` policy:

| Policy | Behavior |
| --- | --- |
| `"raise"` | Default. Raise on unknown or unreachable values. |
| `"warn"` | Emit a warning and continue. |
| `"ignore"` | Ignore unknown or unreachable values. |

Prefer the default for experiments and production replay. Softer policies are useful when migrating old payloads or rendering exploratory UIs.

## Select Keys vs Complex Values

Nested dictionaries inside `values=` are interpreted as nested parameter paths. If you need a select option whose runtime value is a dictionary, use dict-backed `select`:

```python
def config(hp: HP):
    return hp.select(
        {
            "small": {"layers": 2},
            "large": {"layers": 4},
        },
        name="model",
        default="small",
    )

instantiate(config, values={"model": "large"})
# => {"layers": 4}
```

Do not pass `values={"model": {"layers": 4}}`; Hypster will treat that as a nested parameter path.
