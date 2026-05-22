# 🔧 Values & Overrides

This page explains how to provide overrides to a configuration function using dotted keys and nested dictionaries.

## Basics

A configuration function receives an `HP` object and returns anything (dict, dataclass, model, etc.). Parameters are named explicitly using `name=...`:

```python
from hypster import HP, instantiate

def child(hp: HP):
    x = hp.int(10, name="x")
    y = hp.int(20, name="y")
    return {"x": x, "y": y}

def parent(hp: HP):
    return hp.nest(child, name="child")
```

## Dotted keys

Use dotted keys to override nested parameters:

```python
instantiate(parent, values={"child.x": 15})
# => {"x": 15, "y": 20}
```

## Nested dicts

Provide a nested dict under the nested scope name:

```python
instantiate(parent, values={"child": {"x": 25}})
# => {"x": 25, "y": 20}
```

## Duplicate paths

Dotted keys and nested dictionaries are two ways to spell the same parameter path. Do not provide the same path twice. Hypster raises even when both values are identical, because duplicate input is ambiguous for logging and replay.

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

## Deeply nested

Dotted keys can target deeper levels, and nested dictionaries can express the same shape. You can mix both forms as long as each final parameter path appears once.

```python
instantiate(parent, values={"child": {"x": 25}, "other.y": 30})
```

## Unknown and unreachable parameters

* Unknown or unreachable values (values for parameters on a branch that wasn’t taken) are handled by `on_unknown` in `instantiate`:
  * `on_unknown="raise"` (default): raise an error
  * `on_unknown="warn"`: issue a warning
  * `on_unknown="ignore"`: silently ignore

Nested dictionaries participate in the same unknown/unreachable check as dotted paths. Hypster includes helpful suggestions when a name looks like a typo, and errors guide you to run `explore(config, values=...)` to inspect the active branch.
