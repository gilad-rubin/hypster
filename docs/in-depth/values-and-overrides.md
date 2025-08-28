# 🔧 Values & Overrides

This page explains how to provide overrides to a configuration function using dotted keys and nested dictionaries, and which one takes precedence when both are provided.

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

## Precedence: nested dict wins

If you provide both dotted keys and a nested dict for the same scope, the nested dict wins.

```python
instantiate(
    parent,
    values={
        "child.x": 100,         # dotted notation
        "child": {"x": 200},  # nested dict
    },
)
# => {"x": 200, "y": 20}
```

## Deeply nested

Dotted keys can target deeper levels when there is no nested dict override at that level. You can mix and match as needed, keeping in mind that a nested dict for a scope overrides dotted keys for that scope.

## Unknown and unreachable parameters

* Unknown or unreachable values (values for parameters on a branch that wasn’t taken) are handled by `on_unknown` in `instantiate`:
  * `on_unknown="warn"` (default): issue a warning
  * `on_unknown="raise"`: raise an error
  * `on_unknown="ignore"`: silently ignore

Hypster will include helpful suggestions when a name looks like a typo.
