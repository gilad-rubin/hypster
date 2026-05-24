# Values & Overrides

`values=` is the dictionary you pass to `instantiate()`, `instantiate_with_params()`, or `explore()` to select concrete parameter values.

{% code overflow="wrap" %}
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
{% endcode %}

## Top-Level Values

{% code overflow="wrap" %}
```python
def config(hp: HP):
    return {"batch_size": hp.int(32, name="batch_size")}

instantiate(config, values={"batch_size": 64})
# => {"batch_size": 64}
```
{% endcode %}

## Dotted Keys

Use dotted keys for nested parameters:

{% code overflow="wrap" %}
```python
instantiate(parent, values={"child.x": 15})
# => {"child": {"x": 15, "y": 20}}
```
{% endcode %}

## Nested Dictionaries

Nested dictionaries are normalized to the same dotted paths:

{% code overflow="wrap" %}
```python
instantiate(parent, values={"child": {"x": 25}})
# => {"child": {"x": 25, "y": 20}}
```
{% endcode %}

You can mix dotted keys and nested dictionaries as long as each final parameter path appears once.

## Nested Scope Names Are Not Leaves

A nested scope name is a prefix for child parameters, not a parameter leaf by itself. These forms are valid because they target `child.x`:

{% code overflow="wrap" %}
```python
instantiate(parent, values={"child.x": 15})
instantiate(parent, values={"child": {"x": 15}})
```
{% endcode %}

This raises because `child` is a scope, not a selectable parameter:

{% code overflow="wrap" %}
```python
instantiate(parent, values={"child": 123})
# ValueError: Unknown or unreachable parameters
```
{% endcode %}

## Duplicate Paths

This raises because both entries target `child.x`:

{% code overflow="wrap" %}
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
{% endcode %}

Hypster raises even when the duplicate values are identical. A single canonical path keeps experiment logs and replay payloads unambiguous.

## Conditional Reachability

Only parameters touched by the active branch may appear in `values=`.

{% code overflow="wrap" %}
```python
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def linear_model(hp: HP) -> LogisticRegression:
    C = hp.float(1.0, name="C", min=1e-4, max=100.0)
    return LogisticRegression(C=C, max_iter=1000)

def forest_model(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10)
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42)

model_options = {"linear": linear_model, "forest": forest_model}

def model_config(hp: HP) -> ClassifierMixin:
    selected_config = hp.select(model_options, name="family", default="linear", options_only=True)
    return hp.nest(selected_config, name="model")

instantiate(model_config, values={"family": "linear", "model.n_estimators": 500})
# ValueError: Unknown or unreachable parameters
```
{% endcode %}

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

{% code overflow="wrap" %}
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
{% endcode %}

Do not pass `values={"model": {"layers": 4}}`; Hypster will treat that as a nested parameter path.
