# Observing Past Runs

When a past run stores Hypster params, you can inspect what those params mean against the current config function.

## Inspect The Active Branch

{% code overflow="wrap" %}
```python
from hypster import HP, explore
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

def config(hp: HP) -> ClassifierMixin:
    selected_config = hp.select(model_options, name="model_family", default="linear", options_only=True)
    return hp.nest(selected_config, name="model")

past_params = {"model_family": "forest", "model.n_estimators": 500}

explore(config, values=past_params)
```
{% endcode %}

The tree shows the reachable parameters for that recorded branch.

## Handle Old Or Partial Payloads

Use `on_unknown="warn"` when reviewing old payloads that may include stale fields:

{% code overflow="wrap" %}
```python
explore(config, values={"model_family": "linear", "model.n_estimators": 500}, on_unknown="warn")
```
{% endcode %}

Do not replay with `on_unknown="ignore"` until you have decided which old values should be dropped or migrated.

## Compare Defaults

{% code overflow="wrap" %}
```python
schema = explore(config, values=past_params, return_schema=True)
current_branch_defaults = schema.defaults()
```
{% endcode %}

This is useful when you want to know which values were explicit in a past run and which current defaults would apply if they were omitted.
