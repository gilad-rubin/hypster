# Observing Past Runs

When a past run stores Hypster params, you can inspect what those params mean against the current config function.

## Inspect The Active Branch

{% code overflow="wrap" %}
```python
from hypster import HP, explore

def config(hp: HP):
    model = hp.select(["linear", "forest"], name="model", default="linear", options_only=True)

    if model == "forest":
        return {"model": model, "n_estimators": hp.int(200, name="n_estimators", min=10)}

    return {"model": model, "alpha": hp.float(0.1, name="alpha", min=0.0, max=10.0)}

past_params = {"model": "forest", "n_estimators": 500}

explore(config, values=past_params)
```
{% endcode %}

The tree shows the reachable parameters for that recorded branch.

## Handle Old Or Partial Payloads

Use `on_unknown="warn"` when reviewing old payloads that may include stale fields:

{% code overflow="wrap" %}
```python
explore(config, values={"model": "linear", "n_estimators": 500}, on_unknown="warn")
```
{% endcode %}

Do not replay with `on_unknown="ignore"` until you have decided which old values should be dropped or migrated.

## Compare Defaults

{% code overflow="wrap" %}
```python
schema = explore(config, values=past_params, return_info=True)
current_branch_defaults = schema.defaults()
```
{% endcode %}

This is useful when you want to know which values were explicit in a past run and which current defaults would apply if they were omitted.
