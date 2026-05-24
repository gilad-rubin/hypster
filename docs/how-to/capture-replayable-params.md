# Capture Replayable Params

Use `instantiate_with_params()` when a run needs a durable parameter record.

## Capture Params

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate, instantiate_with_params

def report_config(hp: HP):
    return {
        "audience": hp.select(["exec", "technical"], name="audience", default="technical", options_only=True),
        "include_appendix": hp.bool(True, name="include_appendix"),
        "max_pages": hp.int(12, name="max_pages", min=1, max=100),
    }

run = instantiate_with_params(
    report_config,
    values={"audience": "exec", "max_pages": 6},
)

assert run.value == {
    "audience": "exec",
    "include_appendix": True,
    "max_pages": 6,
}
assert run.params == {
    "audience": "exec",
    "include_appendix": True,
    "max_pages": 6,
}
```
{% endcode %}

## Replay Later

{% code overflow="wrap" %}
```python
replayed = instantiate(report_config, values=run.params)
assert replayed == run.value
```
{% endcode %}

Captured params include defaults, so replay does not silently pick up later default changes:

{% code overflow="wrap" %}
```python
def old_config(hp: HP):
    return {"batch_size": hp.int(64, name="batch_size")}


old_run = instantiate_with_params(old_config)
assert old_run.params == {"batch_size": 64}


def new_config(hp: HP):
    return {"batch_size": hp.int(128, name="batch_size")}


assert instantiate(new_config, values=old_run.params) == {"batch_size": 64}
```
{% endcode %}

## Store Params As JSON

`run.params` only contains values selected by `hp.*` calls. It is intended to be JSON-friendly when your parameter values are JSON-friendly.

{% code overflow="wrap" %}
```python
import json

payload = json.dumps(run.params, sort_keys=True)
restored_params = json.loads(payload)

assert instantiate(report_config, values=restored_params) == run.value
```
{% endcode %}

## Complex Runtime Objects

Use dict-backed `select` when a runtime choice is a complex object. The params record the simple key, not the complex mapped value.

{% code overflow="wrap" %}
```python
def model_config(hp: HP):
    return hp.select(
        {
            "small": {"layers": 2, "units": [64, 32]},
            "large": {"layers": 4, "units": [256, 128]},
        },
        name="model",
        default="small",
    )

run = instantiate_with_params(model_config, values={"model": "large"})

assert run.value == {"layers": 4, "units": [256, 128]}
assert run.params == {"model": "large"}
```
{% endcode %}
