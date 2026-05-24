# Create A Replayable Config

Use this recipe when you want a small config that can be inspected, instantiated, logged, and replayed later.

The config is a normal Python function. Keep it cheap and side-effect-free so `explore()` can run it to inspect parameters and replay tools can execute it safely.

## 1. Define A Typed Config

{% code overflow="wrap" %}
```python
from dataclasses import dataclass

from hypster import HP


@dataclass(frozen=True)
class DataLoadSettings:
    path: str
    batch_size: int
    shuffle: bool


def data_config(hp: HP) -> DataLoadSettings:
    return DataLoadSettings(
        path=hp.text("data/train.csv", name="path"),
        batch_size=hp.int(64, name="batch_size", min=1),
        shuffle=hp.bool(True, name="shuffle"),
    )
```
{% endcode %}

## 2. Inspect The Parameters

Print the active tree when you want a quick human check:

{% code overflow="wrap" %}
```python
from hypster import explore

explore(data_config)
```
{% endcode %}

Use structured metadata when another tool needs to render fields:

{% code overflow="wrap" %}
```python
schema = explore(data_config, return_info=True)
fields = schema.to_dict()["parameters"]
```
{% endcode %}

## 3. Instantiate With Overrides

{% code overflow="wrap" %}
```python
from hypster import instantiate

settings = instantiate(data_config, values={"batch_size": 128})

assert settings == DataLoadSettings(
    path="data/train.csv",
    batch_size=128,
    shuffle=True,
)
```
{% endcode %}

## 4. Capture Params For Replay

{% code overflow="wrap" %}
```python
from hypster import instantiate_with_params

run = instantiate_with_params(data_config, values={"batch_size": 128})

assert run.value == settings
assert run.params == {
    "path": "data/train.csv",
    "batch_size": 128,
    "shuffle": True,
}
```
{% endcode %}

## 5. Store And Replay

{% code overflow="wrap" %}
```python
import json

payload = json.dumps(run.params, sort_keys=True)
restored_params = json.loads(payload)

replayed = instantiate(data_config, values=restored_params)
assert replayed == run.value
```
{% endcode %}

Because `run.params` includes defaulted values, replay does not silently change if the config's defaults are edited later.
