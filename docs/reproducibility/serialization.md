# Serialization

Hypster does not define a custom serialization format. The recommended reproducibility artifact is the `params` dictionary returned by `instantiate_with_params()`.

## JSON Params

{% code overflow="wrap" %}
```python
import json
from hypster import HP, explore, instantiate, instantiate_with_params
from my_app.llms import LLMClient

def config(hp: HP) -> LLMClient:
    provider = hp.select(["openai", "gemini"], name="provider", default="openai", options_only=True)
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    return LLMClient(provider=provider, temperature=temperature)

run = instantiate_with_params(config, values={"provider": "gemini"})
payload = json.dumps(run.params, sort_keys=True)
restored = json.loads(payload)

assert instantiate(config, values=restored).provider == run.value.provider
```
{% endcode %}

## Complex Runtime Values

Use dict-backed `select` so the serialized params contain simple keys:

{% code overflow="wrap" %}
```python
from my_app.models import LargeMLP, SmallMLP

model_options = {
    "small": SmallMLP,
    "large": LargeMLP,
}

def model_config(hp: HP):
    model_cls = hp.select(model_options, name="model", default="small", options_only=True)
    return model_cls()
```
{% endcode %}

`instantiate_with_params(model_config, values={"model": "large"}).params` records `{"model": "large"}`, not the mapped class.

## Schema Serialization

`explore(config, return_schema=True).to_dict()` returns JSON-serializable schema metadata for UIs, catalogs, and validation tools.

{% code overflow="wrap" %}
```python
schema = explore(config, return_schema=True).to_dict()
json.dumps(schema)
```
{% endcode %}

Schema metadata is not a replacement for selected params. Use schema for rendering and params for replay.

## Versioned Replay Artifacts

When params leave the current process, store them with enough identity to understand which code and data produced the original run:

{% code overflow="wrap" %}
```python
import json
import hypster

artifact = {
    "kind": "hypster-run-params",
    "hypster_version": hypster.__version__,
    "config_name": "config",
    "app_version": "2026.05.24",
    "git_commit": "abc1234",
    "dataset_id": "warehouse/churn/2026-05-01",
    "params": run.params,
}

payload = json.dumps(artifact, sort_keys=True)
restored = json.loads(payload)

replayed = instantiate(config, values=restored["params"])
```
{% endcode %}

If replay fails after the config evolves, inspect the old payload with `explore(config, values=restored["params"], on_unknown="warn")`, migrate the parameter names deliberately, and save the migrated artifact as a new record.

## Replay After Defaults Change

The versioned artifact still protects you when defaults change, because replay uses stored params:

{% code overflow="wrap" %}
```python
import json
import hypster
from hypster import HP, instantiate, instantiate_with_params

def training_config(hp: HP) -> int:
    return hp.int(64, name="batch_size")


old_run = instantiate_with_params(training_config)
artifact = {
    "kind": "hypster-run-params",
    "hypster_version": hypster.__version__,
    "config_name": "training_config",
    "app_version": "2026.05.24",
    "git_commit": "abc1234",
    "params": old_run.params,
}

payload = json.dumps(artifact, sort_keys=True)
restored = json.loads(payload)


def training_config(hp: HP) -> int:
    return hp.int(128, name="batch_size")


assert instantiate(training_config, values=restored["params"]) == 64
```
{% endcode %}
