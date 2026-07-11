# Instantiate And Replay

Use `instantiate()` to execute a config function and get its returned runtime value.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def linear_model(hp: HP) -> LogisticRegression:
    C = hp.float(1.0, name="C", min=1e-4, max=100.0)
    return LogisticRegression(C=C, max_iter=1000)

def forest_model(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42)

model_options = {"linear": linear_model, "forest": forest_model}

def model_config(hp: HP) -> ClassifierMixin:
    selected_config = hp.select(model_options, name="family", default="forest", options_only=True)
    return hp.nest(selected_config, name="model")

model = instantiate(model_config, values={"family": "forest", "model.n_estimators": 500})
assert isinstance(model, RandomForestClassifier)
```
{% endcode %}

If you want to inspect a branch before running it, use [`explore()`](exploring-a-configuration-space.md).

## Log Selected Params

Use `instantiate_with_params()` when you need a replayable record for experiment tracking tools such as MLflow, Weights & Biases, or a database table. The returned `params` dictionary includes every reachable `hp.*` parameter on that run, including defaults the user did not override.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate, instantiate_with_params
from my_app.llms import OpenAIClient

def llm_config(hp: HP) -> OpenAIClient:
    model_name = hp.select(["gpt-5.5-mini", "gpt-5.5"], name="model_name")
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    return OpenAIClient(model=model_name, temperature=temperature)

run = instantiate_with_params(llm_config, values={"model_name": "gpt-5.5"})

assert run.value.model == "gpt-5.5"
assert run.params == {"model_name": "gpt-5.5", "temperature": 0.2}

replayed = instantiate(llm_config, values=run.params)
assert replayed.model == run.value.model
```
{% endcode %}

`instantiate_with_params()` accepts the same `values` and `on_unknown` arguments as `instantiate()`, plus any direct execution arguments your config requires. It does not change what your config returns; it adds a sidecar for logging and replay.

{% hint style="warning" %}
`instantiate_with_params()` reserves the keyword `tracker` for its own tracker hook. If your config needs an execution argument literally named `tracker`, rename it — passing `tracker=` binds to `instantiate_with_params()`'s tracker, not to your config.
{% endhint %}

## Unknown Parameters

Unknown or conditionally unreachable values raise by default:

{% code overflow="wrap" %}
```python
instantiate(model_config, values={"n_trees": 200})
# ValueError: Unknown or unreachable parameters:
#   - 'n_trees': Unknown parameter
#
# Run explore(config, values=...) to inspect the active branch.
# Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects.
```
{% endcode %}

Use `on_unknown="warn"` or `on_unknown="ignore"` only when you intentionally want softer handling:

{% code overflow="wrap" %}
```python
instantiate(model_config, values={"n_trees": 200}, on_unknown="warn")
```
{% endcode %}

## Dotted Keys vs Nested Dicts

Nested values can be provided with dotted keys or nested dictionaries:

{% code overflow="wrap" %}
```python
def child_config(hp: HP):
    return {"x": hp.int(1, name="x")}

def parent_config(hp: HP):
    return {"child": hp.nest(child_config, name="child")}

assert instantiate(parent_config, values={"child.x": 10}) == {"child": {"x": 10}}
assert instantiate(parent_config, values={"child": {"x": 10}}) == {"child": {"x": 10}}
```
{% endcode %}

Do not provide both forms for the same final path in the same call. Hypster raises duplicate-path errors to keep logs unambiguous. The nested scope name itself is not a leaf parameter, so `values={"child": 10}` raises as unknown or unreachable.

See [Values & Overrides](../in-depth/values-and-overrides.md) for more examples.

## Nullable Parameters

`None` is an explicit value in Hypster, not a marker for "not selected". Use `allow_none=True` when `None` is part of a parameter's domain:

{% code overflow="wrap" %}
```python
def config(hp: HP):
    max_depth = hp.int(None, name="max_depth", min=1, max=100, allow_none=True)
    thinking_level = hp.select(
        [None, "low", "medium", "high"],
        name="thinking_level",
        default=None,
        allow_none=True,
    )
    return {"max_depth": max_depth, "thinking_level": thinking_level}
```
{% endcode %}

Nullable elements are supported for `multi_select(..., allow_none=True)`. They are not supported for `multi_int`, `multi_float`, `multi_text`, or `multi_bool`; those calls raise with guidance if `allow_none=True` is passed.

Nullable choices are captured and replayed like any other selected parameter:

{% code overflow="wrap" %}
```python
from hypster import instantiate_with_params

run = instantiate_with_params(config, values={"thinking_level": None})

assert run.value == {"max_depth": None, "thinking_level": None}
assert run.params["thinking_level"] is None
assert instantiate(config, values=run.params) == run.value
```
{% endcode %}

## Passing Execution Arguments To Nested Configs

When using `hp.nest`, pass child execution arguments directly as keyword arguments.

{% code overflow="wrap" %}
```python
def child(hp: HP, multiplier: int, offset: int = 0) -> int:
    base = hp.int(5, name="base")
    return base * multiplier + offset

def parent(hp: HP):
    calc1 = hp.nest(child, name="calc1", multiplier=2)
    calc2 = hp.nest(child, name="calc2", multiplier=3, offset=10)
    return {"calc1": calc1, "calc2": calc2}

result = instantiate(parent)
assert result == {"calc1": 10, "calc2": 25}
```
{% endcode %}
