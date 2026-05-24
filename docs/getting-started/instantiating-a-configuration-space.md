# Instantiate And Replay

Use `instantiate()` to execute a config function and get its returned runtime value.

```python
from hypster import HP, instantiate

def model_config(hp: HP):
    family = hp.select(["linear", "forest"], name="family", default="forest", options_only=True)

    if family == "linear":
        return {"family": family, "alpha": hp.float(0.1, name="alpha", min=0.0, max=10.0)}

    return {"family": family, "n_estimators": hp.int(200, name="n_estimators", min=10, max=1000)}

cfg = instantiate(model_config, values={"family": "forest", "n_estimators": 500})
assert cfg == {"family": "forest", "n_estimators": 500}
```

If you want to inspect a branch before running it, use [`explore()`](exploring-a-configuration-space.md).

## Log Selected Params

Use `instantiate_with_params()` when you need a replayable record for experiment tracking tools such as MLflow, Weights & Biases, or a database table. The returned `params` dictionary includes every reachable `hp.*` parameter on that run, including defaults the user did not override.

```python
from hypster import HP, instantiate, instantiate_with_params

def llm_config(hp: HP):
    provider = hp.select(["gemini", "openai"], name="provider", default="gemini", options_only=True)
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    return {"provider": provider, "temperature": temperature}

run = instantiate_with_params(llm_config, values={"provider": "openai"})

assert run.value == {"provider": "openai", "temperature": 0.2}
assert run.params == {"provider": "openai", "temperature": 0.2}

replayed = instantiate(llm_config, values=run.params)
assert replayed == run.value
```

`instantiate_with_params()` accepts the same `values`, `args`, `kwargs`, and `on_unknown` arguments as `instantiate()`. It does not change what your config returns; it adds a sidecar for logging and replay.

## Unknown Parameters

Unknown or conditionally unreachable values raise by default:

```python
instantiate(model_config, values={"n_trees": 200})
# ValueError: Unknown or unreachable parameters:
#   - 'n_trees': Unknown parameter
#
# Run explore(config, values=...) to inspect the active branch.
# Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects.
```

Use `on_unknown="warn"` or `on_unknown="ignore"` only when you intentionally want softer handling:

```python
instantiate(model_config, values={"n_trees": 200}, on_unknown="warn")
```

## Dotted Keys vs Nested Dicts

Nested values can be provided with dotted keys or nested dictionaries:

```python
def child_config(hp: HP):
    return {"x": hp.int(1, name="x")}

def parent_config(hp: HP):
    return {"child": hp.nest(child_config, name="child")}

assert instantiate(parent_config, values={"child.x": 10}) == {"child": {"x": 10}}
assert instantiate(parent_config, values={"child": {"x": 10}}) == {"child": {"x": 10}}
```

Do not provide both forms for the same final path in the same call. Hypster raises duplicate-path errors to keep logs unambiguous. The nested scope name itself is not a leaf parameter, so `values={"child": 10}` raises as unknown or unreachable.

See [Values & Overrides](../in-depth/values-and-overrides.md) for more examples.

## Nullable Parameters

`None` is an explicit value in Hypster, not a marker for "not selected". Use `allow_none=True` when `None` is part of a parameter's domain:

```python
def config(hp: HP):
    max_depth = hp.int(None, name="max_depth", allow_none=True)
    thinking_level = hp.select(
        [None, "low", "medium", "high"],
        name="thinking_level",
        default=None,
        allow_none=True,
    )
    return {"max_depth": max_depth, "thinking_level": thinking_level}
```

Nullable elements are supported for `multi_select(..., allow_none=True)`. They are not supported for `multi_int`, `multi_float`, `multi_text`, or `multi_bool`; those calls raise with guidance if `allow_none=True` is passed.

Nullable choices are captured and replayed like any other selected parameter:

```python
from hypster import instantiate_with_params

run = instantiate_with_params(config, values={"thinking_level": None})

assert run.value == {"max_depth": None, "thinking_level": None}
assert run.params["thinking_level"] is None
assert instantiate(config, values=run.params) == run.value
```

## Passing Args/Kwargs To Nested Configs

When using `hp.nest`, you can pass `args=` and `kwargs=` to the child function.

```python
def child(hp: HP, multiplier: int, offset: int = 0) -> int:
    base = hp.int(5, name="base")
    return base * multiplier + offset

def parent(hp: HP):
    calc1 = hp.nest(child, name="calc1", args=(2,))
    calc2 = hp.nest(child, name="calc2", args=(3,), kwargs={"offset": 10})
    return {"calc1": calc1, "calc2": calc2}

result = instantiate(parent)
assert result == {"calc1": 10, "calc2": 25}
```
