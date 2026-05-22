# ⚡ Instantiating a Config Function

Use `instantiate(func, values=...)` to execute a configuration function with optional overrides.

If you want to inspect the active parameters first, use [`explore()`](exploring-a-configuration-space.md).

```python
from hypster import instantiate

result = instantiate(model_cfg, values={
    "kind": "rf",
    "n_estimators": 200,
    "max_depth": 12.5,
})
# => {"model": ("rf", 200, 12.5)}
```

## Logging selected params

Use `instantiate_with_params()` when you need a replayable record for experiment tracking tools such as MLflow. The returned `params` dictionary includes every reachable `hp.*` parameter on that run, including defaults the user did not override.

```python
from hypster import HP, instantiate, instantiate_with_params

def llm_config(hp: HP):
    provider = hp.select(["gemini", "openai"], name="provider", default="gemini")
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    return {"provider": provider, "temperature": temperature}

run = instantiate_with_params(llm_config, values={"provider": "openai"})

run.value
# => {"provider": "openai", "temperature": 0.2}

run.params
# => {"provider": "openai", "temperature": 0.2}

replayed = instantiate(llm_config, values=run.params)
assert replayed == run.value
```

`instantiate_with_params()` accepts the same `values`, `args`, `kwargs`, and `on_unknown` arguments as `instantiate()`. It does not change what your config returns; it adds a sidecar for logging and replay.

## Unknown parameters

Unknown or conditionally unreachable values raise by default:

```python
instantiate(model_cfg, values={"n_trees": 200})
# ValueError: Unknown or unreachable parameters:
#   - 'n_trees': Unknown parameter
#
# Run explore(config, values=...) to inspect the active branch.
# Nested dict values are interpreted as parameter paths; use dict-backed select keys for objects.
```

Use `on_unknown="warn"` or `on_unknown="ignore"` only when you intentionally want softer handling:

```python
instantiate(model_cfg, values={"n_trees": 200}, on_unknown="warn")
```

## Dotted keys vs nested dicts

See [Values & Overrides](../in-depth/values-and-overrides.md) for dotted paths, nested dicts, and duplicate-key validation.

The same `values=` format also works with `explore()` when you want to inspect a specific conditional branch.

## Nullable parameters

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

## Passing args/kwargs to nested configs

When using `hp.nest`, you can pass `args=`/`kwargs=` to the child function; these are forwarded at call time.

```python

def child(hp: HP, multiplier: int, offset: int = 0) -> int:
    base = hp.int(5, name="base")
    return base * multiplier + offset


def parent(hp: HP):
    calc1 = hp.nest(child, name="calc1", args=(2,))
    calc2 = hp.nest(child, name="calc2", args=(3,), kwargs={"offset": 10})
    return {"calc1": calc1, "calc2": calc2}
```
